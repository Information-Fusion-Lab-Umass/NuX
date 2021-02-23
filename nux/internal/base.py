from functools import partial, wraps
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
import haiku as hk
from abc import ABC, abstractmethod
from typing import Optional, Mapping, Type, Callable, Iterable, Any, Sequence, Union, Tuple, MutableMapping, NamedTuple, Set, TypeVar
import collections
import contextlib

__all__ = ["get_constant",
           "get_tree_shapes"]

""" This file is largely adapted from haiku._src.base """

from haiku._src.base import ThreadLocalStack, \
                            MutableParams, \
                            MutableState, \
                            PRNGSequence, \
                            Frame, \
                            frame_stack, \
                            extract_state, \
                            Stack, \
                            ModuleState, \
                            StatePair, \
                            current_frame, \
                            current_bundle_name
from haiku._src.typing import PRNGKey, Params, State
from haiku._src import data_structures

# Probably a bad name
Constant = Any
MutableConstant = MutableMapping[str, MutableMapping[str, Constant]]

# Called FrameData in Haiku.  Don't want to confuse name with the frame's state
FrameData = collections.namedtuple("FrameData", "params,state,constants,rng")

class CustomFrame(NamedTuple):
  """A frame represents all of the per-transform values in NuX.
     Adapted from the Frame class in Haiku
  """

  # JAX values.  This is the core frame data
  params: Union[Params, MutableParams]
  state: Optional[MutableState]
  constants: Optional[MutableConstant] # <--- This is the difference between the Frame in Haiku and here
  rng_stack: Stack[Optional["PRNGSequence"]]

  # Pure python values.
  module_stack: Stack[ModuleState]
  counter_stack: Stack[collections.Counter]
  used_names_stack: Stack[Set[str]]

  @property
  def params_frozen(self):
    return isinstance(self.params, data_structures.FlatMapping)

  @classmethod
  def create_from_frame_data(cls, frame_data):
    rng = frame_data.rng if frame_data.rng is None else PRNGSequence(frame_data.rng)
    params = frame_data.params
    state = frame_data.state
    constants = frame_data.constants
    return CustomFrame.create(params, state, constants, rng)

  @classmethod
  def create_from_params_and_state(cls, params, bundled_state):
    (state, constants, rng) = bundled_state
    rng = rng if rng is None else PRNGSequence(rng)
    return CustomFrame.create(params, state, constants, rng)

  @classmethod
  def create(cls, params, state, constants, rng: Optional["PRNGSequence"]):
    """Creates a new frame."""
    frame = CustomFrame(params=params,
                        state=state,
                        constants=constants,
                        rng_stack=Stack(),
                        module_stack=Stack(),
                        counter_stack=Stack(),
                        used_names_stack=Stack())
    frame.rng_stack.push(rng)
    frame.counter_stack.push(collections.Counter())
    frame.used_names_stack.push(set())
    return frame

  def evolve(self, params, state, constants, rng):
    # Copy this frame, but replace the frame's data
    rng_stack = self.rng_stack.clone()
    rng_stack.push(rng)
    return CustomFrame(params=params,
                       state=state,
                       constants=constants,
                       rng_stack=rng_stack,
                       module_stack=self.module_stack.clone(),
                       counter_stack=self.counter_stack.clone(),
                       used_names_stack=self.used_names_stack.clone())

  @contextlib.contextmanager
  def module(self, module_state: ModuleState):
    with self.module_stack(module_state), \
         self.counter_stack(collections.Counter()), \
         self.used_names_stack(set()):
      yield

################################################################################################################

class CustomHaikuContext(object):
  """Collects and injects values for computations."""

  __slots__ = ("__params", "__state", "__constants", "__rng",
               "__expected_stack", "__names", "__counter")

  def __init__(self,
               params: Union[Params, MutableParams],
               state: Union[State, MutableState],
               constants: Union[Constant, MutableConstant],
               rng: Optional["PRNGSequence"],
  ):
    self.__params = params
    self.__state = state
    self.__constants = constants
    self.__rng = rng
    self.__expected_stack = ThreadLocalStack()
    self.__names = set()
    self.__counter = collections.Counter()

  def collect_params(self) -> Params:
    return data_structures.to_immutable_dict(self.__params)

  def collect_initial_state(self) -> State:
    return extract_state(self.__state, initial=True)

  def collect_state(self) -> State:
    return extract_state(self.__state, initial=False)

  def collect_constants(self) -> State:
    return data_structures.to_immutable_dict(self.__constants)

  def __enter__(self):
    frame = CustomFrame.create(params=self.__params, state=self.__state, constants=self.__constants, rng=self.__rng)
    frame.used_names_stack.push(self.__names)
    frame.counter_stack.push(self.__counter)
    self.__expected_stack.push(frame)
    frame_stack.push(frame)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    actual = frame_stack.pop()
    expected = self.__expected_stack.pop()
    assert actual is expected

################################################################################################################

def new_custom_context(*,
                params: Optional[Params] = None,
                state: Optional[State] = None,
                constants: Optional[State] = None,
                rng: Optional[Union[PRNGKey, int]] = None,
) -> CustomHaikuContext:

  if params is None:
    params = collections.defaultdict(dict)
  else:
    params = data_structures.to_immutable_dict(params)

  if state is None:
    state = collections.defaultdict(dict)
  else:
    state = {m: {k: StatePair(v, v) for k, v in p.items()}
             for m, p in state.items()}

  if constants is None:
    constants = collections.defaultdict(dict)
  else:
    constants = data_structures.to_immutable_dict(constants)

  if rng is not None and not isinstance(rng, PRNGSequence):
    rng = PRNGSequence(rng)

  return CustomHaikuContext(params, state, constants, rng)

################################################################################################################

def get_constant(name: str,
                 value: Any=None,
                 init=None,
                 do_not_set=False):
  constants = current_frame().constants[current_bundle_name()]
  saved_value = constants.get(name, None)
  if saved_value is None:
    if do_not_set:
      return None

    if init is not None:
      value = init(value)
      constants[name] = value
    else:
      constants[name] = value
  else:
    assert name in constants, f"Missing {name} in constants"
    value = saved_value

  return value

################################################################################################################

def get_tree_shapes(name: str,
                    pytree: Any=None,
                    batch_axes: Optional[Sequence[int]] = (),
                    do_not_set: Optional[bool] = False,
) -> Any:

  def get_unbatched_shape(x):
    x_shape = [s for i, s in enumerate(x.shape) if i not in batch_axes]
    x_shape = tuple(x_shape)
    return x_shape

  def apply_get_shapes(x):
    return jax.tree_map(get_unbatched_shape, x)

  return get_constant(name, pytree, init=apply_get_shapes, do_not_set=do_not_set)
