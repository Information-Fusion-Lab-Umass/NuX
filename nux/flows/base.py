from functools import partial
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
import haiku as hk
from abc import ABC, abstractmethod
from typing import Optional, Mapping, Type, Callable, Iterable, Any, Sequence, Union, Tuple, MutableMapping, NamedTuple, Set
import nux.util as util
import collections
import contextlib

__all__ = ["transform_flow",
           "Layer"]

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
                            MutableParams, \
                            MutableState
from haiku._src.typing import PRNGKey, Params, State
from haiku._src import data_structures

# Probably a bad name
Constant = Any
MutableConstant = MutableMapping[str, MutableMapping[str, Constant]]

class CustomFrame(NamedTuple):
  """A frame represents all of the per-transform values in Haiku."""

  # JAX values.
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

  def evolve(self, params, state, rng):
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

from haiku._src.transform import TransformedWithState, to_prng_sequence, check_mapping, INIT_RNG_ERROR, APPLY_RNG_STATE_ERROR, APPLY_RNG_ERROR

def transform_flow(create_fun) -> TransformedWithState:

  # We will keep the expected shapes for the flow here so that
  # JAX will compile these constants
  constants = None

  def init_fn(rng: Optional[Union[PRNGKey]],
              inputs: Mapping[str, jnp.ndarray],
              batch_axes=(),
              **kwargs,
  ) -> Tuple[Params, State]:
    """ Initializes your function collecting parameters and state. """
    rng = to_prng_sequence(rng, err_msg=INIT_RNG_ERROR)
    with new_custom_context(rng=rng) as ctx:
      # Create the model
      model = create_fun()

      # Load the batch axes for the inputs
      Layer.batch_axes = batch_axes

      key = hk.next_rng_key()

      # Initialize the model
      outputs = model(inputs, key, **kwargs)

      # Unset the batch axes
      Layer.batch_axes = ()

    nonlocal constants
    params, state, constants = ctx.collect_params(), ctx.collect_initial_state(), ctx.collect_constants()
    return params, state

  def apply_fn(params: Optional[Params],
               state: Optional[State],
               rng: Optional[Union[PRNGKey]],
               inputs,
               **kwargs,
  ) -> Tuple[Any, State]:
    """ Applies your function injecting parameters and state. """
    params = check_mapping("params", params)
    state = check_mapping("state", state)

    rng = to_prng_sequence(rng, err_msg=(APPLY_RNG_STATE_ERROR if state else APPLY_RNG_ERROR))
    with new_custom_context(params=params, state=state, constants=constants, rng=rng) as ctx:
      model = create_fun()
      key = hk.next_rng_key()
      out = model(inputs, key, **kwargs)
    return out, ctx.collect_state()

  return TransformedWithState(init_fn, apply_fn)

################################################################################################################

from haiku._src.base import current_frame, current_bundle_name, StatePair

def get_constant(name: str, value: Any, init=None, do_not_set=False):
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

def get_tree_shapes(name: str,
                    pytree: Any,
                    batch_axes: Optional[Sequence[int]] = (),
                    do_not_set: Optional[bool] = False,
) -> Any:

  def get_unbatched_shape(x):
    x_shape = [s for i, s in enumerate(x.shape) if i not in batch_axes]
    x_shape = tuple(x_shape)
    return x_shape

  def apply_get_shapes(x):
    return jax.tree_map(get_unbatched_shape, x)
    # return partial(jax.tree_map, get_unbatched_shape)(x)

  return get_constant(name, pytree, init=apply_get_shapes, do_not_set=do_not_set)

################################################################################################################

class Layer(hk.Module, ABC):

  batch_axes = ()

  def __init__(self, name=None):
    """ This base class will keep track of the input and output shapes of each function call
        so that we can know the batch size of inputs and automatically use vmap to make unbatched
        code work with batched code.
    """
    super().__init__(name=name)

  def get_unbatched_shapes(self, sample):
    if sample == False:
      return self.unbatched_input_shapes
    else:
      return self.unbatched_output_shapes

  def __call__(self,
               inputs: Mapping[str, jnp.ndarray],
               rng: jnp.ndarray=None,
               sample: Optional[bool]=False,
               **kwargs
    ) -> Mapping[str, jnp.ndarray]:

    batch_axes = Layer.batch_axes

    if sample == False:
      self.unbatched_input_shapes = get_tree_shapes("unbatched_input_shapes", inputs, batch_axes=batch_axes)
      self.unbatched_output_shapes = get_tree_shapes("unbatched_output_shapes", None, batch_axes=batch_axes, do_not_set=True)
    else:
      self.unbatched_input_shapes = get_tree_shapes("unbatched_input_shapes", None, batch_axes=batch_axes, do_not_set=True)
      self.unbatched_output_shapes = get_tree_shapes("unbatched_output_shapes", inputs, batch_axes=batch_axes)

    # For convenience, also get the batch axes
    if sample == False:
      self.batch_shape = inputs["x"].shape[:-len(self.unbatched_input_shapes["x"])]
    else:
      self.batch_shape = inputs["x"].shape[:-len(self.unbatched_output_shapes["x"])]

    # Run the actual function
    outputs = self.call(inputs, rng, sample=sample, **kwargs)

    if sample == False:
      # Keep track of the initial output shapes
      get_tree_shapes("unbatched_output_shapes", outputs, batch_axes=batch_axes)
    else:
      get_tree_shapes("unbatched_input_shapes", outputs, batch_axes=batch_axes)

    return outputs

  def auto_batch(self, fun, in_axes=None, expected_depth=None):

    vmap_kwargs = {}
    if in_axes is not None:
      vmap_kwargs["in_axes"] = in_axes
    # if out_axes is not None:
    #   vmap_kwargs["out_axes"] = out_axes

    # We might have functions that expect batched code
    batch_depth = len(self.batch_shape)
    if expected_depth is None:
      n_newaxis = 0
      n_vmaps = batch_depth
    elif expected_depth > batch_depth:
      n_newaxis = expected_depth - batch_depth
      n_vmaps = 0
    elif expected_depth == batch_depth:
      n_newaxis = 0
      n_vmaps = 0
    elif expected_depth < batch_depth:
      n_newaxis = 0
      n_vmaps = batch_depth - expected_depth

    def vmapped_fun(*args, **kwargs):
      nonlocal self, batch_depth, expected_depth
      in_shapes = jax.tree_map(lambda x: x.shape, args)

      # Add in dummy axes to the args
      args = list(args)
      for _ in range(n_newaxis):
        if in_axes is not None:
          for i, ax in enumerate(in_axes):
            if ax is not None:
              args[i] = jax.tree_map(lambda x: jnp.expand_dims(x, ax), args[i])
        else:
          args = jax.tree_map(lambda x: x[None], args)
      args = tuple(args)

      # Apply vmap
      vf = partial(fun, **kwargs)
      for i in range(n_vmaps):
        vf = vmap(vf, **vmap_kwargs)

      # Run the function
      out = vf(*args)

      # Remove the dummy axes
      for _ in range(n_newaxis):
        out = jax.tree_map(lambda x: x[0], out)

      return out

    return vmapped_fun

  @abstractmethod
  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
    ) -> Mapping[str, jnp.ndarray]:
    """ The expectation is that inputs will be a dicionary with
        "x" holding data and "y" holding possible labels.  Other inputs
        can be passed in too """
    pass

################################################################################################################
