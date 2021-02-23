from functools import partial, wraps
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
import haiku as hk
from abc import ABC, abstractmethod
from typing import Optional, Mapping, Type, Callable, Iterable, Any, Sequence, Union, Tuple, MutableMapping, NamedTuple, Set, TypeVar
import collections
import contextlib
from nux.internal.base import FrameData, CustomFrame

from haiku._src.base import PRNGSequence, \
                            frame_stack, \
                            StatePair, \
                            PRNGSequenceState, \
                            current_frame, \
                            params_frozen
T = TypeVar("T")

__all__ = ["make_functional_modules"]

################################################################################################################

# class custom_vjp(jax.custom_vjp):
#   # Doesn't work for some reason

#   def defvjp(self, fwd, bwd):
#     def fwd_wrapper(*args):
#       out, ctx = fwd(*args)
#       frame_data = get_frame_data()
#       return out, (ctx, frame_data)

#     def bwd_wrapper(ctx_and_frame_data, g):
#       (ctx, frame_data) = ctx_and_frame_data
#       with frame_stack(CustomFrame.create_from_frame_data(frame_data)):
#         return bwd(ctx, g)

################################################################################################################

@contextlib.contextmanager
def make_functional_modules(modules):

  def wrap_module(module):

    def wrapped(params, bundled_state, *args, **kwargs):
      frame_data = to_frame_data(params, bundled_state)
      with temporary_frame_data(frame_data):
        out = module(*args, **kwargs)
        return out, get_bundled_state()

    return wrapped

  did_finalize = False
  exception = None
  original_treedef = None

  def finalize(params, bundled_state, state_only=True):
    nonlocal did_finalize, original_treedef
    # assert jax.tree_structure(bundled_state) == original_treedef
    update_modified_frame_data_from_args(params, bundled_state, state_only=state_only)
    did_finalize = True

  try:
    params, bundled_state = get_params_and_bundled_state()
    original_treedef = jax.tree_structure(bundled_state)

    wrapped_modules = [wrap_module(module) for module in modules]

    yield wrapped_modules, params, bundled_state, finalize
  except Exception as e:
    exception = e
  finally:
    if exception is not None:
      raise exception
    assert did_finalize, "Did you forget to call 'finalize(frame_data)' before exiting the with statement?"

################################################################################################################

def copy_structure(bundle: T) -> T:
  return jax.tree_map(lambda x: x, bundle)

def get_frame_data() -> FrameData:
  frame = current_frame()
  rng = frame.rng_stack.peek()
  if rng is not None:
    rng = rng.internal_state
  return FrameData(params=copy_structure(frame.params),
                   state=copy_structure(frame.state),
                   constants=copy_structure(frame.constants),
                   rng=copy_structure(rng))

def get_params_and_bundled_state():
  frame_data = get_frame_data()
  return frame_data.params, (frame_data.state, frame_data.constants, frame_data.rng)

def to_frame_data(params, bundled_state):
  (state, constants, rng) = bundled_state
  return FrameData(params=params,
                   state=state,
                   constants=constants,
                   rng=rng)

def get_bundled_state():
  frame = current_frame()
  rng = frame.rng_stack.peek()
  if rng is not None:
    rng = rng.internal_state
  return (frame.state, frame.constants, rng)

################################################################################################################

def update_recursive_skip_none(dst: MutableMapping[Any, Any], src: Mapping[Any, Any]):
  # Set dst with the items of src that aren't None
  for k, v in src.items():
    if isinstance(v, collections.Mapping):
      dst.setdefault(k, {})
      update_recursive_skip_none(dst[k], v)
    else:
      if v is not None:
        # NOTE: We only expect `None` values thanks to `difference`.
        dst[k] = v

def update_modified_frame_data_from_args(params, bundled_state, state_only=False):
  (state, constants, rng) = bundled_state
  frame = current_frame()
  if state_only == False and not params_frozen():
    update_recursive_skip_none(frame.params, params)
    assert 0
  update_recursive_skip_none(frame.state, state)
  if state_only == False and not params_frozen():
    update_recursive_skip_none(frame.constants, constants)
    assert 0
  rng = rng
  if rng is not None:
    frame.rng_stack.peek().replace_internal_state(rng)

def temporary_frame_data(frame_data: FrameData):
  """Pushes a temporary copy of the frame_data."""
  frame_data = copy_structure(frame_data)
  rng = frame_data.rng if frame_data.rng is None else PRNGSequence(frame_data.rng)
  params = frame_data.params
  state = frame_data.state
  constants = frame_data.constants
  assert params is not None, "Must initialize module before this call"
  assert state is not None, "Must initialize module before this call"
  assert constants is not None, "Must initialize module before this call"

  frame = current_frame()
  frame = frame.evolve(params=params, state=state, constants=constants, rng=rng)
  return frame_stack(frame)

################################################################################################################

class Box:
  """A pytree leaf that acts as a box."""

  def __init__(self, value):
    self.value = value

TwoLevelMapping = Mapping[Any, Mapping[Any, Any]]
TwoLevelMappingToBox = Mapping[Any, Mapping[Any, Box]]

def box_and_fill_missing(original_tree: TwoLevelMapping,
                         modified_tree: TwoLevelMapping,
) -> Tuple[TwoLevelMappingToBox, TwoLevelMappingToBox]:
  """ Wraps the leaves of "original_tree" and "modified_tree" in Box objects.  Assumes that
      all of the branches contained in "original_tree" are in "modified_tree".
      If "original_tree" does not contain original_tree branch of "modified_tree",
      then we create a Box(None) leaf in the original_tree.
  """
  boxed_modified_tree = {k: {} for k in modified_tree}
  boxed_original_tree = {k: {} for k in modified_tree}

  # Loop through every key in every dictionary in modified_tree
  for k1, v1 in modified_tree.items():
    for k2 in v1:
      # Create original_tree box for the leaves of modified_tree
      boxed_modified_tree[k1][k2] = Box(modified_tree[k1][k2])

      # Forcefully create original_tree box with the leaves of original_tree.
      # The Box will have None if original_tree does not have this leaf.
      if k1 in original_tree and k2 in original_tree[k1]:
        boxed_original_tree[k1][k2] = Box(original_tree[k1][k2])
      else:
        boxed_original_tree[k1][k2] = Box(None)

  return boxed_original_tree, boxed_modified_tree

def difference(before: FrameData, after: FrameData) -> FrameData:
  """Returns an FrameData object with unchanged items set to ``None``.
  Note that to determine what values have changed we compare them by identity
  not by value. This is only reasonable to do if `difference` is used to compare
  state *inside* a JAX transform (e.g. comparing the arguments passed into JIT
  with the values that you are about to return from it).
  This function never produces false negatives (e.g. we will never incorrectly
  say that a piece of state is unchanged when it has), however it may produce
  false positives. One well known case is if a value is traced by an inner JAX
  transform but unchanged, the identity of the Python object will differ from
  the value passed into the outer function, but the value will not have changed.
  In this case `difference` will say that the value has changed. For example if
  the following change happened inside a function whose state was being diffed
  we would defensively say that ``u`` had changed value even though it had only
  changed Python identity:
  >>> u = hk.get_state("u", [], init=jnp.ones)
  >>> u, _ = jax.jit(lambda a: a, a ** 2)(u)
  >>> hk.set_state("u", u)
  Args:
    before: state before.
    after: state after.
  Returns:
    The difference between before and after, with any values that have the same
    identity before and after set to `None`.
  """

  def if_changed(are_different, box_a, box_b):
    if box_a.value is None or are_different(box_a.value, box_b.value):
      return box_b.value
    else:
      return None

  # Each leaf of params_after is not None if the corresponding
  # leaf of after.params does not match that of before.params
  are_different = lambda a, b: a is not b
  params_before, params_after = box_and_fill_missing(before.params, after.params)
  params_after = jax.tree_multimap(partial(if_changed, are_different),
                                   params_before, params_after)

  # Each leaf of state_after is not None if the corresponding
  # leaf of after.state does not match that of before.state
  def is_new_state(a: StatePair, b: StatePair):
    return a.initial is not b.initial or a.current is not b.current

  state_before, state_afterr = box_and_fill_missing(before.state, after.state)
  state_after = jax.tree_multimap(partial(if_changed, is_new_state),
                                  state_before, state_afterr)

  # Each leaf of constants_after is not None if the corresponding
  # leaf of after.constants does not match that of before.constants
  are_different = lambda a, b: a is not b
  constants_before, constants_after = box_and_fill_missing(before.constants, after.constants)
  constants_after = jax.tree_multimap(partial(if_changed, are_different),
                                      constants_before, constants_after)

  # See if rng changed.  Not sure why this would change.
  def is_new_rng(a: Optional[PRNGSequenceState],
                 b: Optional[PRNGSequenceState]):
    if a is None:
      return True
    assert len(a) == 2 and len(b) == 2
    return a[0] is not b[0] or a[1] is not b[1]

  rng = after.rng if is_new_rng(before.rng, after.rng) else None

  return FrameData(params_after, state_after, constants_after, rng)

