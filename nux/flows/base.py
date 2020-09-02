from functools import partial
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
import haiku as hk
from abc import ABC, abstractmethod
from typing import Optional, Mapping, Type, Callable, Iterable, Any, Sequence, Union, Tuple
import nux.util as util

__all__ = ["transform_flow",
           "Layer",
           "AutoBatchedLayer"]

################################################################################################################

from haiku._src.transform import TransformedWithState, to_prng_sequence, check_mapping, INIT_RNG_ERROR, APPLY_RNG_STATE_ERROR, APPLY_RNG_ERROR
from haiku._src.typing import PRNGKey, PRNGSeed, Params, State
from haiku._src.base import new_context

def transform_flow(create_fun) -> TransformedWithState:

  def init_fn(
      rng: Optional[Union[PRNGKey, PRNGSeed]],
      inputs: Mapping[str, jnp.ndarray],
      batch_axes=(),
      **kwargs,
  ) -> Tuple[Params, State]:
    """Initializes your function collecting parameters and state."""
    rng = to_prng_sequence(rng, err_msg=INIT_RNG_ERROR)
    with new_context(rng=rng) as ctx:
      # Create the model
      model = create_fun()

      # Load the batch axes for the inputs
      Layer.batch_axes = batch_axes

      # Initialize the model
      outputs = model(inputs, **kwargs)

      # We also need to run it in reverse to initialize the sample shapes!
      model(outputs, sample=True, **kwargs)

      # Unset the batch axes
      Layer.batch_axes = ()

    return ctx.collect_params(), ctx.collect_initial_state()

  def apply_fn(
      params: Optional[Params],
      state: Optional[State],
      rng: Optional[Union[PRNGKey, PRNGSeed]],
      *args,
      **kwargs,
  ) -> Tuple[Any, State]:
    """Applies your function injecting parameters and state."""
    params = check_mapping("params", params)
    state = check_mapping("state", state)
    rng = to_prng_sequence(
        rng, err_msg=(APPLY_RNG_STATE_ERROR if state else APPLY_RNG_ERROR))
    with new_context(params=params, state=state, rng=rng) as ctx:
      model = create_fun()
      out = model(*args, **kwargs)
    return out, ctx.collect_state()

  return TransformedWithState(init_fn, apply_fn)

################################################################################################################

from haiku._src.base import current_frame, current_bundle_name, StatePair
from haiku._src.typing import ParamName, Shape

def get_tree_shapes(name: ParamName,
                    pytree: Any,
                    batch_axes: Optional[Sequence[int]] = ()
) -> Shape:
  state = current_frame().state[current_bundle_name()]
  value = state.get(name, None)
  if value is None:

    def get_unbatched_shape(x):
      x_shape = [s for i, s in enumerate(x.shape) if i not in batch_axes]
      x_shape = tuple(x_shape)
      return x_shape

    tree_shapes = jax.tree_util.tree_map(get_unbatched_shape, pytree)

    value = state[name] = StatePair(tree_shapes, tree_shapes)
  return value.current

################################################################################################################

class Layer(hk.Module, ABC):

  batch_axes = ()

  def __init__(self, name=None):
    super().__init__(name=name)

  def set_expected_shapes(self, inputs: Mapping[str, jnp.ndarray], sample: Optional[bool]=False):
    # Keep track of the initial input shapes
    if sample == False:
      self.expected_shapes = get_tree_shapes("input_shapes", inputs, batch_axes=Layer.batch_axes)
    else:
      self.expected_shapes = get_tree_shapes("sample_input_shapes", inputs, batch_axes=Layer.batch_axes)

  def __call__(self, inputs: Mapping[str, jnp.ndarray], sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    self.set_expected_shapes(inputs, sample=sample)
    return self.call(inputs, sample, **kwargs)

  @abstractmethod
  def call(self, inputs: Mapping[str, jnp.ndarray], sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    """ The expectation is that inputs will be a dicionary with
        "x" holding data and "y" holding possible labels.  Other inputs
        can be passed in too """
    pass

################################################################################################################

class AutoBatchedLayer(Layer):

  def __call__(self, inputs: Mapping[str, jnp.ndarray], sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    self.set_expected_shapes(inputs, sample=sample)

    # Determine the passed input shapes
    input_shapes = util.tree_shapes(inputs)

    recurse = False
    batch_size = None
    input_in_axes = {}

    # Figure out which inputs need to be vmapped over
    for name, expected_shape in self.expected_shapes.items():
      input_shape = input_shapes[name]
      input_ndim = len(input_shape)
      expected_ndim = len(expected_shape)

      # If the dimensinoality of the input is more then expected, we need to vmap
      if input_ndim > expected_ndim:
        input_in_axes[name] = 0
        recurse = True

        # We need to make sure that the batch sizes are the same across all
        # of the vmap arguments
        if batch_size is None:
          batch_size = input_shape[0]
        else:
          assert input_shape[0] == batch_size, "Batch size mismatch."
      else:
        # We don't need to vmap over this input
        input_in_axes[name] = None

    # Evaluate the vmapped function
    if recurse:
      return vmap(partial(self, sample=sample, **kwargs), in_axes=(input_in_axes,))(inputs)

    # Evaluate the function
    outputs = self.call(inputs, sample=sample, **kwargs)

    # Record the output shapes.  outputs is unbatched!
    if sample == False:
      get_tree_shapes("output_shapes", outputs)
    else:
      get_tree_shapes("sample_output_shapes", outputs)

    return outputs

################################################################################################################
