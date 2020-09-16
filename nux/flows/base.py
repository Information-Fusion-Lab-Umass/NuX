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
           "AutoBatchedLayer",
           "AutoBatchedLayer"]

################################################################################################################

from haiku._src.transform import TransformedWithState, to_prng_sequence, check_mapping, INIT_RNG_ERROR, APPLY_RNG_STATE_ERROR, APPLY_RNG_ERROR
from haiku._src.typing import PRNGKey, Params, State
from haiku._src.base import new_context

def transform_flow(create_fun) -> TransformedWithState:

  def init_fn(
      rng: Optional[Union[PRNGKey]],
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

      key = hk.next_rng_key()

      # Initialize the model
      outputs = model(inputs, key, **kwargs)

      # We also need to run it in reverse to initialize the sample shapes!
      inputs_for_reconstr = inputs.copy()
      inputs_for_reconstr.update(outputs) # We might have condition variables in inputs!
      model(inputs_for_reconstr, key, sample=True, **kwargs)

      # Unset the batch axes
      Layer.batch_axes = ()

    return ctx.collect_params(), ctx.collect_initial_state()

  def apply_fn(
      params: Optional[Params],
      state: Optional[State],
      rng: Optional[Union[PRNGKey]],
      inputs,
      **kwargs,
  ) -> Tuple[Any, State]:
    """Applies your function injecting parameters and state."""
    params = check_mapping("params", params)
    state = check_mapping("state", state)
    rng = to_prng_sequence(rng, err_msg=(APPLY_RNG_STATE_ERROR if state else APPLY_RNG_ERROR))
    with new_context(params=params, state=state, rng=rng) as ctx:
      model = create_fun()
      key = hk.next_rng_key()
      out = model(inputs, key, **kwargs)
    return out, ctx.collect_state()

  return TransformedWithState(init_fn, apply_fn)

################################################################################################################

from haiku._src.base import current_frame, current_bundle_name, assert_context, ParamContext, create_parameter, run_custom_getters, current_module
from haiku._src.typing import Initializer

def get_parameter_no_shape_check(
    name: str,
    shape: Sequence[int],
    dtype: Any = jnp.float32,
    init: Initializer = None,
) -> jnp.ndarray:
  assert_context("get_parameter")
  assert init is not None, "Initializer must be specified."

  bundle_name = current_bundle_name()
  frame = current_frame()

  if frame.params_frozen and bundle_name not in frame.params:
    raise ValueError(
        "Unable to retrieve parameter {!r} for module {!r}. "
        "All parameters must be created as part of `init`.".format(
            name, bundle_name))

  params = frame.params[bundle_name]
  param = params.get(name)
  fq_name = bundle_name + "/" + name
  if param is None:
    if frame.params_frozen:
      raise ValueError(
          "Unable to retrieve parameter {!r} for module {!r}. "
          "All parameters must be created as part of `init`.".format(
              name, bundle_name))

    param = create_parameter(fq_name, shape, dtype, init)
    params[name] = param  # pytype: disable=unsupported-operands

  # Custom getters allow a hook for users to customize the value returned by
  # get_parameter. For example casting values to some dtype.
  param = run_custom_getters(fq_name, param)

  # Don't do this!!!
  # assert param.shape == tuple(shape), (
  #     "{!r} with shape {!r} does not match shape={!r} dtype={!r}".format(
  #         fq_name, param.shape, shape, dtype))

  return param

################################################################################################################

from haiku._src.base import current_frame, current_bundle_name, StatePair

def get_tree_shapes(name: str,
                    pytree: Any,
                    batch_axes: Optional[Sequence[int]] = ()
) -> Any:
  state = current_frame().state[current_bundle_name()]
  value = state.get(name, None)
  if value is None:

    @jit
    def get_unbatched_shape(x):
      x_shape = [s for i, s in enumerate(x.shape) if i not in batch_axes]
      x_shape = tuple(x_shape)
      return x_shape

    tree_shapes = jit(partial(jax.tree_util.tree_map, get_unbatched_shape))(pytree)

    value = state[name] = StatePair(tree_shapes, tree_shapes)
  return value.current

################################################################################################################

class Layer(hk.Module, ABC):

  batch_axes = ()

  def __init__(self, name=None):
    super().__init__(name=name)

  def set_expected_shapes(self,
                          inputs: Mapping[str, jnp.ndarray],
                          sample: Optional[bool]=False,
                          no_batching: Optional[bool]=False):

    # We can force the code to not batch.  This is useful if we define an unbatched layer inside an auto-batched layer
    batch_axes = Layer.batch_axes if no_batching == False else ()

    # Keep track of the initial input shapes
    if sample == False:
      self.expected_shapes = get_tree_shapes("input_shapes", inputs, batch_axes=batch_axes)
    else:
      self.expected_shapes = get_tree_shapes("sample_input_shapes", inputs, batch_axes=batch_axes)

  def __call__(self,
               inputs: Mapping[str, jnp.ndarray],
               rng: jnp.ndarray=None,
               sample: Optional[bool]=False,
               no_batching: Optional[bool]=False,
               **kwargs
    ) -> Mapping[str, jnp.ndarray]:
    self.set_expected_shapes(inputs, sample=sample, no_batching=no_batching)
    return self.call(inputs, rng, sample, no_batching=no_batching, **kwargs)

  @abstractmethod
  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           no_batching: Optional[bool]=False,
           **kwargs
    ) -> Mapping[str, jnp.ndarray]:
    """ The expectation is that inputs will be a dicionary with
        "x" holding data and "y" holding possible labels.  Other inputs
        can be passed in too """
    pass

################################################################################################################

class AutoBatchedLayer(Layer):

  def __call__(self,
               inputs: Mapping[str, jnp.ndarray],
               rng: jnp.ndarray=None,
               sample: Optional[bool]=False,
               no_batching: Optional[bool]=False,
               **kwargs
    ) -> Mapping[str, jnp.ndarray]:

    # Set the expected shapes
    self.set_expected_shapes(inputs, sample=sample, no_batching=no_batching)

    # Determine the passed input shapes
    input_shapes = util.tree_shapes(inputs)

    input_ndim = len(input_shapes["x"])
    expected_ndim = len(self.expected_shapes["x"])

    if input_ndim > expected_ndim:
      # Need to vmap over the random key too
      batch_size = input_shapes["x"][0]
      rngs = random.split(rng, batch_size) if rng is not None else tuple([None]*batch_size)
      return vmap(partial(self, sample=sample, no_batching=no_batching, **kwargs))(inputs, rngs)

    # Evaluate the function or do flow norm initialization
    flow_norm = kwargs.get("flow_norm", False)
    if flow_norm is False:
      outputs = self.call(inputs, rng, sample=sample, no_batching=no_batching, **kwargs)
    else:
      input_no_grad = jax.lax.stop_gradient(inputs)
      outputs = self.call(input_no_grad, rng, sample=sample, no_batching=no_batching, **kwargs)
      outputs["flow_norm"] = -0.5*jnp.sum(outputs["x"]**2) + outputs.get("log_det", jnp.array(0.0))

    # Record the output shapes.  outputs is unbatched!
    if sample == False:
      get_tree_shapes("output_shapes", outputs)
    else:
      get_tree_shapes("sample_output_shapes", outputs)

    return outputs
