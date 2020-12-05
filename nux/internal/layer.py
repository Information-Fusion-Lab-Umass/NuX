from functools import partial, wraps
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
import haiku as hk
from abc import ABC, abstractmethod
from typing import Optional, Mapping, Type, Callable, Iterable, Any, Sequence, Union, Tuple, MutableMapping, NamedTuple, Set, TypeVar
import nux.util as util
from nux.internal.base import get_constant, new_custom_context
from nux.internal.functional import make_functional_modules

from haiku._src.typing import PRNGKey, Params, State
from haiku._src.transform import TransformedWithState, \
                                 to_prng_sequence, \
                                 check_mapping, \
                                 INIT_RNG_ERROR, \
                                 APPLY_RNG_STATE_ERROR, \
                                 APPLY_RNG_ERROR

__all__ = ["Layer", "transform_flow"]

################################################################################################################

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

  return get_constant(name, pytree, init=apply_get_shapes, do_not_set=do_not_set)

################################################################################################################

class Layer(hk.Module, ABC):

  batch_axes = ()

  def __init__(self, name=None, invertible_ad=False, use_flow_norm_init=False):
    """ This base class will keep track of the input and output shapes of each function call
        so that we can know the batch size of inputs and automatically use vmap to make unbatched
        code work with batched code.
    """
    super().__init__(name=name)
    self.invertible_ad = invertible_ad
    self.use_flow_norm_init = use_flow_norm_init

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
    if self.invertible_ad == False:
      outputs = self.call(inputs, rng, sample=sample, **kwargs)
    else:
      assert 0, "Not implemented"

    if sample == False:
      # Keep track of the initial output shapes
      get_tree_shapes("unbatched_output_shapes", outputs, batch_axes=batch_axes)
    else:
      get_tree_shapes("unbatched_input_shapes", outputs, batch_axes=batch_axes)

    if self.use_flow_norm_init:
      self.flow_norm_init(inputs, rng, sample, **kwargs)

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

  def flow_norm_init(self,
                     inputs: Mapping[str, jnp.ndarray],
                     rng: jnp.ndarray=None,
                     sample: Optional[bool]=False,
                     **kwargs):
    """ Initialize this layer so that its outputs are normally distributed
    """

    # Check if we've set flow norm (will be False the first time)
    flow_norm_set = get_constant("flow_norm_set", False, do_not_set=True)

    # Set that we're checking
    get_constant("flow_norm_set", True, do_not_set=False)

    if not flow_norm_set:
      # Train this layer over the input batch to generate a unit normal output

      def loss_fun(inputs, rng, sample=False, **kwargs):
        outputs = self(inputs, rng, sample=sample, **kwargs)
        z = outputs["x"]

        @self.auto_batch
        def unit_gaussian(z):
          return -0.5*jnp.sum(z.ravel()**2) # + const
        log_pz = unit_gaussian(z)

        log_px = log_pz + outputs["log_det"]
        return -log_px.mean()

      with make_functional_modules([loss_fun]) as ([apply_fun], \
                                                   params, \
                                                   state, \
                                                   finalize_params_and_state):
        import optax
        opt_init, opt_update = optax.adam(learning_rate=1e-4)
        opt_state = opt_init(params)
        opt_update = jit(opt_update)

        grad_fun = jax.value_and_grad(apply_fun, has_aux=True)
        grad_fun = partial(grad_fun, sample=sample, **kwargs)
        grad_fun = jit(grad_fun)

        import tqdm
        pbar = tqdm.tqdm(list(enumerate(random.split(rng, 200))))
        for i, rng in pbar:
          (loss, state), grad = grad_fun(params, state, inputs, rng)
          updates, opt_state = opt_update(grad, opt_state, params)
          if jnp.any(jnp.isnan(jax.flatten_util.ravel_pytree(updates)[0])):
            break
          params = jit(optax.apply_updates)(params, updates)

          pbar.set_description(f"loss: {loss}")

        finalize_params_and_state(params, state)

################################################################################################################

def transform_flow(create_fun) -> TransformedWithState:

  # We will keep the expected shapes for the flow here so that
  # JAX will compile these constants
  constants = None

  def init_fn(rng: Optional[Union[PRNGKey]],
              inputs: Mapping[str, jnp.ndarray],
              batch_axes=(),
              **kwargs
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
               **kwargs
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
