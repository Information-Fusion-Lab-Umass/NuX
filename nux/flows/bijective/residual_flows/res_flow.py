import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap, jit
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence
from nux.internal.layer import InvertibleLayer, Layer
from nux.internal.base import CustomFrame
from haiku._src.typing import PRNGKey
import nux.networks as net
from nux.internal.functional import make_functional_modules

import nux.internal.functional as functional
import haiku._src.base as hk_base

from nux.flows.bijective.residual_flows.exact import res_flow_exact
from nux.flows.bijective.residual_flows.trace_estimator import res_flow_sliced_estimate
from nux.flows.bijective.residual_flows.inverse import fixed_point

class ResidualFlow(InvertibleLayer):

  def __init__(self,
               create_network: Callable=None,
               fixed_point_iters: Optional[int]=1000,
               exact_log_det: Optional[bool]=False,
               use_trace_estimator: bool=True,
               network_kwargs: Optional=None,
               name: str="residual_flow"
  ):
    """ Residual flows https://arxiv.org/pdf/1906.02735.pdf

    Args:
      create_network   : Function to create the conditioner network.  Should accept a tuple
                         specifying the output shape.  See coupling_base.py
      fixed_point_iters: Max number of iterations for inverse
      exact_log_det    : Whether or not to compute the exact jacobian determinant with autodiff
      network_kwargs   : Dictionary with settings for the default network (see get_default_network in util.py)
      name             : Optional name for this module.
    """
    super().__init__(name=name)
    self.create_network      = create_network
    self.fixed_point_iters   = fixed_point_iters
    self.exact_log_det       = exact_log_det
    self.network_kwargs      = network_kwargs
    self.use_trace_estimator = use_trace_estimator

  def get_network(self, out_shape):
    if self.create_network is not None:
      return self.create_network(out_shape)
    return util.get_default_network(out_shape, network_kwargs=self.network_kwargs, lipschitz=True)

  @property
  def true_res_fun(self):
    def fun(x, rng, **kwargs):
      return self.res_block({"x": x}, rng, **kwargs)["x"]
    return fun

  @property
  def auto_batched_res_block(self):
    return self.true_res_fun
    # return self.auto_batch(self.res_block, expected_depth=1, in_axes=(0, None))

  def exact_forward(self, x, rng):
    res_fun = partial(res_flow_exact, self.true_res_fun)
    # res_fun = partial(res_flow_exact, self.res_block)
    z, log_det = self.auto_batch(res_fun, in_axes=(0, None))(x, rng)
    return z, log_det

  def init_if_needed(self, x, rng):
    # Before extracting the frame data, we need to make sure that the
    # network is initialized!
    # running_init_fn = not hk_base.params_frozen()
    # if running_init_fn:
    if Layer._is_initializing:
      self.auto_batched_res_block(x, rng)

  def forward(self, x, rng, update_params):
    self.init_if_needed(x, rng)

    batch_info = self.unbatched_input_shapes["x"], self.batch_shape

    with make_functional_modules([self.auto_batched_res_block]) as ([apply_fun], \
                                                                    params, \
                                                                    state, \
                                                                    finalize):
      if self.use_trace_estimator:
        z, log_det, state = res_flow_sliced_estimate(apply_fun, params, state, x, rng, batch_info)
      else:
        z, log_det, state = res_flow_estimate(apply_fun, params, state, x, rng, batch_info)

      # Ensure that we don't backprop through state (this shouldn't affect anything)
      state = jax.lax.stop_gradient(state)

      # Update the Haiku global states
      finalize(params, state)

    return z, log_det

  def invert(self, z, rng):
    self.init_if_needed(z, rng)

    # State will be held constant during the fixed point iterations
    fun = partial(self.auto_batched_res_block, update_params=False)

    with make_functional_modules([fun]) as ([apply_fun], \
                                            params, \
                                            state, \
                                            finalize):
      # Make sure we don't use a different random key at every step of the fixed point iterations.
      deterministic_apply_fun = lambda params, state, x: apply_fun(params, state, x, rng)

      # Run the fixed point iterations to invert at z.  We can do reverse-mode through this!
      x = fixed_point(deterministic_apply_fun, params, state, z, rng)

      # Update the Haiku global states
      finalize(params, state)

    return x

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           res_block_only: bool=False,
           use_exact_log_det: bool=False,
           no_log_det: bool=False,
           force_update_params: bool=False,
           **kwargs
    ) -> Mapping[str, jnp.ndarray]:
    x_shape = self.get_unbatched_shapes(sample)["x"]
    self.res_block = self.get_network(x_shape)

    # Every once in a while we might want to forcibly run a lot of power iterations
    if force_update_params:
      self.res_block.max_power_iters = 100

    if res_block_only:
      x = inputs["x"]
      gx = self.auto_batched_res_block(x, rng)
      return {"x": gx, "log_det": jnp.zeros(self.batch_shape)}

    if sample == False:
      x = inputs["x"]

      if self.exact_log_det or use_exact_log_det:
        z, log_det = self.exact_forward(x, rng)
      else:

        update_params = True if kwargs.get("is_training", True) else False
        z, log_det = self.forward(x, rng, update_params)

      outputs = {"x": z, "log_det": log_det}
    else:
      z = inputs["x"]
      x = self.invert(z, rng)

      if no_log_det == False:
        if self.exact_log_det or use_exact_log_det:
          _, log_det = self.exact_forward(x, rng)
        else:
          _, log_det = self.forward(x, rng, update_params=False)
      else:
        log_det = jnp.zeros(self.batch_shape)


      outputs = {"x": x, "log_det": log_det}

    return outputs

################################################################################################################

if __name__ == "__main__":
  from debug import *
  from nux.tests.bijective_test import flow_test

  def create_resnet_network(out_shape):
    return net.ReverseBottleneckConv(out_channel=out_shape[-1],
                                     hidden_channel=16,
                                     nonlinearity="lipswish",
                                     normalization=None,
                                     parameter_norm="differentiable_spectral_norm",
                                     use_bias=True,
                                     dropout_rate=None,
                                     gate=False,
                                     activate_last=False,
                                     max_singular_value=0.999,
                                     max_power_iters=1)

  def create_fun():
    return ResidualFlow(create_network=create_resnet_network)

  rng = random.PRNGKey(1)
  # x = random.normal(rng, (10, 8))
  x = random.normal(rng, (10, 4, 4, 3))

  inputs = {"x": x}
  flow_test(create_fun, jax.tree_map(lambda x:x[0], inputs), rng)

  flow = nux.Flow(create_fun, rng, inputs, batch_axes=(0,))
  print(f"flow.n_params: {flow.n_params}")

  def loss(params, state, key, inputs):
    outputs, _ = flow._apply_fun(params, state, key, inputs)
    log_px = outputs.get("log_pz", 0.0) + outputs.get("log_det", 0.0)
    return -log_px.mean()

  outputs = flow.scan_apply(rng, inputs)
  samples = flow.sample(rng, n_samples=4)
  trainer = nux.MaximumLikelihoodTrainer(flow)

  trainer.grad_step(rng, inputs)
  trainer.grad_step_for_loop(rng, inputs)
  trainer.grad_step_scan_loop(rng, inputs)

  gradfun = jax.grad(loss)
  gradfun = jax.jit(gradfun)
  gradfun(flow.params, flow.state, rng, inputs)

  assert 0