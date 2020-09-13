import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap, jit
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence
from nux.flows.base import *
import nux.util as util
from haiku._src.typing import PRNGKey

################################################################################################################

@partial(jit, static_argnums=(0, 1))
def log_det_estimate(residual_vjp, probe_shape, rng):
  """ Biased log det estimate for the moment """
  @jit
  def scan_fun(carry, inputs):
    k = inputs
    w = carry
    w, = residual_vjp(w)
    coeff = (-1)**(k+1)/k
    return w, coeff*w

  # Generate the probe vector for the trace estimate
  v = random.normal(rng, probe_shape)

  # Compute the terms in the power series
  n_terms = 10
  series_indices = jnp.arange(1, 1 + n_terms)
  _, terms = jax.lax.scan(scan_fun, v, series_indices)
  return terms.sum()

################################################################################################################

class ResidualFlow(AutoBatchedLayer):

  def __init__(self,
               res_block_create_fun: Callable=None,
               scale: Optional[float]=0.85,
               spectral_norm_iters: Optional[int]=1,
               fixed_point_iters: Optional[int]=100,
               name: str="residual_flow",
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.res_block_create_fun = res_block_create_fun
    self.scale                = scale
    self.spectral_norm_iters  = spectral_norm_iters
    self.fixed_point_iters    = fixed_point_iters

  # Initialize the residual block
  def default_res_block(self, x):
    network = util.SimpleMLP(x.shape, [16]*3, is_additive=True)
    return network(x)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           **kwargs
    ) -> Mapping[str, jnp.ndarray]:

    res_block_create_fun = self.default_res_block if self.res_block_create_fun is None else self.res_block_create_fun
    res_block = hk.transform(res_block_create_fun)
    params = res_block.init(rng, inputs["x"])

    # Apply spectral normalization
    sp = hk.SNParamsTree(n_steps=self.spectral_norm_iters, ignore_regex="[b]*")
    params = sp(params)

    if sample == False:
      x = inputs["x"]
      rng1, rng2 = random.split(rng, 2)
      apply_fun = partial(res_block.apply, params, rng1)
      gx, residual_vjp = jax.vjp(apply_fun, x)
      log_det = log_det_estimate(residual_vjp, x.shape, rng2)
      z = x + gx
      outputs = {"x": z, "log_det": log_det}
    else:
      z = inputs["x"]

      # Need to apply fixed point iterations
      def body_fun(i, x):
        gx = res_block.apply(params, rng, x)
        x = z - gx
        return x

      x = jax.lax.fori_loop(0, self.fixed_point_iters, body_fun, z)
      log_det = 0.0 # Don't worry about this for the moment
      outputs = {"x": x, "log_det": log_det}

    return outputs

################################################################################################################
