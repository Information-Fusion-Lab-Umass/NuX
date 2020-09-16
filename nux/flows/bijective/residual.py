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
import nux.networks as net

################################################################################################################

def res_flow_estimate(res_block, x, rng):
  """ Unbiased log det estimate.  Going to need some Haiku engineering to make the
      neumann gradient series work """

  gx, residual_vjp = jax.vjp(res_block, x)

  @jit
  def scan_fun(carry, inputs):
    (w, v) = carry
    w, = residual_vjp(w)
    return (w, v), (w, jnp.sum(w*v))

  trace_key, roulette_key = random.split(rng, 2)

  # Generate the probe vector for the trace estimate
  v = random.normal(trace_key, x.shape)

  # Compute the terms in the power series
  n_terms = 6
  k = jnp.arange(1, 1 + n_terms)
  _, (grad_terms, terms) = jax.lax.scan(scan_fun, (v, v), k)

  # Compute the standard scaling terms
  coeff = (-1)**(k + 1)/k

  # Compute the roulette scaling terms
  n_exact = 4
  roulette_k = k[:-n_exact]
  p = 0.5
  u = random.uniform(roulette_key, (1,))[0]
  N = jnp.floor(jnp.log(u)/jnp.log(1 - p)).astype(jnp.int32) + 1
  p_N_geq_k = (1 - p)**roulette_k

  # Zero out the terms that are over N
  roulette_coeff = jnp.where(roulette_k > N, 0.0, 1/p_N_geq_k)

  # We don't want to apply this to the exact terms
  roulette_coeff = jnp.hstack([jnp.ones(n_exact), roulette_coeff])

  return x + gx, (coeff*roulette_coeff*terms).sum()

################################################################################################################

def res_flow_exact(res_block, x):

  flat_x, unflatten = jax.flatten_util.ravel_pytree(x)

  def apply_res_block(flat_x):
    x = unflatten(flat_x)
    out = x + res_block(x)
    return jax.flatten_util.ravel_pytree(out)[0]

  J = jax.jacobian(apply_res_block)(flat_x)
  log_det = jnp.linalg.slogdet(J)[1]

  z = x + res_block(x)
  return z, log_det

################################################################################################################

class ResidualFlow(AutoBatchedLayer):

  def __init__(self,
               res_block_create_fun: Callable=None,
               scale: Optional[float]=0.85,
               spectral_norm_iters: Optional[int]=1,
               fixed_point_iters: Optional[int]=1000,
               layer_sizes: Sequence[int]=[1024]*4,
               n_channels: Optional[int]=256,
               exact_log_det: Optional[bool]=False,
               name: str="residual_flow",
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.res_block_create_fun = res_block_create_fun
    self.scale                = scale
    self.spectral_norm_iters  = spectral_norm_iters
    self.fixed_point_iters    = fixed_point_iters
    self.layer_sizes          = layer_sizes
    self.n_channels           = n_channels
    self.exact_log_det        = exact_log_det

  # Initialize the residual block
  def default_res_block(self, x):
    out_dim = x.shape[-1]
    if(x.ndim < 3):
      return net.MLP(out_dim=out_dim,
                     layer_sizes=self.layer_sizes,
                     parameter_norm="spectral",
                     nonlinearity="relu")
    else:

      return net.ConvBlock(out_channel=out_dim,
                           hidden_channel=self.n_channels,
                           parameter_norm="spectral",
                           nonlinearity="relu")

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           compare=False,
           **kwargs
    ) -> Mapping[str, jnp.ndarray]:
    res_block_create_fun = self.default_res_block if self.res_block_create_fun is None else self.res_block_create_fun
    res_block = res_block_create_fun(inputs["x"])

    if sample == False:
      x = inputs["x"]
      if(self.exact_log_det):
        z, log_det = res_flow_exact(res_block, x)

        if(compare):
          rngs = random.split(rng, 1000)
          _, log_det_ests = vmap(partial(res_flow_estimate, res_block, x))(rngs)
          print()
          print()
          print()
          print()
          print()
          print("log_det", log_det)
          print("log_det_ests.mean()", log_det_ests.mean())
          print("log_det_ests.std()", log_det_ests.std())

      else:
        z, log_det = res_flow_estimate(res_block, x, rng)
      outputs = {"x": z, "log_det": log_det}
    else:
      z = inputs["x"]

      # Need to apply fixed point iterations
      def body_fun(i, x):
        gx = res_block(x)
        x = z - gx
        return x

      x = jax.lax.fori_loop(0, self.fixed_point_iters, body_fun, z)
      log_det = 0.0 # Don't worry about this for the moment
      outputs = {"x": x, "log_det": log_det}

    return outputs

################################################################################################################