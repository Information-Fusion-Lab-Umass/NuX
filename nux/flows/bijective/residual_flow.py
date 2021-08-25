import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import nux.util as util

__all__ = ["ResidualFlow"]

def geometric_roulette_coefficients(key, n_terms):
  # Compute the roulette coefficients using a geometric distribution
  k = jnp.arange(n_terms)
  p = 0.5
  u = random.uniform(key, (1,))[0]
  N = jnp.floor(jnp.log(u)/jnp.log(1 - p)) + 1
  p_N_geq_k = (1 - p)**k

  # Zero out the terms that are over N
  roulette_coeff = jnp.where(k > N, 0.0, 1/p_N_geq_k)
  return roulette_coeff

def log_det_and_surrogate(rng, batch_shape, x_shape, vjp_x):

  k1, k2 = random.split(rng, 2)
  v = random.normal(k1, batch_shape + x_shape)

  def scan_body(carry, inputs):
    k = inputs
    w = carry
    w = vjp_x(w)
    term = util.batched_vdot(w, v, x_shape)

    log_det_term = term/k
    grad_term = -w

    w *= -1

    return w, (log_det_term, grad_term)

  n_exact = 4
  n_total = 8
  k = jnp.arange(1, n_total + 1)

  roulette_coeff = geometric_roulette_coefficients(k2, n_total - n_exact)
  roulette_coeff = jnp.hstack([jnp.ones(n_exact), roulette_coeff])

  _, (log_det_terms, grad_terms) = jax.lax.scan(scan_body, v, k, unroll=10)

  log_det_est = (log_det_terms*util.broadcast_to_first_axis(roulette_coeff, log_det_terms.ndim)).sum(axis=0)
  log_det_est = jax.lax.stop_gradient(log_det_est)

  b = (grad_terms*util.broadcast_to_first_axis(roulette_coeff, grad_terms.ndim)).sum(axis=0)
  b = jax.lax.stop_gradient(b)

  surrogate_objective = util.batched_vdot(vjp_x(b), v, x_shape)

  return log_det_est, surrogate_objective

def res_flow_inv(fun, z, *args):

  def flatten(x):
    return jax.flatten_util.ravel_pytree(x)[0]

  def cond(val):
    x_prev, x, i = val
    max_iters_reached = jnp.where(i >= 1000, True, False)
    tolerance_achieved = jnp.allclose(flatten(x_prev) - flatten(x), 0.0, atol=1e-6)
    return ~(max_iters_reached | tolerance_achieved)

  def loop(val):
    _, x, i = val
    x_new = z - fun(x, *args)
    return x, x_new, i + 1

  # Initialize x
  _, x, n_iters = jax.lax.while_loop(cond, loop, (jnp.zeros_like(z), z, 0.0))
  return x

################################################################################################################

class ResidualFlow():

  def __init__(self, res_block):
    self.res_block = res_block

  def get_params(self):
    return self.res_block.get_params()

  def __call__(self, x, params=None, inverse=False, rng_key=None, no_llc=False, **kwargs):
    self.params = params

    def apply_fun(x):
      x = self.res_block(x, params=self.params, rng_key=rng_key, **kwargs)
      return x

    if self.params is None:
      apply_fun(x)
      self.params = self.get_params()

    if inverse == False:
      if no_llc == True:
        z = x + apply_fun(x)
      else:
        gx, vjp = jax.vjp(apply_fun, x)
        z = x + gx
    else:
      z = res_flow_inv(apply_fun, x)
      if no_llc == False:
        _, vjp = jax.vjp(apply_fun, z)

    if no_llc:
      return z, jnp.zeros(z.shape[0])

    batch_shape, x_shape = x.shape[:1], x.shape[1:]
    vjp_x = lambda x: vjp(x)[0]
    log_det, surrogate = log_det_and_surrogate(rng_key, batch_shape, x_shape, vjp_x)
    res_log_det = jax.lax.stop_gradient(log_det) + util.only_gradient(surrogate)

    return z, res_log_det

################################################################################################################

if __name__ == "__main__":
  from debug import *
  import nux

  rng_key = random.PRNGKey(1)
  # x_shape = (16, 4, 4, 3)
  x_shape = (16, 3)
  x, aux = random.normal(rng_key, (2,)+x_shape)

  filter_shape    = (3, 3)
  hidden_channel  = 16
  dropout_prob    = 0.2
  n_layers        = 4
  # res_block = nux.LinfLipschitzConvResBlock(filter_shape,
  #                                           hidden_channel,
  #                                           n_layers,
  #                                           dropout_prob)

  res_block = nux.LinfLipschitzDenseResBlock(hidden_channel,
                                             n_layers,
                                             dropout_prob)

  flow = ResidualFlow(res_block)

  z, log_det = flow(x, rng_key=rng_key)
  params = flow.get_params()

  import pdb; pdb.set_trace()
