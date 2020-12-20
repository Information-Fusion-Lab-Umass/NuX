import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from functools import partial
from typing import Optional, Mapping, Callable, Sequence
from haiku._src.typing import PRNGKey

def res_flow_exact(res_block, x, rng):
  # This must be called using auto-batch so that jax.jacobian works!

  flat_x, unflatten = jax.flatten_util.ravel_pytree(x)

  def apply_res_block(flat_x):
    x = unflatten(flat_x)
    out = x + res_block(x[None], rng, update_params=False)[0]
    return jax.flatten_util.ravel_pytree(out)[0]

  J = jax.jacobian(apply_res_block)(flat_x)

  log_det = jnp.linalg.slogdet(J)[1]

  z = x + res_block(x[None], rng, update_params=True)[0]
  return z, log_det
