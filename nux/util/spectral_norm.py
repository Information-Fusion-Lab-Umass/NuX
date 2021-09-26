import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
from collections import namedtuple

def spectral_norm_iter(mvp, v, return_u=False):
  uhat, mvpT = jax.vjp(lambda x: mvp(x), v)
  u = uhat*jax.lax.rsqrt((uhat**2).sum())
  if return_u:
    return u
  vhat = mvpT(u)[0]
  v = vhat*jax.lax.rsqrt((vhat**2).sum())
  return v

def max_singular_value(mvp, v, n_iters=10):
  sn_iter = partial(spectral_norm_iter, mvp)

  if n_iters == -1:

    def flatten(x):
      return jax.flatten_util.ravel_pytree(x)[0]

    def cond(val):
      v_prev, v, i = val
      max_iters_reached = jnp.where(i >= 5000, True, False)
      tolerance_achieved = jnp.allclose(flatten(v_prev) - flatten(v), 0.0, atol=1e-6)
      first_iter = jnp.where(i > 0.0, False, True)
      return ~(max_iters_reached | tolerance_achieved) | first_iter

    def loop(val):
      _, v, i = val
      v_new = sn_iter(v)
      return v, v_new, i + 1

    # Initialize v
    _, v, _ = jax.lax.while_loop(cond, loop, (v, v, 0.0))
  else:

    def body(v, _):
      return sn_iter(v), ()

    v, _ = jax.lax.scan(body, v, jnp.arange(n_iters), unroll=1)

  u = sn_iter(v, return_u=True)

  u = jax.lax.stop_gradient(u)
  v = jax.lax.stop_gradient(v)
  sigma = jnp.vdot(u, mvp(v))
  return sigma, v
