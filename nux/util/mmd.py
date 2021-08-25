import jax
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial

__all__ = ["mmd2_estimate"]

def kernel(x, y, sigma):
  dx = x - y
  return jnp.exp(-0.5/sigma**2*jnp.sum(dx**2))

def mmd2_estimate(x, y, sigma=0.2):
  N, M = x.shape[0], y.shape[0]

  k = vmap(vmap(kernel, in_axes=(0, None, None)), in_axes=(None, 0, None))
  kxy = k(x, y, sigma)
  kxx = k(x, x, sigma)
  kyy = k(y, y, sigma)

  kxx_no_diag = kxx.at[jnp.diag_indices(x.shape[0])].set(0.0)
  kyy_no_diag = kyy.at[jnp.diag_indices(y.shape[0])].set(0.0)

  term1 = kxx_no_diag.sum()/(N*(N-1))
  term2 = kyy_no_diag.sum()/(M*(M-1))
  mmd2 = term1 + term2 - 2*kxy.mean()
  return mmd2
