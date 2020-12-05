import jax
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial

__all__ = ["mmd2_estimate"]

@jit
def kernel(x, y, sigma=0.2):
    dx = x - y
    return jnp.exp(-0.5/sigma**2*jnp.sum(dx**2))

@jit
def mmd2_estimate(x, y, sigma=0.2):
    N, M = x.shape[0], y.shape[0]

    k = vmap(vmap(partial(kernel, sigma=sigma), in_axes=(0, None)), in_axes=(None, 0))
    kxy = k(x, y)
    kxx = k(x, x)
    kyy = k(y, y)

    diag_idx = jax.ops.index[jnp.diag_indices(x.shape[0])]
    kxx_no_diag = jax.ops.index_update(kxx, diag_idx, 0.0)

    term1 = (kxx.sum() - jnp.diag(kxx).sum())/(N*(N-1))
    term2 = (kyy.sum() - jnp.diag(kyy).sum())/(M*(M-1))
    mmd2 = term1 + term2 - 2*kxy.mean()
    return mmd2
