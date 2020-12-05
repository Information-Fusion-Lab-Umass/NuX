import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
import nux.util as util
from typing import Optional, Mapping, Callable, Sequence, Any, Union, Tuple
import haiku as hk

__all__ = ["spectral_norm_apply",
           "spectral_norm_conv_apply",
           "check_spectral_norm"]

################################################################################################################

def check_spectral_norm(pytree):
  """ Check the spectral norm of the leaves of pytree """
  def spectral_norm_apply(val):
    return jnp.linalg.norm(val, ord=2) if val.ndim == 2 else 0

  return jax.tree_util.tree_map(spectral_norm_apply, pytree)

################################################################################################################

@jit
def spectral_norm_iter(W, uv):
  """ Perform a single spectral norm iteration """
  u, v = uv

  # Perform the spectral norm iterations
  v = W.T@u
  v *= jax.lax.rsqrt(jnp.dot(v, v) + 1e-12)

  u = W@v
  u *= jax.lax.rsqrt(jnp.dot(u, u) + 1e-12)

  return (u, v)

@partial(jit, static_argnums=(4, 5))
def spectral_norm_apply(W: jnp.ndarray,
                        u: jnp.ndarray,
                        v: jnp.ndarray,
                        scale: float,
                        n_iters: int,
                        update_params: bool):
  """ Perform at most n_iters single spectral norm iterations """

  if update_params:

    # Perform the spectral norm iterations
    fp = jax.jit(util.fixed_point, static_argnums=(0,))
    (u, v) = fp(spectral_norm_iter, W, (u, v), n_iters)

    # Other implementations stop the gradient, but we can get the gradient
    # wrt W efficiently by backprop-ing through the fixed point iters.
    u = jax.lax.stop_gradient(u)
    v = jax.lax.stop_gradient(v)

  # Estimate the largest singular value of W
  sigma = jnp.einsum("i,ij,j", u, W, v)

  # Scale coefficient to account for the fact that sigma can be an under-estimate.
  factor = jnp.where(scale < sigma, scale/sigma, 1.0)

  return W*factor, u, v

################################################################################################################

def spectral_norm_conv_iter(W, uv, stride, padding):
  """ Perform a single spectral norm iteration """
  u, v = uv

  # Perform the spectral norm iterations
  v = jax.lax.conv_transpose(u[None], W, strides=stride, padding=padding, dimension_numbers=("NHWC", "HWIO", "NHWC"), transpose_kernel=True)[0]
  v *= jax.lax.rsqrt(jnp.sum(v**2) + 1e-12)

  u = jax.lax.conv_general_dilated(v[None], W, window_strides=stride, padding=padding, dimension_numbers=("NHWC", "HWIO", "NHWC"))[0]
  u *= jax.lax.rsqrt(jnp.sum(u**2) + 1e-12)

  return (u, v)

@partial(jit, static_argnums=(2, 3, 5))
def spectral_norm_conv_apply(W: jnp.ndarray,
                             u: jnp.ndarray,
                             stride: Sequence[int],
                             padding: Union[str, Sequence[Tuple[int, int]]],
                             scale: float,
                             n_iters: int):
  """ Perform n_iters single spectral norm iterations """
  height, width, C_in, C_out = W.shape
  assert u.shape == (height, width, C_out)

  # v is set inside the loop, so just pass in a dummy value.
  v = jnp.zeros((height, width, C_in))

  # Perform the spectral norm iterations
  body = partial(spectral_norm_conv_iter, stride=stride, padding=padding)
  fp = jax.jit(util.fixed_point, static_argnums=(0,))
  (u, v) = fp(body, W, (u, v), n_iters)

  # Estimate the largest singular value of W
  Wv = jax.lax.conv_general_dilated(v[None], W, window_strides=stride, padding=padding, dimension_numbers=("NHWC", "HWIO", "NHWC"))[0]
  sigma = jnp.sum(u*Wv)

  # Scale coefficient to account for the fact that sigma can be an under-estimate.
  factor = jnp.where(scale < sigma, scale/sigma, 1.0)

  return W*factor, u
