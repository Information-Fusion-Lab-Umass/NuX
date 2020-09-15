import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
import nux.util as util
from typing import Optional, Mapping, Callable, Sequence, Any, Union, Tuple
import haiku as hk

__all__ = ["spectral_norm_apply",
           "spectral_norm_conv_apply",
           "spectral_norm_tree",
           "initialize_spectral_norm_u_tree",
           "check_spectral_norm",
           "spectral_norm_apply_in_context"]

################################################################################################################

@jit
def spectral_norm_body(carry: Sequence[jnp.ndarray],
                       inputs: int):
  """ Perform a single spectral norm iteration """
  W, u, v = carry

  # Perform the spectral norm iterations
  v = W.T@u
  v *= jax.lax.rsqrt(jnp.dot(v, v) + 1e-12)

  u = W@v
  u *= jax.lax.rsqrt(jnp.dot(u, u) + 1e-12)

  return (W, u, v), inputs

@partial(jit, static_argnums=(3,))
def spectral_norm_apply(W: jnp.ndarray,
                        u: jnp.ndarray,
                        scale: float,
                        n_iters: int):
  """ Perform n_iters single spectral norm iterations """

  # v is set inside the loop, so just pass in a dummy value.
  v = jnp.zeros((W.shape[1],))

  # Perform the spectral norm iterations
  (W, u, v), _ = jax.lax.scan(spectral_norm_body, (W, u, v), jnp.arange(n_iters))

  # We don't want to backprop through the spectral norm iterations!
  u = jax.lax.stop_gradient(u)
  v = jax.lax.stop_gradient(v)

  # Estimate the largest singular value of W
  sigma = jnp.einsum("i,ij,j", u, W, v)

  # Scale coefficient to account for the fact that sigma can be an under-estimate.
  factor = jnp.where(scale < sigma, scale/sigma, 1.0)

  return W*factor, u

################################################################################################################

@partial(jit, static_argnums=(3,))
def spectral_norm_tree(pytree: Any,
                       u_tree: Any,
                       scale: float,
                       n_iters: int):
  """ Apply spectral normalization to the leaves of a pytree """

  # Apply spectral norm to leaves of a tree if the leaf is a matrix
  def apply_spectral_norm(val, u):
    return spectral_norm_apply(val, u, scale, n_iters) if val.ndim == 2 else (val, u)

  # tree_multimap_multiout will unzip the result of tree_multimap
  return util.tree_multimap_multiout(apply_spectral_norm, pytree, u_tree)

def initialize_spectral_norm_u_tree(key, pytree):
  """ Initialize the vectors that we'll need for each leaf of pytree """
  key_tree = util.key_tree_like(key, pytree)

  # Initialize the u tree
  def gen_u(key, val):
    return random.normal(key, (val.shape[0],)) if val.ndim == 2 else ()
  return jax.tree_util.tree_multimap(gen_u, key_tree, pytree)

def check_spectral_norm(pytree):
  """ Check the spectral norm of the leaves of pytree """

  def spectral_norm_apply(val):
    return jnp.linalg.norm(val, ord=2) if val.ndim == 2 else 0

  return jax.tree_util.tree_map(spectral_norm_apply, pytree)

################################################################################################################

def spectral_norm_apply_in_context(params, rng, scale, spectral_norm_iters):
  # Don't need a new hk.transform to use this

  # Initialize the spectral norm tree for the params of the residual block
  def u_init(shape, dtype):
    rng = hk.next_rng_key()
    u_tree = initialize_spectral_norm_u_tree(rng, params)
    return u_tree

  u_tree = hk.get_state("u_tree", shape=(), dtype=(), init=u_init)

  # Apply spectral normalization
  params, u_tree = spectral_norm_tree(params, u_tree, scale, spectral_norm_iters)
  hk.set_state("u_tree", u_tree)

  return params

################################################################################################################

def spectral_norm_conv_body(carry: Sequence[jnp.ndarray],
                            inputs: int,
                            stride: Sequence[int],
                            padding: Union[str, Sequence[Tuple[int, int]]]):
  """ Perform a single spectral norm iteration """
  W, u, v = carry

  # Perform the spectral norm iterations
  v = jax.lax.conv_transpose(u[None], W, strides=stride, padding=padding, dimension_numbers=("NHWC", "HWIO", "NHWC"), transpose_kernel=True)[0]
  v *= jax.lax.rsqrt(jnp.sum(v**2) + 1e-12)

  u = jax.lax.conv_general_dilated(v[None], W, window_strides=stride, padding=padding, dimension_numbers=("NHWC", "HWIO", "NHWC"))[0]
  u *= jax.lax.rsqrt(jnp.sum(u**2) + 1e-12)

  return (W, u, v), inputs

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
  body = partial(spectral_norm_conv_body, stride=stride, padding=padding)
  (W, u, v), _ = jax.lax.scan(body, (W, u, v), jnp.arange(n_iters))

  # We don't want to backprop through the spectral norm iterations!
  u = jax.lax.stop_gradient(u)
  v = jax.lax.stop_gradient(v)

  # Estimate the largest singular value of W
  Wv = jax.lax.conv_general_dilated(v[None], W, window_strides=stride, padding=padding, dimension_numbers=("NHWC", "HWIO", "NHWC"))[0]
  sigma = jnp.sum(u*Wv)

  # Scale coefficient to account for the fact that sigma can be an under-estimate.
  factor = jnp.where(scale < sigma, scale/sigma, 1.0)

  return W*factor, u
