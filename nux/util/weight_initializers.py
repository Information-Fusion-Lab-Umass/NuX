import jax.numpy as jnp
from jax import jit, random, vmap
from functools import partial
import jax
import nux.util as util
import haiku as hk
import nux.util.spectral_norm as sn
from typing import Optional, Mapping, Callable, Sequence, Any
import haiku._src.base as hk_base

################################################################################################################

def weight_with_spectral_norm(x: jnp.ndarray,
                              out_dim: int,
                              name_suffix: str="",
                              w_init: Callable=None,
                              b_init: Callable=None,
                              is_training: bool=True,
                              update_params: bool=True,
                              use_bias: bool=True,
                              force_in_dim: Optional=None,
                              max_singular_value: float=0.99,
                              max_power_iters: int=1,
                              **kwargs):
  in_dim, dtype = x.shape[-1], x.dtype
  if force_in_dim:
    in_dim = force_in_dim

  w = hk.get_parameter(f"w_{name_suffix}", (out_dim, in_dim), dtype, init=w_init)
  if use_bias:
    b = hk.get_parameter(f"b_{name_suffix}", (out_dim,), dtype, init=b_init)

  u = hk.get_state(f"u_{name_suffix}", (out_dim,), dtype, init=hk.initializers.RandomNormal())
  v = hk.get_state(f"v_{name_suffix}", (in_dim,), dtype, init=hk.initializers.RandomNormal())
  w, u, v = sn.spectral_norm_apply(w,
                                   u,
                                   v,
                                   max_singular_value,
                                   max_power_iters,
                                   update_params)

  running_init_fn = not hk_base.params_frozen()
  if running_init_fn:
      w, u, v = sn.spectral_norm_apply(w,
                                       u,
                                       v,
                                       max_singular_value,
                                       None,
                                       True)

  if is_training == True or running_init_fn:
    hk.set_state(f"u_{name_suffix}", u)
    hk.set_state(f"v_{name_suffix}", v)

  if use_bias:
    return w, b
  return w

def conv_weight_with_spectral_norm(x: jnp.ndarray,
                                   kernel_shape: Sequence[int],
                                   out_channel: int,
                                   name_suffix: str="",
                                   w_init: Callable=None,
                                   b_init: Callable=None,
                                   use_bias: bool=True,
                                   is_training: bool=True,
                                   update_params: bool=True,
                                   max_singular_value: float=0.8,
                                   max_power_iters: int=1,
                                   **conv_kwargs):
  batch_size, H, W, C = x.shape
  w_shape = kernel_shape + (C, out_channel)

  w = hk.get_parameter(f"w_{name_suffix}", w_shape, x.dtype, init=w_init)
  if use_bias:
    b = hk.get_parameter(f"b_{name_suffix}", (out_channel,), init=b_init)

  u = hk.get_state(f"u_{name_suffix}", kernel_shape + (out_channel,), init=hk.initializers.RandomNormal())
  v = hk.get_state(f"v_{name_suffix}", kernel_shape + (C,), init=hk.initializers.RandomNormal())
  w, u, v = sn.spectral_norm_conv_apply(w,
                                        u,
                                        v,
                                        conv_kwargs["stride"],
                                        conv_kwargs["padding"],
                                        max_singular_value,
                                        max_power_iters,
                                        update_params)

  # Run for a lot of steps when we're first initializing
  running_init_fn = not hk_base.params_frozen()
  if running_init_fn:
    w, u, v = sn.spectral_norm_conv_apply(w,
                                          u,
                                          v,
                                          conv_kwargs["stride"],
                                          conv_kwargs["padding"],
                                          max_singular_value,
                                          None,
                                          True)

  if is_training == True or running_init_fn:
    hk.set_state(f"u_{name_suffix}", u)
    hk.set_state(f"v_{name_suffix}", v)

  if use_bias:
    b = hk.get_parameter(f"b_{name_suffix}", (out_channel,), x.dtype, init=b_init)

  if use_bias:
    return w, b
  return w

################################################################################################################

def weight_with_weight_norm(x: jnp.ndarray,
                            out_dim: int,
                            name_suffix: str="",
                            w_init: Callable=None,
                            b_init: Callable=None,
                            is_training: bool=True,
                            use_bias: bool=True,
                            force_in_dim: Optional=None,
                            **kwargs):
  in_dim, dtype = x.shape[-1], x.dtype
  if force_in_dim:
    in_dim = force_in_dim

  w = hk.get_parameter(f"w_{name_suffix}", (out_dim, in_dim), dtype, init=hk.initializers.RandomNormal(stddev=0.05))
  w *= jax.lax.rsqrt(jnp.sum(w**2, axis=1))[:,None]

  def g_init(shape, dtype):
    if x.ndim == 1:
      return jnp.ones(shape, dtype=dtype)
    t = jnp.dot(x, w.T)
    return 1/(jnp.std(t, axis=0) + 1e-5)

  def b_init(shape, dtype):
    if x.ndim == 1:
      return jnp.zeros(shape, dtype=dtype)
    t = jnp.dot(x, w.T)
    return -jnp.mean(t, axis=0)/(jnp.std(t, axis=0) + 1e-5)

  g = hk.get_parameter(f"g_{name_suffix}", (out_dim,), dtype, init=g_init)
  if use_bias:
    b = hk.get_parameter(f"b_{name_suffix}", (out_dim,), dtype, init=b_init)

  w *= g[:,None]

  if use_bias:
    return w, b
  return w

def conv_weight_with_weight_norm(x: jnp.ndarray,
                                 kernel_shape: Sequence[int],
                                 out_channel: int,
                                 name_suffix: str="",
                                 w_init: Callable=None,
                                 b_init: Callable=None,
                                 use_bias: bool=True,
                                 is_training: bool=True,
                                 **conv_kwargs):
  batch_size, H, W, C = x.shape
  w_shape = kernel_shape + (C, out_channel)

  w = hk.get_parameter("w", w_shape, x.dtype, init=hk.initializers.RandomNormal(stddev=0.05))
  w *= jax.lax.rsqrt((w**2).sum(axis=(0, 1, 2)))[None,None,None,:]

  def g_init(shape, dtype):
    t = util.apply_conv(x, w, **conv_kwargs)
    return 1/(jnp.std(t, axis=(0, 1, 2)) + 1e-5)

  def b_init(shape, dtype):
    t = util.apply_conv(x, w, **conv_kwargs)
    return -jnp.mean(t, axis=(0, 1, 2))/(jnp.std(t, axis=(0, 1, 2)) + 1e-5)

  g = hk.get_parameter(f"g_{name_suffix}", (out_channel,), x.dtype, init=g_init)
  if use_bias:
    b = hk.get_parameter(f"b_{name_suffix}", (out_channel,), x.dtype, init=b_init)

  w *= g[None,None,None,:]

  if use_bias:
    return w, b
  return w
