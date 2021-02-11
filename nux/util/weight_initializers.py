import jax.numpy as jnp
from jax import jit, random, vmap
from functools import partial
import jax
import nux.util as util
import haiku as hk
import nux.util.spectral_norm as sn
from typing import Optional, Mapping, Callable, Sequence, Any
import haiku._src.base as hk_base
import types

################################################################################################################

def apply_sn(*,
             mvp,
             mvpT,
             w_shape,
             b_shape,
             out_shape,
             dtype,
             w_init,
             b_init,
             name_suffix,
             is_training,
             use_bias,
             max_singular_value,
             max_power_iters,
             use_proximal_gradient=False,
             monitor_progress=False,
             monitor_iters=20,
             return_sigma=False,
             **kwargs):

  w_exists = util.check_if_parameter_exists(f"w_{name_suffix}")

  w = hk.get_parameter(f"w_{name_suffix}", w_shape, dtype, init=w_init)
  u = hk.get_state(f"u_{name_suffix}", out_shape, dtype, init=hk.initializers.RandomNormal())
  if use_proximal_gradient == False:
    zeta = hk.get_state(f"zeta_{name_suffix}", out_shape, dtype, init=hk.initializers.RandomNormal())
    state = (u, zeta)
  else:
    state = (u,)

  if use_proximal_gradient == False:
    estimate_max_singular_value = jax.jit(sn.max_singular_value, static_argnums=(0, 1))
  else:
    estimate_max_singular_value = jax.jit(sn.max_singular_value_no_grad, static_argnums=(0, 1))

  if w_exists == False:
    max_power_iters = 1000

  if monitor_progress:
    estimates = []

  for i in range(max_power_iters):
    sigma, *state = estimate_max_singular_value(mvp, mvpT, w, *state)
    if monitor_progress:
      estimates.append(sigma)

  if monitor_progress:
    sigma_for_test = sigma
    state_for_test = state
    for i in range(monitor_iters - max_power_iters):
      sigma_for_test, *state_for_test = estimate_max_singular_value(mvp, mvpT, w, *state_for_test)
      estimates.append(sigma_for_test)

    estimates = jnp.array(estimates)

    sigma_for_test = jax.lax.stop_gradient(sigma_for_test)
    state_for_test = jax.lax.stop_gradient(state_for_test)

  state = jax.lax.stop_gradient(state)

  if is_training == True or w_exists == False:
    u = state[0]
    hk.set_state(f"u_{name_suffix}", u)
    if use_proximal_gradient == False:
      zeta = state[1]
      hk.set_state(f"zeta_{name_suffix}", zeta)

  if return_sigma == False:
    factor = jnp.where(max_singular_value < sigma, max_singular_value/sigma, 1.0)
    w = w*factor
    w_ret = w
  else:
    w_ret = (w, sigma)

  if use_bias:
    b = hk.get_parameter(f"b_{name_suffix}", b_shape, dtype, init=b_init)
    ret = (w_ret, b)
  else:
    ret = w_ret

  if monitor_progress:
    ret = (ret, estimates)

  return ret

################################################################################################################

def weight_with_good_spectral_norm(x: jnp.ndarray,
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

  w_shape   = (out_dim, in_dim)
  b_shape   = (out_dim,)
  out_shape = (out_dim,)

  def mvp(A, x):
    return A@x

  def mvpT(A, x):
    return A.T@x

  return apply_sn(mvp=mvp,
                  mvpT=mvpT,
                  w_shape=w_shape,
                  b_shape=b_shape,
                  out_shape=out_shape,
                  dtype=dtype,
                  w_init=w_init,
                  b_init=b_init,
                  name_suffix=name_suffix,
                  is_training=is_training,
                  use_bias=use_bias,
                  max_singular_value=max_singular_value,
                  max_power_iters=max_power_iters)

def conv_weight_with_good_spectral_norm(x: jnp.ndarray,
                                        kernel_shape: Sequence[int],
                                        out_channel: int,
                                        name_suffix: str="",
                                        w_init: Callable=None,
                                        b_init: Callable=None,
                                        use_bias: bool=True,
                                        is_training: bool=True,
                                        update_params: bool=True,
                                        max_singular_value: float=0.95,
                                        max_power_iters: int=1,
                                        stride: Sequence[int]=(1, 1),
                                        padding: str="SAME",
                                        **kwargs):
  batch_size, H, W, C = x.shape
  dtype = x.dtype
  # stride, padding = kwargs["stride"], kwargs["padding"]

  w_shape   = kernel_shape + (C, out_channel)
  b_shape   = (out_channel,)
  out_shape = (H, W, out_channel)

  def mvp(A, x):
    return jax.lax.conv_general_dilated(x[None],
                                        A,
                                        window_strides=stride,
                                        padding=padding,
                                        dimension_numbers=("NHWC", "HWIO", "NHWC"))[0]

  def mvpT(A, x):
    return jax.lax.conv_transpose(x[None],
                                  A,
                                  strides=stride,
                                  padding=padding,
                                  dimension_numbers=("NHWC", "HWIO", "NHWC"),
                                  transpose_kernel=True)[0]

  return apply_sn(mvp=mvp,
                  mvpT=mvpT,
                  w_shape=w_shape,
                  b_shape=b_shape,
                  out_shape=out_shape,
                  dtype=dtype,
                  w_init=w_init,
                  b_init=b_init,
                  name_suffix=name_suffix,
                  is_training=is_training,
                  use_bias=use_bias,
                  max_singular_value=max_singular_value,
                  max_power_iters=max_power_iters,
                  **kwargs)

def i2c_conv_weight_with_good_spectral_norm(x: jnp.ndarray,
                                            kernel_shape: Sequence[int],
                                            out_channel: int,
                                            name_suffix: str="",
                                            w_init: Callable=None,
                                            b_init: Callable=None,
                                            use_bias: bool=True,
                                            is_training: bool=True,
                                            update_params: bool=True,
                                            max_singular_value: float=0.95,
                                            max_power_iters: int=1,
                                            **conv_kwargs):
  batch_size, H, W, C = x.shape
  Kx, Ky = kernel_shape
  dtype = x.dtype
  stride, padding = conv_kwargs["stride"], conv_kwargs["padding"]

  w_shape   = (H, W, Kx, Ky, C, out_channel)
  b_shape   = (out_channel,)
  out_shape = (H, W, out_channel)

  def mvp(A, x):
    assert x.ndim == 3
    out = util.apply_im2col_conv(x,
                                  A,
                                  filter_shape=kernel_shape,
                                  stride=stride,
                                  padding=padding,
                                  lhs_dilation=(1, 1),
                                  rhs_dilation=(1, 1),
                                  dimension_numbers=("NHWC", "HWIO", "NHWC"),
                                  transpose=False)
    assert out.ndim == 3
    return out

  def mvpT(A, y):
    assert y.ndim == 3
    input_image = types.SimpleNamespace(shape=(H, W, C), dtype=jnp.float32)
    mvpt = jax.linear_transpose(lambda x: mvp(A, x), input_image)
    out = mvpt(y)[0]
    assert out.ndim == 3
    return out

  return apply_sn(mvp=mvp,
                  mvpT=mvpT,
                  w_shape=w_shape,
                  b_shape=b_shape,
                  out_shape=out_shape,
                  dtype=dtype,
                  w_init=w_init,
                  b_init=b_init,
                  name_suffix=name_suffix,
                  is_training=is_training,
                  use_bias=use_bias,
                  max_singular_value=max_singular_value,
                  max_power_iters=max_power_iters)

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
                                   max_singular_value: float=0.95,
                                   max_power_iters: int=1,
                                   **conv_kwargs):
  batch_size, H, W, C = x.shape
  w_shape = kernel_shape + (C, out_channel)

  w = hk.get_parameter(f"w_{name_suffix}", w_shape, x.dtype, init=w_init)
  if use_bias:
    b = hk.get_parameter(f"b_{name_suffix}", (out_channel,), init=b_init)

  u = hk.get_state(f"u_{name_suffix}", (H, W, out_channel), init=hk.initializers.RandomNormal())
  v = hk.get_state(f"v_{name_suffix}", (H, W, C), init=hk.initializers.RandomNormal())
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
