from jax.config import config
# config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import nux.util as util
from jax.flatten_util import ravel_pytree
import einops

################################################################################################################

def im2col_fun(x,
               filter_shape: Sequence[int],
               stride: Sequence[int],
               padding: Union[str, Sequence[Tuple[int, int]]],
               lhs_dilation: Sequence[int],
               rhs_dilation: Sequence[int],
               dimension_numbers: Sequence[str]):
  # Turn (H,W,C_in) into (H,W,C_in*Kx*Ky) so that we can just do
  # a matrix multiply with filter.reshape((C_in*Kx*Ky,C_out)) to
  # get the convolution result of shape (H,W,C_out).
  three_dim = False
  if x.ndim == 3:
    x = x[None]
    three_dim = True
  out = jax.lax.conv_general_dilated_patches(x,
                                             filter_shape=filter_shape,
                                             window_strides=stride,
                                             padding=padding,
                                             lhs_dilation=lhs_dilation,
                                             rhs_dilation=rhs_dilation,
                                             dimension_numbers=dimension_numbers)
  if three_dim:
    out = out[0]
  return out

################################################################################################################

def im2col_conv(x_i2c, k_i2c, mask, w):
  x_i2c = x_i2c*mask
  if k_i2c is not None:
    return jnp.einsum("...hwiuv,...hwouv,uvio->...hwo", x_i2c, k_i2c, w, optimize=True)
  return jnp.einsum("...hwiuv,uvio->...hwo", x_i2c, w, optimize=True)

def pac_features_se(im2col, f, s, t):
  f_i2c = im2col(f)
  f_diff = f_i2c - f[...,None,None]
  summed = jnp.sum(f_diff**2, axis=-3)
  k_i2c = jnp.exp(-0.5*s[...,None,None,:]*summed[...,None])
  k_i2c *= t[...,None,None,:]
  k_i2c = einops.rearrange(k_i2c, "... h w u v c -> ... h w c u v")
  return k_i2c

def pac_base(im2col, theta, w):
  Kx, _, _, C = w.shape
  pad = Kx//2

  f, s, t = theta[...,:-2*C], theta[...,-2*C:-C], theta[...,-C:]
  f = util.square_sigmoid(f, gamma=1.0)*2 - 1.0
  t = util.square_sigmoid(t, gamma=1.0)*1.0
  s = util.square_sigmoid(s, gamma=1.0)*1.0

  # Generate the convolutional kernel from the features
  k_i2c = pac_features_se(im2col, f, s, t)

  # Get the diagonal of the transformation
  diag_jacobian = k_i2c[...,pad,pad]*w[pad,pad,jnp.arange(C),jnp.arange(C)]
  return k_i2c, diag_jacobian

def pac_ldu_mvp(x, theta, w, order, inverse=False, **im2col_kwargs):
  H, W, C = x.shape[-3:]
  Kx, Ky = w.shape[:2]
  assert Kx == Ky
  pad = Kx//2

  def im2col(x):
    return im2col_fun(x, **im2col_kwargs).reshape(x.shape + (Kx, Ky))

  # Convert the imageto the patches view
  x_i2c = im2col(x)
  if theta is not None:
    k_i2c, diag_jacobian = pac_base(im2col, theta, w)
  else:
    diag_jacobian = w[pad,pad,jnp.arange(C),jnp.arange(C)]
    diag_jacobian = jnp.broadcast_to(diag_jacobian, x.shape)
    k_i2c = None

  # For autoregressive convs
  order_i2c = im2col(order)
  upper_mask = order[...,None,None] >= order_i2c
  lower_mask = ~upper_mask

  # Compute the output using the LDU decomposition
  if inverse == False:
    z = im2col_conv(x_i2c, k_i2c, upper_mask, jnp.triu(w))
    z_i2c = im2col(z)
    z = im2col_conv(z_i2c, k_i2c, lower_mask, w) + z
  else:
    def mvp(mask, x):
      x_i2c = im2col(x)
      return im2col_conv(x_i2c, k_i2c, mask, w) + x
    z = util.weighted_jacobi(partial(mvp, lower_mask), x)

    def mvp(mask, x):
      x_i2c = im2col(x)
      return im2col_conv(x_i2c, k_i2c, mask, jnp.triu(w))
    z = util.weighted_jacobi(partial(mvp, upper_mask), z, diagonal=diag_jacobian)

  return z, diag_jacobian

################################################################################################################

class PACFlow():

  def __init__(self,
               feature_dim: int=8,
               kernel_size: int=5,
               order_type: str="s_curve",
               pixel_adaptive: bool=True,
               zero_init: bool=True
  ):
    """
    """
    assert kernel_size%2 == 1
    self.kernel_shape   = (kernel_size, kernel_size)
    self.feature_dim    = feature_dim
    self.order_type     = order_type
    self.zero_init      = zero_init
    self.pixel_adaptive = pixel_adaptive

  def make_mvp(self, theta, w, inverse=False):

    Kx, Ky = self.kernel_shape
    pad = Kx//2
    self.im2col_kwargs = dict(filter_shape=self.kernel_shape,
                              stride=(1, 1),
                              padding=((pad, Kx - pad - 1), (pad, Ky - pad - 1)),
                              lhs_dilation=(1, 1),
                              rhs_dilation=(1, 1),
                              dimension_numbers=("NHWC", "HWIO", "NHWC"))

    H, W, C = self.x_shape
    order_shape = H, W, 1

    if self.order_type == "raster":
      order = jnp.arange(1, 1 + util.list_prod(order_shape)).reshape(order_shape)
    elif self.order_type == "s_curve":
      order = jnp.arange(1, 1 + util.list_prod(order_shape)).reshape(order_shape)
      order = order.at[::2,:,:].set(order[::2,:,:][:,::-1])

    order *= 1.0

    mvp = partial(pac_ldu_mvp,
                  theta=theta,
                  w=w,
                  order=order,
                  inverse=inverse,
                  **self.im2col_kwargs)

    if inverse == False:
      mvp = jax.checkpoint(mvp)
    return mvp

  def jac(self, x):
    x_flat, unflatten = jax.flatten_util.ravel_pytree(x)

    def flat_mvp(x_flat):
      x = unflatten(x_flat)
      z = self.mvp(x)[0]
      return jax.flatten_util.ravel_pytree(z)[0]

    return jax.jacobian(flat_mvp)(x_flat)

  def get_params(self):
    if self.pixel_adaptive == False:
      return dict(w=self.w)
    return dict(w=self.w, theta=self.theta)

  @property
  def coupling_param_keys(self):
    if self.pixel_adaptive == False:
      return ()
    return ("theta",)

  def get_param_dim(self, dim):
    if self.pixel_adaptive == False:
      return 0
    return self.feature_dim + 2*dim

  def extract_coupling_params(self, theta):
    if self.pixel_adaptive == False:
      return ()
    return (theta,)

  def __call__(self, x, params=None, inverse=False, rng_key=None, **kwargs):
    self.x_shape = x.shape[1:]
    Kx, Ky = self.kernel_shape
    filter_size = Kx*Ky
    H, W, C = self.x_shape

    # Initialize w
    if params is None or "w" not in params or params["w"] is None:
      self.w = random.normal(rng_key, shape=self.kernel_shape + (C, C))
      if self.zero_init:
        pad = Kx//2
        self.w = self.w.at[pad,pad,jnp.arange(C),jnp.arange(C)].set(1.0)
    else:
      self.w = params["w"]

    if self.pixel_adaptive == True:
      if params is None:
        self.theta = random.normal(rng_key, shape=(H, W, 2*C + self.feature_dim))
      else:
        self.theta = params["theta"]
    else:
      self.theta = None

    # Apply the linear function
    if inverse == False:
      self.mvp = self.make_mvp(self.theta, self.w)
      z, diag_jacobian = self.mvp(x)
    else:
      self.mvp = self.make_mvp(self.theta, self.w, inverse=True)
      z, diag_jacobian = self.mvp(x)

    # Estimate the true log det
    if self.theta is not None:
      flat_diag = diag_jacobian.reshape(self.theta.shape[:-3] + (-1,))
    else:
      flat_diag = diag_jacobian.reshape(x.shape[:1] + (-1,))

    log_det = jnp.log(jnp.abs(flat_diag)).sum(axis=-1)
    log_det = jnp.broadcast_to(log_det, x.shape[:1])

    return z, log_det

class EmergingConv(PACFlow):
  def __init__(self,
               kernel_size: int=5,
               order_type: str="s_curve",
               zero_init: bool=True
  ):
    super().__init__(feature_dim=None,
                     kernel_size=kernel_size,
                     order_type=order_type,
                     pixel_adaptive=False,
                     zero_init=zero_init)

################################################################################################################

if __name__ == "__main__":

  from debug import *
  import matplotlib.pyplot as plt

  rng_key = random.PRNGKey(0)
  x_shape = (4, 4, 2)
  x = random.normal(rng_key, shape=(2,) + x_shape)
  flow = PACFlow(feature_dim=8,
                 kernel_size=5,
                 order_type="s_curve",
                 pixel_adaptive=False,
                 zero_init=False)
  z, log_det = flow(x, rng_key=rng_key)
  params = flow.get_params()

  reconstr, _ = flow(z, params, inverse=True)
  z2, _ = flow(reconstr, params, inverse=False)
  # assert jnp.allclose(x, reconstr, atol=1e-5)

  def jac(x):
    flat_x, unflatten = ravel_pytree(x)
    def flat_call(flat_x):
      x = unflatten(flat_x)
      z, _ = flow(x[None], params=params)
      return z.ravel()

    return jax.jacobian(flat_call)(flat_x)

  J = jax.vmap(jac)(x)
  true_log_det = jnp.linalg.slogdet(J)[1]

  # assert jnp.allclose(log_det, true_log_det)
  import pdb; pdb.set_trace()
