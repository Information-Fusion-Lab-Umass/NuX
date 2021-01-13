import jax.numpy as jnp
from jax import jit, random
from functools import partial, reduce
import numpy as np
import jax
import haiku as hk
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import jax.lax as lax

################################################################################################################

def apply_conv(x: jnp.ndarray,
               w: jnp.ndarray,
               stride: Sequence[int],
               padding: Sequence[int],
               lhs_dilation: Sequence[int],
               rhs_dilation: Sequence[int],
               dimension_numbers: Sequence[str],
               transpose: bool):

  if transpose == False:
    return jax.lax.conv_general_dilated(x,
                                        w,
                                        window_strides=stride,
                                        padding=padding,
                                        lhs_dilation=lhs_dilation,
                                        rhs_dilation=rhs_dilation,
                                        dimension_numbers=dimension_numbers)

  return jax.lax.conv_transpose(x,
                                w,
                                strides=stride,
                                padding=padding,
                                rhs_dilation=rhs_dilation,
                                dimension_numbers=dimension_numbers,
                                transpose_kernel=True)

################################################################################################################

def conv_general_dilated_patches(
    lhs: jnp.ndarray,
    filter_shape: Sequence[int],
    window_strides: Sequence[int],
    padding: Union[str, Sequence[Tuple[int, int]]],
    lhs_dilation: Sequence[int] = None,
    rhs_dilation: Sequence[int] = None,
    dimension_numbers: lax.ConvGeneralDilatedDimensionNumbers = None,
) -> jnp.ndarray:
  # This hasn't been added to JAX yet
  filter_shape = tuple(filter_shape)
  dimension_numbers = lax.conv_dimension_numbers(
      lhs.shape, (1, 1) + filter_shape, dimension_numbers)

  lhs_spec, rhs_spec, out_spec = dimension_numbers

  spatial_size = util.list_prod(filter_shape)
  n_channels = lhs.shape[lhs_spec[1]]

  # Move separate `lhs` spatial locations into separate `rhs` channels.
  rhs = jnp.eye(spatial_size, dtype=lhs.dtype).reshape(filter_shape * 2)

  rhs = rhs.reshape((spatial_size, 1) + filter_shape)
  rhs = jnp.tile(rhs, (n_channels,) + (1,) * (rhs.ndim - 1))
  rhs = jnp.moveaxis(rhs, (0, 1), (rhs_spec[0], rhs_spec[1]))

  out = lax.conv_general_dilated(
      lhs=lhs,
      rhs=rhs,
      window_strides=window_strides,
      padding=padding,
      lhs_dilation=lhs_dilation,
      rhs_dilation=rhs_dilation,
      dimension_numbers=dimension_numbers,
      precision=None,
      feature_group_count=n_channels
  )
  return out

def im2col_same_shape(x, filter_shape):
  # Turn (H,W,C_in) into (H,W,C_in*Kx*Ky) so that we can just do
  # a matrix multiply with filter.reshape((C_in*Kx*Ky,C_out)) to
  # get the convolution result of shape (H,W,C_out).
  if x.ndim == 3:
    x = x[None]
  out = conv_general_dilated_patches(x[None],
                                     filter_shape=filter_shape,
                                     window_strides=(1, 1),
                                     padding="SAME",
                                     lhs_dilation=(1, 1),
                                     rhs_dilation=(1, 1),
                                     dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
  if x.ndim == 3:
    out = out[0]
  return out

def apply_im2col_conv(x: jnp.ndarray,
                      w: jnp.ndarray,
                      filter_shape: Sequence[int]):
  H, W, C_in = x.shape[-3:]
  Kx, Ky = filter_shape
  assert w.shape == (H, W, C_in*Kx*Ky, C_out)

  x_i2c = im2col(x, filter_shape)
  assert x_i2c.shape[-3:] == (H, W, C_in*Kx*Ky)

  return jnp.einsum("...hwi,io->hwo", x_i2c, w)

################################################################################################################

def list_prod(x):
  # We might run into JAX tracer issues if we do something like multiply the elements of a shape tuple with jnp
  return np.prod(x)
  # return reduce(mul, x, 1)

################################################################################################################

@jit
def whiten(x):
  U, s, VT = jnp.linalg.svd(x, full_matrices=False)
  return jnp.dot(U, VT)

################################################################################################################

def broadcast_to_first_axis(x, ndim):
  return jnp.expand_dims(x, axis=tuple(range(1, ndim)))

def last_axes(shape):
  return tuple(range(-1, -1 - len(shape), -1))

################################################################################################################

def constrain_log_scale(log_x):
  # return jax.nn.log_sigmoid(log_x)
  return jnp.logaddexp(jax.nn.log_sigmoid(log_x), -7)

################################################################################################################

def proximal_relu(x, gamma=0.5):
  # https://arxiv.org/pdf/1901.08431.pdf
  return 0.5*(x + jnp.sqrt(x**2 + 4*gamma))

def proximal_sigmoid(x, gamma=0.5):
  # Derivative of proximal relu.  Basically sigmoid without saturated gradients.
  return 0.5*(1 + x*jax.lax.rsqrt(x**2 + 4*gamma))

################################################################################################################

def get_plot_bounds(data):
  (xmin, ymin), (xmax, ymax) = data.min(axis=0), data.max(axis=0)
  xspread, yspread = xmax - xmin, ymax - ymin
  xmin -= 0.1*xspread
  xmax += 0.1*xspread
  ymin -= 0.1*yspread
  ymax += 0.1*yspread
  return (xmin, xmax), (ymin, ymax)

################################################################################################################

# There is a bug in logsumexp!
def lse(a, axis=None, b=None, keepdims=False, return_sign=False):
  if b is not None:
    a, b = jnp.broadcast_arrays(a, b)
    a = a + jnp.where(b, jnp.log(jnp.abs(b)), -jnp.inf)
    b = jnp.sign(b)

  return jax.scipy.special.logsumexp(a, axis=axis, b=b, keepdims=keepdims, return_sign=return_sign)
