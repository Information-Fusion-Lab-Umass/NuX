import jax.numpy as jnp
from jax import jit, random
from functools import partial, reduce
import numpy as np
import jax
import haiku as hk
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import jax.lax as lax
from jax._src.lax.lax import _conv_transpose_padding

def transpose_conv_kwargs(*,
                          batched_input_shape,
                          kernel_shape,
                          stride,
                          padding,
                          input_dilation,
                          kernel_dilation,
                          dimension_numbers):
  ndims = len(batched_input_shape)
  kernel_dim = len(kernel_shape)

  dimension_numbers = jax.lax.conv_dimension_numbers(batched_input_shape,
                                                     kernel_shape,
                                                     dimension_numbers)

  # Get the spatial dims
  k_shape = np.take(kernel_shape, dimension_numbers.rhs_spec)
  kernel_spatial_dims = k_shape[2:]

  if padding in ["SAME", "VALID"]:
    effective_k_size = map(lambda k, r: (k - 1)*r + 1,
                           kernel_spatial_dims,
                           kernel_dilation)
    padding = [_conv_transpose_padding(k, s, padding) for k, s in zip(effective_k_size, stride)]
  else:
    padding = padding

  return dict(stride=input_dilation,
              padding=padding,
              lhs_dilation=(1,)*(ndims - 2),
              rhs_dilation=stride,
              dimension_numbers=dimension_numbers)

################################################################################################################

def apply_conv(x: jnp.ndarray,
               w: jnp.ndarray,
               stride: Sequence[int],
               padding: Sequence[int],
               lhs_dilation: Sequence[int],
               rhs_dilation: Sequence[int],
               dimension_numbers: Sequence[str],
               transpose: bool,
               **kwargs):

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

  spatial_size = np.prod(filter_shape)
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

def im2col(x,
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
  out = conv_general_dilated_patches(x,
                                     filter_shape=filter_shape,
                                     window_strides=stride,
                                     padding=padding,
                                     lhs_dilation=lhs_dilation,
                                     rhs_dilation=rhs_dilation,
                                     dimension_numbers=dimension_numbers)
  if three_dim:
    out = out[0]
  return out

def apply_im2col_conv(x: jnp.ndarray,
                      w: jnp.ndarray,
                      filter_shape: Sequence[int],
                      stride: Sequence[int],
                      padding: Union[str, Sequence[Tuple[int, int]]],
                      lhs_dilation: Sequence[int],
                      rhs_dilation: Sequence[int],
                      dimension_numbers: Sequence[str],
                      transpose: bool,
                      **kwargs):
  H, W, C_in = x.shape[-3:]
  Kx, Ky = filter_shape
  C_out = w.shape[-1]

  # assert w.shape == (H, W, Kx, Ky, C_in, C_out)
  w = w.reshape((H, W, Kx*Ky*C_in, C_out))

  x_i2c = im2col(x,
                 filter_shape=filter_shape,
                 stride=stride,
                 padding=padding,
                 lhs_dilation=lhs_dilation,
                 rhs_dilation=rhs_dilation,
                 dimension_numbers=dimension_numbers)

  # if transpose:
  #   x_i2c = jax.ops.index_update(x_i2c, jax.ops.index[...,:], x_i2c[...,::-1])

  assert x_i2c.shape[-3:] == (H, W, C_in*Kx*Ky)

  if x.ndim == 3:
    out = jnp.einsum("hwi,hwio->hwo", x_i2c, w)
  else:
    out = jnp.einsum("bhwi,hwio->bhwo", x_i2c, w)

  # import pdb; pdb.set_trace()

  return out

################################################################################################################
