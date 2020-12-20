import jax.numpy as jnp
from jax import jit, random
from functools import partial, reduce
import numpy as np
import jax
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence, Any

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

def get_plot_bounds(data):
  (xmin, ymin), (xmax, ymax) = data.min(axis=0), data.max(axis=0)
  xspread, yspread = xmax - xmin, ymax - ymin
  xmin -= 0.1*xspread
  xmax += 0.1*xspread
  ymin -= 0.1*yspread
  ymax += 0.1*yspread
  return (xmin, xmax), (ymin, ymax)