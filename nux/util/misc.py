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

# def cheat_shift_scale_init(x, param, name, shape, dtype):

#   def shift_init(shape, dtype):
#     if x.ndim == len(shape):
#       return jnp.zeros(shape, dtype)

#     axes = tuple(jnp.arange(len(x.shape) - len(shape)))
#     return jnp.mean(x, axis=axes)

#   def log_scale_init(shape, dtype):
#     if x.ndim == len(shape):
#       return jnp.zeros(shape, dtype)

#     axes = tuple(jnp.arange(len(x.shape) - len(shape)))
#     return jnp.log(jnp.std(z, axis=axes) + 1e-5)

#   shift = hk.get_parameter(f"{name}_shift", shape=shape, dtype=dtype, shift_init)
#   scale = hk.get_parameter(f"{name}_scale", shape=shape, dtype=dtype, log_scale_init)

#   return scale*param + shift