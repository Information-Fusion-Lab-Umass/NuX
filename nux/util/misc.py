import jax.numpy as jnp
from jax import jit, random
from functools import partial, reduce
import numpy as np
import jax
# import haiku as hk
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import jax.lax as lax

def scaled_weight_standardization_conv(w, gain=None, eps=1e-4):
  fan_in = list_prod(w.shape[:-1])
  mean = jnp.mean(w, axis=(0, 1, 2), keepdims=True)
  var = jnp.var(w, axis=(0, 1, 2), keepdims=True)
  weight = (w - mean)*jax.lax.rsqrt(var*fan_in + eps)
  if gain is not None:
    weight *= gain
  return weight

################################################################################################################

def linear_lr(i, lr=1e-4):
  return lr

def linear_warmup_schedule(i, warmup=1000, lr_decay=1.0):
  return jnp.where(i < warmup,
                   i/warmup,
                   (lr_decay**(i - warmup)))

def linear_warmup_lr_schedule(i, warmup=1000, lr_decay=1.0, lr=1e-4):
  return jnp.where(i < warmup,
                   lr*i/warmup,
                   lr*(lr_decay**(i - warmup)))

################################################################################################################

def conv(w, x):
  no_batch = False
  if x.ndim == 3:
    no_batch = True
    x = x[None]
  out = jax.lax.conv_general_dilated(x,
                                     w,
                                     window_strides=(1, 1),
                                     padding="SAME",
                                     lhs_dilation=(1, 1),
                                     rhs_dilation=(1, 1),
                                     dimension_numbers=("NHWC", "HWIO", "NHWC"))
  if no_batch:
    out = out[0]
  return out

################################################################################################################

def tree_shapes(pytree):
  return jax.tree_util.tree_map(lambda x:x.shape, pytree)

################################################################################################################

def list_prod(x):
  # We might run into JAX tracer issues if we do something like multiply the elements of a shape tuple with jnp
  return np.prod(x)

################################################################################################################

@jit
def whiten(x):
  U, s, VT = jnp.linalg.svd(x, full_matrices=False)
  return jnp.dot(U, VT)

################################################################################################################

def broadcast_to_first_axis(x, ndim):
  if x.ndim == 0:
    return x
  return jnp.expand_dims(x, axis=tuple(range(1, ndim)))

def last_axes(shape):
  return tuple(range(-1, -1 - len(shape), -1))

def get_reduce_axes(axes, ndim, offset=0):
  if isinstance(axes, int):
    axes = (axes,)
  keep_axes = [ax%ndim for ax in axes]
  reduce_axes = tuple([ax + offset for ax in range(ndim) if ax not in keep_axes])
  return reduce_axes

def index_list(shape, axis):
  ndim = len(shape)
  axis = [ax%ndim for ax in axis]
  shapes = [s for i, s in enumerate(shape) if i in axis]
  return tuple(shapes)

################################################################################################################

def batched_vdot(x, y, x_shape):
  assert x.shape == y.shape
  sum_axes = last_axes(x_shape)
  return jnp.sum(x*y, axis=sum_axes)

################################################################################################################

def constrain_log_scale(log_x):
  # return jax.nn.log_sigmoid(log_x)
  return jnp.logaddexp(jax.nn.log_sigmoid(log_x), -7)

################################################################################################################

def square_plus(x, gamma=0.5):
  # https://arxiv.org/pdf/1901.08431.pdf
  out = 0.5*(x + jnp.sqrt(x**2 + 4*gamma))
  out = jnp.maximum(out, 0.0)
  return out

def square_sigmoid(x, gamma=0.5):
  # Derivative of proximal relu.  Basically sigmoid without saturated gradients.
  return 0.5*(1 + x*jax.lax.rsqrt(x**2 + 4*gamma))

def square_swish(x, gamma=0.5):
  x2 = x**2
  out = 0.5*(x + x2*jax.lax.rsqrt(x2 + 4*gamma))
  return out

################################################################################################################

def str_to_nonlinearity(name):
  if name == "relu":
    nonlinearity = jax.nn.relu
  elif name == "tanh":
    nonlinearity = jnp.tanh
  elif name == "sigmoid":
    nonlinearity = jax.nn.sigmoid
  elif name == "swish":
    nonlinearity = jax.nn.swish
  elif name == "lipswish":
    nonlinearity = lambda x: jax.nn.swish(x)/1.1
  elif name == "square_lipswish":
    nonlinearity = lambda x: square_swish(x, gamma=0.5)/(0.5 + 2/9*np.sqrt(6))
  elif name == "square_swish":
    nonlinearity = square_swish
  elif name == "square_plus":
    nonlinearity = square_plus
  else:
    assert 0, "Invalid nonlinearity"

  return nonlinearity

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

################################################################################################################

class _NeedsInitialization(Exception):
    pass

def check_if_parameter_exists(name):
  exists = True
  try:
    def init(shape, dtype):
      raise _NeedsInitialization
    hk.get_parameter(name, (), float, init=init)
  except _NeedsInitialization:
    exists = False
  except AssertionError:
    exists = True

  return exists

################################################################################################################

def only_gradient(x):
  return x - jax.lax.stop_gradient(x)

################################################################################################################

def mean_and_std(x, axis=-1, keepdims=False):
  mean = jnp.mean(x, axis=axis, keepdims=keepdims)
  std = jnp.std(x, axis=axis, keepdims=keepdims)
  return mean, std

def mean_and_inverse_std(x, axis=-1, keepdims=False):
  mean = jnp.mean(x, axis=axis, keepdims=keepdims)
  mean_sq = jnp.mean(lax.square(x), axis=axis, keepdims=keepdims)
  var = mean_sq - lax.square(mean)
  inv_std = lax.rsqrt(var + 1e-6)
  return mean, inv_std

################################################################################################################

# https://github.com/deepmind/dm-haiku/issues#issuecomment-611203193
from typing import Any, NamedTuple

class Box(NamedTuple):
  value: Any
  shape = property(fget=lambda _: ())
  dtype = jnp.float32

def get_state_tree(name, init):
  return hk.get_state(name, (), jnp.float32, init=lambda *_: Box(init())).value

def set_state_tree(name, val):
  return hk.set_state(name, Box(val))

def get_parameter_tree(name, init):
  return hk.get_parameter(name, (), jnp.float32, init=lambda *_: Box(init())).value
