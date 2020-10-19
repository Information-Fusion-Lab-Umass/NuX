import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
import haiku as hk
import nux.spectral_norm as sn
from typing import Optional, Mapping, Callable, Sequence, Any

__all__ = ["MLP"]

def data_dependent_param_init(x: jnp.ndarray,
                              out_dim: int,
                              name_suffix: str="",
                              w_init: Callable=None,
                              b_init: Callable=None,
                              is_training: bool=True,
                              parameter_norm: str=None):
  in_dim, dtype = x.shape[-1], x.dtype

  if parameter_norm == "spectral_norm":
    w = hk.get_parameter(f"w_{name_suffix}", (out_dim, in_dim), dtype, init=w_init)
    b = hk.get_parameter(f"b_{name_suffix}", (out_dim,), dtype, init=b_init)

    u = hk.get_state(f"u_{name_suffix}", (out_dim,), dtype, init=hk.initializers.RandomNormal())
    w, u = sn.spectral_norm_apply(w, u, 0.9, 1)
    if is_training == True:
      hk.set_state(f"u_{name_suffix}", u)

  elif parameter_norm == "weight_norm" and x.shape[0] > 1:
    w = hk.get_parameter(f"w_{name_suffix}", (out_dim, in_dim), dtype, init=hk.initializers.RandomNormal(stddev=0.05))
    w *= jax.lax.rsqrt(jnp.sum(w**2, axis=1))[:,None]

    def g_init(shape, dtype):
      t = jnp.dot(x, w.T)
      return 1/(jnp.std(t, axis=0) + 1e-5)

    def b_init(shape, dtype):
      t = jnp.dot(x, w.T)
      return -jnp.mean(t, axis=0)/(jnp.std(t, axis=0) + 1e-5)

    g = hk.get_parameter(f"g_{name_suffix}", (out_dim,), dtype, init=g_init)
    b = hk.get_parameter(f"b_{name_suffix}", (out_dim,), dtype, init=b_init)

    w *= g[:,None]

  else:
    w = hk.get_parameter(f"w_{name_suffix}", (out_dim, in_dim), init=w_init)
    b = hk.get_parameter(f"b_{name_suffix}", (out_dim,), init=b_init)

  return w, b

################################################################################################################

class MLP(hk.Module):

  def __init__(self,
               out_dim: Sequence[int],
               layer_sizes: Sequence[int]=[128]*4,
               nonlinearity: str="relu",
               parameter_norm: str=None,
               w_init: Callable=None,
               b_init: Callable=None,
               name: str=None):
    super().__init__(name=name)
    self.out_dim        = out_dim
    self.layer_sizes    = layer_sizes + [self.out_dim]
    self.parameter_norm = parameter_norm

    if nonlinearity == "relu":
      self.nonlinearity = jax.nn.relu
    elif nonlinearity == "tanh":
      self.nonlinearity = jnp.tanh
    elif nonlinearity == "sigmoid":
      self.nonlinearity = jax.nn.sigmoid
    elif nonlinearity == "swish":
      self.nonlinearity = jax.nn.swish(x)
    elif nonlinearity == "lipswish":
      self.nonlinearity = lambda x: jax.nn.swish(x)/1.1
    else:
      assert 0, "Invalid nonlinearity"

    self.w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal") if w_init is None else w_init
    self.b_init = jnp.zeros if b_init is None else b_init

  def __call__(self, x, is_training=True, **kwargs):
    # This function assumes that the input is batched!
    batch_size, in_dim = x.shape

    for i, out_dim in enumerate(self.layer_sizes):

      w, b = data_dependent_param_init(x,
                                       out_dim,
                                       name_suffix=f"{i}",
                                       w_init=self.w_init,
                                       b_init=self.b_init,
                                       is_training=is_training,
                                       parameter_norm=self.parameter_norm)
      x = jnp.dot(x, w.T) + b

      if i < len(self.layer_sizes) - 1:
        x = self.nonlinearity(x)

    return x
