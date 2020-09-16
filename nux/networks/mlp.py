import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
import haiku as hk
import nux.spectral_norm as sn
from typing import Optional, Mapping, Callable, Sequence, Any

__all__ = ["MLP"]

def weight_norm(x):
    out = x*jax.lax.rsqrt(jnp.sum(x**2, axis=0))
    return out

def get_weight(name: str,
               in_dim: int,
               out_dim: int,
               dtype: Any,
               init: Callable,
               parameter_norm: str=None):
  w = hk.get_parameter(name, (out_dim, in_dim), init=init)

  if(parameter_norm == "spectral"):
    u = hk.get_state(f"u_{name}", (out_dim,), init=hk.initializers.RandomNormal())
    w, u = sn.spectral_norm_apply(w, u, 0.9, 1)
    hk.set_state(f"u_{name}", u)

  elif(parameter_norm == "spectral"):
    w = weight_norm(w)

  return w

################################################################################################################

class MLP(hk.Module):

  def __init__(self,
               out_dim: Sequence[int],
               layer_sizes: Sequence[int]=[1024]*4,
               nonlinearity: str="relu",
               parameter_norm: str=None,
               w_init: Callable=None,
               b_init: Callable=None,
               name: str=None):
    super().__init__(name=name)
    self.out_dim        = out_dim
    self.layer_sizes    = layer_sizes + [self.out_dim]
    self.parameter_norm = parameter_norm

    if(nonlinearity == "relu"):
      self.nonlinearity = jax.nn.relu
    elif(nonlinearity == "lipswish"):
      self.nonlinearity = lambda x: jax.nn.swish(x)/1.1

    self.w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal") if w_init is None else w_init
    self.b_init = jnp.zeros if b_init is None else b_init

  def __call__(self, x, **kwargs):

    for i, output_size in enumerate(self.layer_sizes):
      input_size = x.shape[-1]

      w = get_weight(f"w_{i}",
                     input_size,
                     output_size,
                     x.dtype,
                     self.w_init,
                     parameter_norm=self.parameter_norm)

      b = hk.get_parameter(f"b_{i}", [output_size], init=self.b_init)
      x = w@x + b

      if(i < len(self.layer_sizes) - 1):
        x = self.nonlinearity(x)

    return x
