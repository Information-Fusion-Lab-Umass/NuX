import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
import haiku as hk
import nux.spectral_norm as sn
from typing import Optional, Mapping, Callable, Sequence, Any

__all__ = ["SqueezeExcitation"]

class SqueezeExcitation(hk.Module):
  """
  https://arxiv.org/pdf/1709.01507.pdf
  """
  def __init__(self,
               reduce_ratio: int=2,
               w_init: Callable=None,
               name=None):
    super().__init__(name=name)
    self.reduce_ratio = reduce_ratio
    self.w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal") if w_init is None else w_init

  def __call__(self, x, **kwargs):
    H, W, C = x.shape
    c = C//self.reduce_ratio

    w1 = hk.get_parameter("w1", (c, C), x.dtype, init=self.w_init)
    w2 = hk.get_parameter("w2", (C, c), x.dtype, init=self.w_init)

    # Apply the SE transforms
    z = np.mean(x, axis=(-2, -3))
    z = w1@z
    z = jax.nn.relu(z)
    z = w2@z
    z = jax.nn.sigmoid(z)

    # Scale the input
    return x*z[None,None,:]
