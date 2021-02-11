import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
from nux.internal.layer import Layer
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence, Any

__all__ = ["SqueezeExcitation"]

class SqueezeExcitation(Layer):
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

  def call(self,
           inputs,
           rng=None,
           is_training=True,
           update_params=True,
           **kwargs):
    x = inputs["x"]
    H, W, C = x.shape[-3:]
    c = C//self.reduce_ratio

    w1 = hk.get_parameter("w1", (c, C), x.dtype, init=self.w_init)
    w2 = hk.get_parameter("w2", (C, c), x.dtype, init=self.w_init)

    # Apply the SE transforms
    z = jnp.mean(x, axis=(-2, -3))
    z = jnp.dot(z, w1.T)
    z = jax.nn.relu(z)
    z = jnp.dot(z, w2.T)
    z = jax.nn.sigmoid(z)

    # Scale the input
    return {"x": x*z[...,None,None,:]}
