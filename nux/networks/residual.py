import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
import haiku as hk
import nux.spectral_norm as sn
from nux.networks.cnn import ConvBlock
from typing import Optional, Mapping, Callable, Sequence, Any

__all__ = ["ResNet"]

class ResNet(hk.Module):

  def __init__(self,
               n_blocks: int,
               hidden_channel: Sequence[int],
               parameter_norm: str=None,
               nonlinearity: str="relu",
               w_init: Callable=None,
               b_init: Callable=None,
               use_bias: bool=True,
               name=None):
    super().__init__(name=name)

    self.n_blocks       = n_blocks
    self.hidden_channel = hidden_channel
    self.parameter_norm = parameter_norm
    self.nonlinearity   = nonlinearity
    self.w_init         = w_init
    self.b_init         = b_init
    self.use_bias       = use_bias

  def __call__(self, x, **kwargs):
    channel = x.shape[-1]
    for i in range(self.n_blocks):
      conv = ConvBlock(channel,
                       self.hidden_channel,
                       self.parameter_norm,
                       self.nonlinearity,
                       self.w_init,
                       self.b_init,
                       use_bias=False)
      x += conv(x)

    return x
