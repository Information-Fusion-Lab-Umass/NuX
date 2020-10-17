import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
import haiku as hk
import nux.spectral_norm as sn
from nux.networks.cnn import Conv, ConvBlock
from nux.networks.se import SqueezeExcitation
from typing import Optional, Mapping, Callable, Sequence, Any

__all__ = ["ResNet"]

class ResNet(hk.Module):

  def __init__(self,
               n_blocks: int,
               hidden_channel: Sequence[int],
               out_channel: Optional[int]=None,
               parameter_norm: str=None,
               norm: str=None,
               nonlinearity: str="relu",
               w_init: Callable=None,
               b_init: Callable=None,
               identity_init: bool=True,
               squeeze_excite: bool=False,
               use_projection: bool=False,
               zero_last_conv: bool=False,
               name=None):
    super().__init__(name=name)

    self.n_blocks        = n_blocks
    self.hidden_channel  = hidden_channel
    self.parameter_norm  = parameter_norm
    self.norm            = norm
    self.nonlinearity    = nonlinearity
    self.w_init          = w_init
    self.b_init          = b_init
    self.out_channel     = out_channel
    self.identity_init   = identity_init
    self.squeeze_excite = squeeze_excite
    self.use_projection  = use_projection
    self.zero_last_conv  = zero_last_conv

    if(parameter_norm == "spectral_norm"):
      assert self.identity_init == False, "We will divide by 0 with parameter normalization and zero init!"

  def __call__(self, x, **kwargs):
    channel = x.shape[-1]
    for i in range(self.n_blocks):
      conv = ConvBlock(out_channel=channel,
                       hidden_channel=self.hidden_channel,
                       parameter_norm=self.parameter_norm,
                       norm=self.norm,
                       nonlinearity=self.nonlinearity,
                       w_init=self.w_init,
                       b_init=self.b_init,
                       use_bias=False,
                       zero_init=self.identity_init)
      z = conv(x)

      if self.squeeze_excite:
        z = SqueezeExcitation(reduce_ratio=8)(z)

      if self.use_projection:
        conv = Conv(channel,
                    kernel_shape=(1, 1),
                    parameter_norm=self.parameter_norm,
                    stride=(1, 1),
                    padding="SAME",
                    w_init=self.w_init,
                    b_init=self.b_init,
                    use_bias=False)
        x_proj = conv(x)

        x = x_proj + z

      else:
        x += z

    if self.out_channel is not None:
      # Add an extra convolution to change the out channels
      if self.zero_last_conv:
        w_init = hk.initializers.RandomNormal(stddev=0.1)
      else:
        w_init = self.w_init
      conv = Conv(self.out_channel,
                  kernel_shape=(1, 1),
                  parameter_norm=self.parameter_norm,
                  stride=(1, 1),
                  padding="SAME",
                  w_init=w_init,
                  b_init=self.b_init,
                  use_bias=False)
      x = conv(x)

    return x
