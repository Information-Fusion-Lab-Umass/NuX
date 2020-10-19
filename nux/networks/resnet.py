import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
import haiku as hk
import nux.spectral_norm as sn
from nux.networks.cnn import Conv, ConvBlock, BottleneckConv, ReverseBottleneckConv
from nux.networks.se import SqueezeExcitation
from typing import Optional, Mapping, Callable, Sequence, Any

__all__ = ["ResNet"]

class ResNet(hk.Module):

  def __init__(self,
               n_blocks: int,
               hidden_channel: int,
               out_channel: int,
               parameter_norm: str=None,
               normalization: str=None,
               nonlinearity: str="relu",
               squeeze_excite: bool=False,
               block_type: str="reverse_bottleneck",
               name=None):
    super().__init__(name=name)

    self.conv_block_kwargs = dict(hidden_channel=hidden_channel,
                                  parameter_norm=parameter_norm,
                                  normalization=normalization,
                                  nonlinearity=nonlinearity)

    self.n_blocks       = n_blocks
    self.out_channel    = out_channel
    self.squeeze_excite = squeeze_excite

    if block_type == "bottleneck":
      self.conv_block = BottleneckConv
    elif block_type == "reverse_bottleneck":
      self.conv_block = ReverseBottleneckConv
    else:
      assert 0, "Invalid block type"

  def __call__(self, x, is_training=True, **kwargs):
    channel = x.shape[-1]
    for i in range(self.n_blocks):
      z = self.conv_block(out_channel=channel,
                          **self.conv_block_kwargs)(x, is_training=is_training)

      if self.squeeze_excite:
        z = SqueezeExcitation(reduce_ratio=4)(z)

      x += z

    # Add an extra convolution to change the out channels
    conv = Conv(self.out_channel,
                kernel_shape=(1, 1),
                stride=(1, 1),
                padding="SAME",
                parameter_norm=self.conv_block_kwargs["parameter_norm"],
                use_bias=False)
    x = conv(x, is_training=is_training)

    return x
