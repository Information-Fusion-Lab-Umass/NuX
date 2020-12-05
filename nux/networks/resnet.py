import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
import haiku as hk
from nux.networks.cnn import Conv, BottleneckConv, ReverseBottleneckConv
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
               zero_init: bool=False,
               dropout_rate: Optional[float]=0.2,
               gate: bool=True,
               name=None):
    super().__init__(name=name)

    self.conv_block_kwargs = dict(hidden_channel=hidden_channel,
                                  parameter_norm=parameter_norm,
                                  normalization=None,
                                  nonlinearity=nonlinearity,
                                  dropout_rate=dropout_rate,
                                  gate=gate)

    self.n_blocks       = n_blocks
    self.out_channel    = out_channel
    self.squeeze_excite = squeeze_excite
    self.zero_init      = zero_init

    if block_type == "bottleneck":
      self.conv_block = BottleneckConv
    elif block_type == "reverse_bottleneck":
      self.conv_block = ReverseBottleneckConv
    else:
      assert 0, "Invalid block type"

    # Move the normalization to outside of the repeated convolutions
    if normalization == "batch_norm":
      self.norm = lambda name: hk.BatchNorm(name=name, create_scale=True, create_offset=True, decay_rate=0.9, data_format="channels_last")

    elif normalization == "instance_norm":
      def norm(name):
        instance_norm = hk.InstanceNorm(name=name, create_scale=True, create_offset=True)
        def norm_apply(x, **kwargs): # So that this code works with the is_training kwarg
          return instance_norm(x)
        return norm_apply
      self.norm = norm

    elif normalization == "layer_norm":
      def norm(name):
        layer_norm = hk.LayerNorm(axis=-1, name=name, create_scale=True, create_offset=True)
        def norm_apply(x, **kwargs): # So that this code works with the is_training kwarg
          return layer_norm(x)
        return norm_apply
      self.norm = norm

    else:
      self.norm = None

  def __call__(self, x, rng, is_training=True, **kwargs):
    channel = x.shape[-1]

    rngs = jit(random.split, static_argnums=(1,))(rng, 3*self.n_blocks).reshape((self.n_blocks, 3, -1))

    for i, rng_for_convs in enumerate(rngs):
      z = self.conv_block(out_channel=channel,
                          **self.conv_block_kwargs)(x, rng_for_convs, is_training=is_training)

      if self.squeeze_excite:
        z = nux.SqueezeExcitation(reduce_ratio=4)(z)

      x += z

      if self.norm is not None:
        x = self.norm(f"norm_{i}")(x, is_training=is_training)

    # Add an extra convolution to change the out channels
    conv = Conv(self.out_channel,
                kernel_shape=(1, 1),
                stride=(1, 1),
                padding="SAME",
                parameter_norm=self.conv_block_kwargs["parameter_norm"],
                use_bias=False,
                zero_init=self.zero_init)
    x = conv(x, is_training=is_training)

    return x
