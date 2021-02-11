import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
from nux.internal.layer import Layer
import haiku as hk
from nux.networks.cnn import Conv, BottleneckConv, ReverseBottleneckConv
from nux.networks.se import SqueezeExcitation
from typing import Optional, Mapping, Callable, Sequence, Any
from nux.flows.compose import repeat

__all__ = ["ResNet"]

class ResNetStep(Layer):

  def __init__(self,
               hidden_channel: int,
               out_channel: int,
               parameter_norm: str=None,
               normalization: str=None,
               nonlinearity: str="relu",
               squeeze_excite: bool=False,
               block_type: str="reverse_bottleneck",
               zero_init: bool=False,
               dropout_rate: Optional[float]=0.2,
               gate: bool=False,
               gate_final: bool=True,
               use_bias: bool=True,
               max_singular_value: float=0.999,
               max_power_iters: int=1,
               name=None):
    super().__init__(name=name)

    self.conv_block_kwargs = dict(hidden_channel=hidden_channel,
                                  parameter_norm=parameter_norm,
                                  normalization=None,
                                  nonlinearity=nonlinearity,
                                  dropout_rate=dropout_rate,
                                  gate=gate,
                                  use_bias=use_bias,
                                  activate_last=True,
                                  max_singular_value=max_singular_value,
                                  max_power_iters=max_power_iters)
    self.hidden_channel = hidden_channel
    self.parameter_norm = parameter_norm
    self.normalization  = normalization
    self.nonlinearity   = nonlinearity
    self.dropout_rate   = dropout_rate

    self.out_channel    = out_channel
    self.squeeze_excite = squeeze_excite
    self.zero_init      = zero_init
    self.use_bias       = use_bias
    self.gate           = gate
    self.gate_final     = gate_final

    self.max_singular_value = max_singular_value
    self.max_power_iters    = max_power_iters

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

  def call(self,
           inputs,
           rng,
           is_training=True,
           update_params=True,
           **kwargs):
    x = inputs["x"]
    aux = inputs.get("aux", None)

    channel = x.shape[-1]
    z = self.conv_block(out_channel=channel,
                        **self.conv_block_kwargs)({"x": x, "aux": aux},
                                                  rng,
                                                  is_training=is_training)["x"]

    if self.squeeze_excite:
      z = SqueezeExcitation(reduce_ratio=4)(z)["x"]

    x += z

    if self.norm is not None:
      norm = self.auto_batch(self.norm(None), expected_depth=1)
      # norm = self.auto_batch(self.norm(f"norm_{i}"), expected_depth=1)
      x = norm(x, is_training=is_training)

    return {"x": x}

class ResNet(Layer):

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
               gate: bool=False,
               gate_final: bool=True,
               use_bias: bool=True,
               max_singular_value: float=0.95,
               max_power_iters: int=1,
               name=None):
    super().__init__(name=name)

    self.step_kwargs = dict(hidden_channel=hidden_channel,
                            out_channel=out_channel,
                            parameter_norm=parameter_norm,
                            normalization=normalization,
                            nonlinearity=nonlinearity,
                            squeeze_excite=squeeze_excite,
                            block_type=block_type,
                            zero_init=zero_init,
                            dropout_rate=dropout_rate,
                            gate=gate,
                            gate_final=gate_final,
                            use_bias=use_bias,
                            max_singular_value=max_singular_value,
                            max_power_iters=max_power_iters)

    self.hidden_channel = hidden_channel
    self.parameter_norm = parameter_norm
    self.normalization  = normalization
    self.nonlinearity   = nonlinearity
    self.dropout_rate   = dropout_rate

    self.n_blocks       = n_blocks
    self.out_channel    = out_channel
    self.squeeze_excite = squeeze_excite
    self.zero_init      = zero_init
    self.use_bias       = use_bias
    self.gate           = gate
    self.gate_final     = gate_final

    self.max_singular_value = max_singular_value
    self.max_power_iters    = max_power_iters

  def call(self,
           inputs,
           rng,
           is_training=True,
           update_params=True,
           **kwargs):
    x = inputs["x"]
    aux = inputs.get("aux", None)

    # Efficiently apply the residual steps
    self.step = partial(ResNetStep, **self.step_kwargs)
    self.res_flow_steps = repeat(self.step, n_repeats=self.n_blocks)
    x = self.res_flow_steps(inputs,
                            rng,
                            is_training=is_training,
                            update_params=update_params,
                            **kwargs)["x"]

    # Add an extra convolution to change the out channels
    if self.gate_final:
      conv = Conv(2*self.out_channel,
                  kernel_shape=(1, 1),
                  stride=(1, 1),
                  padding="SAME",
                  parameter_norm=self.parameter_norm,
                  use_bias=self.use_bias,
                  zero_init=self.zero_init)
      ab = conv({"x": x}, is_training=is_training)["x"]
      a, b = jnp.split(ab, 2, axis=-1)
      x = a*jax.nn.sigmoid(b)
    else:
      conv = Conv(self.out_channel,
                  kernel_shape=(1, 1),
                  stride=(1, 1),
                  padding="SAME",
                  parameter_norm=self.parameter_norm,
                  use_bias=self.use_bias,
                  zero_init=self.zero_init)
      x = conv({"x": x}, is_training=is_training)["x"]

    return {"x": x}
