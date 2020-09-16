import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
import haiku as hk
import nux.spectral_norm as sn
from typing import Optional, Mapping, Callable, Sequence, Any

__all__ = ["Conv",
           "ConvBlock"]

def get_conv_weight(name: str,
                    in_channels: int,
                    out_channels: int,
                    kernel_shape: Sequence[int],
                    dtype: Any,
                    init: Callable,
                    parameter_norm: str=None,
                    stride: Optional[Sequence[int]]=(1, 1),
                    padding: str="SAME"):
  w_shape = kernel_shape + (in_channels, out_channels)
  w = hk.get_parameter(name, w_shape, dtype, init=init)

  if(parameter_norm == "spectral"):
    u = hk.get_state(f"u_{name}", kernel_shape + (out_channels,), init=hk.initializers.RandomNormal())
    w, u = sn.spectral_norm_conv_apply(w, u, stride, padding, 0.9, 1)
    hk.set_state(f"u_{name}", u)

  elif(parameter_norm == "spectral"):
    assert 0, "Not implemented"

  return w

################################################################################################################

class Conv(hk.Module):

  def __init__(self,
               out_channel: int,
               kernel_shape: Sequence[int],
               parameter_norm: str=None,
               stride: Optional[Sequence[int]]=(1, 1),
               padding: str="SAME",
               w_init: Callable=None,
               b_init: Callable=None,
               use_bias: bool=True,
               name=None):
    super().__init__(name=name)
    self.out_channel = out_channel

    self.parameter_norm = parameter_norm

    self.kernel_shape = kernel_shape
    self.padding      = padding
    self.stride       = stride

    self.w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal") if w_init is None else w_init
    self.b_init = jnp.zeros if b_init is None else b_init

    self.use_bias = use_bias

    self.lhs_dilation      = (1, 1)
    self.rhs_dilation      = (1, 1)
    self.dimension_numbers = ('NHWC', 'HWIO', 'NHWC')

  def __call__(self, x, **kwargs):
    H, W, C = x.shape

    w = get_conv_weight(f"w",
                        C,
                        self.out_channel,
                        self.kernel_shape,
                        x.dtype,
                        self.w_init,
                        parameter_norm=self.parameter_norm,
                        stride=self.stride,
                        padding=self.padding)

    out = jax.lax.conv_general_dilated(x[None],
                                       w,
                                       window_strides=self.stride,
                                       padding=self.padding,
                                       lhs_dilation=self.lhs_dilation,
                                       rhs_dilation=self.rhs_dilation,
                                       dimension_numbers=self.dimension_numbers)[0]

    if(self.use_bias):
      b = hk.get_parameter("b", (self.out_channel,), x.dtype, init=self.b_init)
      out += b

    return out

################################################################################################################

class ConvBlock(hk.Module):

  def __init__(self,
               out_channel: Sequence[int],
               hidden_channel: Sequence[int],
               parameter_norm: str=None,
               norm: str=None,
               nonlinearity: str="relu",
               w_init: Callable=None,
               b_init: Callable=None,
               use_bias: bool=True,
               name=None):
    super().__init__(name=name)
    self.out_channel    = out_channel
    self.hidden_channel = hidden_channel
    self.w_init         = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal')
    self.b_init         = jnp.zeros if b_init is None else b_init
    self.use_bias       = use_bias
    self.norm           = norm

    self.parameter_norm = parameter_norm

    if(nonlinearity == "relu"):
      self.nonlinearity = jax.nn.relu
    elif(nonlinearity == "lipswish"):
      self.nonlinearity = lambda x: jax.nn.swish(x)/1.1
    else:
      assert 0

    if(self.norm == "batch_norm"):
      self.bn0 = hk.BatchNorm(name="bn_0", create_scale=True, create_offset=True, decay_rate=0.9, data_format="channels_last")
      self.bn1 = hk.BatchNorm(name="bn_1", create_scale=True, create_offset=True, decay_rate=0.9, data_format="channels_last")
      self.bn2 = hk.BatchNorm(name="bn_2", create_scale=True, create_offset=True, decay_rate=0.9, data_format="channels_last")

    self.conv0 = Conv(self.hidden_channel,
                      kernel_shape=(3, 3),
                      parameter_norm=self.parameter_norm,
                      stride=(1, 1),
                      padding="SAME",
                      w_init=self.w_init,
                      b_init=self.b_init,
                      use_bias=self.use_bias)

    self.conv1 = Conv(self.hidden_channel,
                      kernel_shape=(1, 1),
                      parameter_norm=self.parameter_norm,
                      stride=(1, 1),
                      padding="SAME",
                      w_init=self.w_init,
                      b_init=self.b_init,
                      use_bias=self.use_bias)

    self.conv2 = Conv(self.out_channel,
                      kernel_shape=(3, 3),
                      parameter_norm=self.parameter_norm,
                      stride=(1, 1),
                      padding="SAME",
                      w_init=self.w_init,
                      b_init=self.b_init,
                      use_bias=self.use_bias)

  def __call__(self, x, is_training=True, **kwargs):
    H, W, C = x.shape

    if(self.norm == "batch_norm"):
      x = self.bn0(x, is_training=is_training)

    x = self.nonlinearity(x)
    x = self.conv0(x)

    if(self.norm == "batch_norm"):
      x = self.bn1(x, is_training=is_training)

    x = self.nonlinearity(x)
    x = self.conv1(x)

    if(self.norm == "batch_norm"):
      x = self.bn2(x, is_training=is_training)

    x = self.nonlinearity(x)
    x = self.conv2(x)

    return x

################################################################################################################
