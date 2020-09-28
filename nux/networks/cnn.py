import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
import haiku as hk
import nux.spectral_norm as sn
from typing import Optional, Mapping, Callable, Sequence, Any

__all__ = ["Conv",
           "ConvBlock"]

################################################################################################################

def apply_conv(x: jnp.ndarray,
               w: jnp.ndarray,
               stride: Sequence[int],
               padding: Sequence[int],
               lhs_dilation: Sequence[int],
               rhs_dilation: Sequence[int],
               dimension_numbers: Sequence[str],
               transpose: bool):

  if transpose == False:
    return jax.lax.conv_general_dilated(x[None],
                                        w,
                                        window_strides=stride,
                                        padding=padding,
                                        lhs_dilation=lhs_dilation,
                                        rhs_dilation=rhs_dilation,
                                        dimension_numbers=dimension_numbers)[0]

  return jax.lax.conv_transpose(x[None],
                                w,
                                strides=stride,
                                padding=padding,
                                rhs_dilation=rhs_dilation,
                                dimension_numbers=dimension_numbers,
                                transpose_kernel=True)[0]

################################################################################################################

class Conv(hk.Module):

  def __init__(self,
               out_channel: int,
               kernel_shape: Sequence[int],
               parameter_norm: str=None,
               stride: Optional[Sequence[int]]=(1, 1),
               padding: str="SAME",
               lhs_dilation: Sequence[int]=(1, 1),
               rhs_dilation: Sequence[int]=(1, 1),
               w_init: Callable=None,
               b_init: Callable=None,
               use_bias: bool=True,
               transpose: bool=False,
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

    self.lhs_dilation      = lhs_dilation
    self.rhs_dilation      = rhs_dilation
    self.dimension_numbers = ('NHWC', 'HWIO', 'NHWC')

    self.transpose = transpose

  def __call__(self, x, **kwargs):
    H, W, C = x.shape
    in_channel = C
    w_shape = self.kernel_shape + (in_channel, self.out_channel)

    if self.parameter_norm == "spectral":
      w = hk.get_parameter("w", w_shape, x.dtype, init=self.w_init)

      u = hk.get_state(f"u_w", kernel_shape + (self.out_channel,), init=hk.initializers.RandomNormal())
      w, u = sn.spectral_norm_conv_apply(w, u, self.stride, self.padding, 0.9, 1)
      hk.set_state(f"u_w", u)

      if self.use_bias:
        b = hk.get_parameter("b", (self.out_channel,), x.dtype, init=self.b_init)

    elif self.parameter_norm == "weight_norm":
      def dd_w_init(shape, dtype):
        # Initializing g doesn't work for some reason, so move the initialization here
        w = self.w_init(shape, dtype)
        w_for_init = w*jax.lax.rsqrt((w**2).sum(axis=(0, 1, 2)))

        out = apply_conv(x,
                         w_for_init,
                         stride=self.stride,
                         padding=self.padding,
                         lhs_dilation=self.lhs_dilation,
                         rhs_dilation=self.rhs_dilation,
                         dimension_numbers=self.dimension_numbers,
                         transpose=self.transpose)

        std = jnp.std(out, axis=(0, 1))
        return w/std

      w = hk.get_parameter("w", w_shape, x.dtype, init=self.w_init)
      w *= jax.lax.rsqrt((w**2).sum(axis=(0, 1, 2)))

      g = hk.get_parameter("g", (self.out_channel,), x.dtype, init=jnp.ones)
      w *= g

      def dd_b_init(shape, dtype):

        out = apply_conv(x,
                         w,
                         stride=self.stride,
                         padding=self.padding,
                         lhs_dilation=self.lhs_dilation,
                         rhs_dilation=self.rhs_dilation,
                         dimension_numbers=self.dimension_numbers,
                         transpose=self.transpose)
        del x
        mean, std = jnp.mean(out, axis=(0, 1)), jnp.std(out, axis=(0, 1))
        return -mean/std

      if self.use_bias:
        b = hk.get_parameter("b", (self.out_channel,), x.dtype, init=jnp.zeros)
        # b = hk.get_parameter("b", (self.out_channel,), x.dtype, init=dd_b_init)

    else:
      w = hk.get_parameter("w", w_shape, x.dtype, init=self.w_init)
      if self.use_bias:
        b = hk.get_parameter("b", (self.out_channel,), x.dtype, init=self.b_init)

    out = apply_conv(x,
                     w,
                     stride=self.stride,
                     padding=self.padding,
                     lhs_dilation=self.lhs_dilation,
                     rhs_dilation=self.rhs_dilation,
                     dimension_numbers=self.dimension_numbers,
                     transpose=self.transpose)

    if self.use_bias:
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
               zero_init: bool=False,
               name=None):
    super().__init__(name=name)
    self.out_channel    = out_channel
    self.hidden_channel = hidden_channel
    self.w_init         = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal')
    self.b_init         = jnp.zeros if b_init is None else b_init
    self.use_bias       = use_bias
    self.norm           = norm
    self.zero_init      = zero_init

    self.parameter_norm = parameter_norm

    if nonlinearity == "relu":
      self.nonlinearity = jax.nn.relu
    elif nonlinearity == "swish":
      self.nonlinearity = jax.nn.swish
    elif nonlinearity == "lipswish":
      self.nonlinearity = lambda x: jax.nn.swish(x)/1.1
    else:
      assert 0

    if self.norm == "batch_norm":
      self.bn0 = hk.BatchNorm(name="bn_0", create_scale=True, create_offset=True, decay_rate=0.9, data_format="channels_last")
      self.bn1 = hk.BatchNorm(name="bn_1", create_scale=True, create_offset=True, decay_rate=0.9, data_format="channels_last")
      self.bn2 = hk.BatchNorm(name="bn_2", create_scale=True, create_offset=True, decay_rate=0.9, data_format="channels_last")

    elif self.norm == "instance_norm":
      self.in0 = hk.InstanceNorm(name="in_0", create_scale=True, create_offset=True)
      self.in1 = hk.InstanceNorm(name="in_1", create_scale=True, create_offset=True)
      self.in2 = hk.InstanceNorm(name="in_2", create_scale=True, create_offset=True)

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

    # We can initialize a resnet to be the identity if the last weight and bias is 0
    last_w_init = hk.initializers.RandomNormal(stddev=0.01) if self.zero_init else self.w_init
    last_b_init = hk.initializers.RandomNormal(stddev=0.01) if self.zero_init else self.b_init

    self.conv2 = Conv(self.out_channel,
                      kernel_shape=(3, 3),
                      parameter_norm=self.parameter_norm,
                      stride=(1, 1),
                      padding="SAME",
                      w_init=last_w_init,
                      b_init=last_b_init,
                      use_bias=self.use_bias)

  def __call__(self, x, is_training=True, **kwargs):
    H, W, C = x.shape

    if self.norm == "batch_norm":
      x = self.bn0(x, is_training=is_training)
    elif self.norm == "instance_norm":
      x = self.in0(x)


    x = self.nonlinearity(x)
    x = self.conv0(x)

    if self.norm == "batch_norm":
      x = self.bn1(x, is_training=is_training)
    elif self.norm == "instance_norm":
      x = self.in1(x)


    x = self.nonlinearity(x)
    x = self.conv1(x)

    if self.norm == "batch_norm":
      x = self.bn2(x, is_training=is_training)
    elif self.norm == "instance_norm":
      x = self.in2(x)


    x = self.nonlinearity(x)
    x = self.conv2(x)

    return x

################################################################################################################