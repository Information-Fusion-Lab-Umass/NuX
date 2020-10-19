import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
import haiku as hk
import nux.spectral_norm as sn
from typing import Optional, Mapping, Callable, Sequence, Any

__all__ = ["Conv",
           "ConvBlock",
           "BottleneckConv",
           "ReverseBottleneckConv"]

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
    return jax.lax.conv_general_dilated(x,
                                        w,
                                        window_strides=stride,
                                        padding=padding,
                                        lhs_dilation=lhs_dilation,
                                        rhs_dilation=rhs_dilation,
                                        dimension_numbers=dimension_numbers)

  return jax.lax.conv_transpose(x,
                                w,
                                strides=stride,
                                padding=padding,
                                rhs_dilation=rhs_dilation,
                                dimension_numbers=dimension_numbers,
                                transpose_kernel=True)

################################################################################################################

def data_dependent_param_init(x: jnp.ndarray,
                              kernel_shape: Sequence[int],
                              out_channel: int,
                              name_suffix: str="",
                              w_init: Callable=None,
                              b_init: Callable=None,
                              parameter_norm: str=None,
                              use_bias: bool=True,
                              is_training: bool=True,
                              **conv_kwargs):
  batch_size, H, W, C = x.shape
  w_shape = kernel_shape + (C, out_channel)

  if parameter_norm == "spectral_norm":
    w = hk.get_parameter(f"w_{name_suffix}", w_shape, x.dtype, init=w_init)
    if use_bias:
      b = hk.get_parameter(f"b_{name_suffix}", (out_channel,), init=b_init)

    u = hk.get_state(f"u_{name_suffix}", kernel_shape + (out_channel,), init=hk.initializers.RandomNormal())
    w, u = sn.spectral_norm_conv_apply(w, u, conv_kwargs["stride"], conv_kwargs["padding"], 0.9, 1)
    if is_training == True:
      hk.set_state(f"u_{name_suffix}", u)

    if use_bias:
      b = hk.get_parameter(f"b_{name_suffix}", (out_channel,), x.dtype, init=b_init)

  elif parameter_norm == "weight_norm" and x.shape[0] > 1:
    w = hk.get_parameter("w", w_shape, x.dtype, init=hk.initializers.RandomNormal(stddev=0.05))
    w *= jax.lax.rsqrt((w**2).sum(axis=(0, 1, 2)))[None,None,None,:]

    def g_init(shape, dtype):
      t = apply_conv(x, w, **conv_kwargs)
      return 1/(jnp.std(t, axis=(0, 1, 2)) + 1e-5)

    def b_init(shape, dtype):
      t = apply_conv(x, w, **conv_kwargs)
      return -jnp.mean(t, axis=(0, 1, 2))/(jnp.std(t, axis=(0, 1, 2)) + 1e-5)

    g = hk.get_parameter(f"g_{name_suffix}", (out_channel,), x.dtype, init=g_init)
    if use_bias:
      b = hk.get_parameter(f"b_{name_suffix}", (out_channel,), x.dtype, init=b_init)

    w *= g[None,None,None,:]

  else:
    w = hk.get_parameter(f"w_{name_suffix}", w_shape, x.dtype, init=w_init)
    if use_bias:
      b = hk.get_parameter(f"b_{name_suffix}", (out_channel,), x.dtype, init=b_init)

  if use_bias:
    return w, b
  return w

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

    self.conv_kwargs = dict(stride=self.stride,
                            padding=self.padding,
                            lhs_dilation=self.lhs_dilation,
                            rhs_dilation=self.rhs_dilation,
                            dimension_numbers=self.dimension_numbers,
                            transpose=self.transpose)

  def __call__(self, x, is_training=True, **kwargs):
    # This function assumes that the input is batched!
    batch_size, H, W, C = x.shape

    params = data_dependent_param_init(x,
                                       self.kernel_shape,
                                       self.out_channel,
                                       name_suffix="",
                                       w_init=self.w_init,
                                       b_init=self.b_init,
                                       parameter_norm=self.parameter_norm,
                                       use_bias=self.use_bias,
                                       is_training=is_training,
                                       **self.conv_kwargs)
    if self.use_bias:
      w, b = params
    else:
      w = params

    out = apply_conv(x, w, **self.conv_kwargs)

    if self.use_bias:
      out += b

    return out

################################################################################################################

class RepeatedConv(hk.Module):

  def __init__(self,
               channel_sizes: Sequence[int],
               kernel_shapes: Sequence[Sequence[int]],
               parameter_norm: str=None,
               stride: Optional[Sequence[int]]=(1, 1),
               padding: str="SAME",
               lhs_dilation: Sequence[int]=(1, 1),
               rhs_dilation: Sequence[int]=(1, 1),
               w_init: Callable=None,
               b_init: Callable=None,
               use_bias: bool=False,
               normalization: str=None,
               nonlinearity: str="relu",
               name=None):
    super().__init__(name=name)

    assert len(channel_sizes) == len(kernel_shapes)
    self.channel_sizes = channel_sizes
    self.kernel_shapes = kernel_shapes

    self.conv_kwargs = dict(parameter_norm=parameter_norm,
                            stride=stride,
                            padding=padding,
                            lhs_dilation=lhs_dilation,
                            rhs_dilation=rhs_dilation,
                            w_init=w_init,
                            b_init=b_init,
                            use_bias=use_bias,
                            transpose=False)

    if nonlinearity == "relu":
      self.nonlinearity = jax.nn.relu
    elif nonlinearity == "tanh":
      self.nonlinearity = jnp.tanh
    elif nonlinearity == "sigmoid":
      self.nonlinearity = jax.nn.sigmoid
    elif nonlinearity == "swish":
      self.nonlinearity = jax.nn.swish(x)
    elif nonlinearity == "lipswish":
      self.nonlinearity = lambda x: jax.nn.swish(x)/1.1
    else:
      assert 0, "Invalid nonlinearity"

    if normalization == "batch_norm":
      self.norm = lambda name: hk.BatchNorm(name=name, create_scale=True, create_offset=True, decay_rate=0.9, data_format="channels_last")

    elif normalization == "instance_norm":
      def norm(name):
        instance_norm = hk.InstanceNorm(name=name, create_scale=True, create_offset=True)
        def norm_apply(x, **kwargs): # So that this code works with the is_training kwarg
          return instance_norm(x)
        return norm_apply
      self.norm = norm

    else:
      self.norm = None

  def __call__(self, x, is_training=True, **kwargs):
    # This function assumes that the input is batched!
    batch_size, H, W, C = x.shape

    for i, (out_channel, kernel_shape) in enumerate(zip(self.channel_sizes, self.kernel_shapes)):

      x = Conv(out_channel, kernel_shape, **self.conv_kwargs)(x, is_training=is_training)

      if self.norm is not None:
        x = self.norm(f"norm_{i}")(x, is_training=is_training)

      if i < len(self.channel_sizes) - 1:
        x = self.nonlinearity(x)

    return x

################################################################################################################

class BottleneckConv(RepeatedConv):
  """ Use if we have a big input channel """
  def __init__(self,
               hidden_channel: int,
               out_channel: int,
               parameter_norm: str=None,
               normalization: str=None,
               nonlinearity: str="relu",
               name=None):

    channel_sizes = [hidden_channel, hidden_channel, out_channel]
    kernel_shapes = [(1, 1), (3, 3), (1, 1)]

    super().__init__(channel_sizes=channel_sizes,
                     kernel_shapes=kernel_shapes,
                     parameter_norm=parameter_norm,
                     normalization=normalization,
                     nonlinearity=nonlinearity,
                     use_bias=False,
                     name=name)

class ReverseBottleneckConv(RepeatedConv):
  """ Use if we have a small input channel """
  def __init__(self,
               hidden_channel: int,
               out_channel: int,
               parameter_norm: str=None,
               normalization: str=None,
               nonlinearity: str="relu",
               name=None):

    channel_sizes = [hidden_channel, hidden_channel, out_channel]
    kernel_shapes = [(3, 3), (1, 1), (3, 3)]

    super().__init__(channel_sizes=channel_sizes,
                     kernel_shapes=kernel_shapes,
                     parameter_norm=parameter_norm,
                     normalization=normalization,
                     nonlinearity=nonlinearity,
                     use_bias=False,
                     name=name)

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

    if nonlinearity == "relu":
      self.nonlinearity = jax.nn.relu
    elif nonlinearity == "tanh":
      self.nonlinearity = jax.nn.tanh
    elif nonlinearity == "sigmoid":
      self.nonlinearity = jax.nn.sigmoid
    elif nonlinearity == "swish":
      self.nonlinearity = jax.nn.swish(x)
    elif nonlinearity == "lipswish":
      self.nonlinearity = lambda x: jax.nn.swish(x)/1.1
    else:
      assert 0, "Invalid nonlinearity"

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
