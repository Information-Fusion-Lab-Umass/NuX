import jax.numpy as jnp
from jax import jit, random
from functools import partial
from nux.internal.layer import Layer
import jax
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence, Any
import nux.util.weight_initializers as init
import nux.util as util

__all__ = ["MLP"]

def data_dependent_param_init(x: jnp.ndarray,
                              out_dim: int,
                              name_suffix: str="",
                              w_init: Callable=None,
                              b_init: Callable=None,
                              is_training: bool=True,
                              parameter_norm: str=None,
                              use_bias: bool=True,
                              update_params: bool=True,
                              **kwargs):

  if parameter_norm == "spectral_norm":
    return init.weight_with_spectral_norm(x=x,
                                          out_dim=out_dim,
                                          name_suffix=name_suffix,
                                          w_init=w_init,
                                          b_init=b_init,
                                          is_training=is_training,
                                          use_bias=use_bias,
                                          **kwargs)
  elif parameter_norm == "differentiable_spectral_norm":
    return init.weight_with_good_spectral_norm(x=x,
                                               out_dim=out_dim,
                                               name_suffix=name_suffix,
                                               w_init=w_init,
                                               b_init=b_init,
                                               is_training=is_training,
                                               update_params=update_params,
                                               use_bias=use_bias,
                                               **kwargs)

  elif parameter_norm == "weight_norm":
    if x.shape[0] > 1:
      return init.weight_with_weight_norm(x=x,
                                          out_dim=out_dim,
                                          name_suffix=name_suffix,
                                          w_init=w_init,
                                          b_init=b_init,
                                          is_training=is_training,
                                          use_bias=use_bias,
                                          **kwargs)

  elif parameter_norm is not None:
    assert 0, "Invalid weight choice.  Expected 'spectral_norm' or 'weight_norm'"

  in_dim, dtype = x.shape[-1], x.dtype

  w = hk.get_parameter(f"w_{name_suffix}", (out_dim, in_dim), init=w_init)
  if use_bias:
    b = hk.get_parameter(f"b_{name_suffix}", (out_dim,), init=b_init)

  if use_bias:
    return w, b
  return w

################################################################################################################

class MLP(Layer):

  def __init__(self,
               out_dim: Sequence[int],
               layer_sizes: Sequence[int]=[128]*4,
               nonlinearity: str="relu",
               dropout_rate: Optional[float]=None,
               parameter_norm: str=None,
               normalization: str=None,
               w_init: Callable=None,
               b_init: Callable=None,
               zero_init: bool=False,
               skip_connection: bool=False,
               max_singular_value: float=0.99,
               max_power_iters: int=1,
               name: str=None):
    super().__init__(name=name)
    self.out_dim         = out_dim
    self.layer_sizes     = layer_sizes + [self.out_dim]
    self.parameter_norm  = parameter_norm
    self.zero_init       = zero_init
    self.dropout_rate    = dropout_rate
    self.skip_connection = skip_connection

    self.max_singular_value = max_singular_value
    self.max_power_iters = max_power_iters

    if nonlinearity == "relu":
      self.nonlinearity = jax.nn.relu
    elif nonlinearity == "tanh":
      self.nonlinearity = jnp.tanh
    elif nonlinearity == "sigmoid":
      self.nonlinearity = jax.nn.sigmoid
    elif nonlinearity == "swish":
      self.nonlinearity = jax.nn.swish
    elif nonlinearity == "lipswish":
      self.nonlinearity = lambda x: jax.nn.swish(x)/1.1
    else:
      assert 0, "Invalid nonlinearity"

    if normalization == "batch_norm":
      self.norm = lambda name: hk.BatchNorm(name=name, create_scale=True, create_offset=True, decay_rate=0.9, data_format="channels_last")

    elif normalization == "mean_only_batch_norm":
      self.norm = lambda name: util.BatchNorm(name=name, mean_only=True, create_offset=True, decay_rate=0.9, data_format="channels_last")

    elif normalization == "instance_norm":
      def norm(name):
        instance_norm = hk.InstanceNorm(name=name, create_scale=True, create_offset=True)
        def norm_apply(x, **kwargs): # So that this code works with the is_training kwarg
          return instance_norm(x)
        return norm_apply
      self.norm = norm

    elif normalization == "layer_norm":
      def norm(name):
        instance_norm = hk.LayerNorm(axis=-1, name=name, create_scale=True, create_offset=True)
        def norm_apply(x, **kwargs): # So that this code works with the is_training kwarg
          return instance_norm(x)
        return norm_apply
      self.norm = norm
    else:
      self.norm = None
    self.w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal") if w_init is None else w_init
    self.b_init = jnp.zeros if b_init is None else b_init

  def call(self,
           inputs,
           rng,
           is_training=True,
           update_params=True,
           **kwargs):
    x = inputs["x"]
    aux = inputs.get("aux", None)

    assert len(self.unbatched_input_shapes["x"]) == 1

    # This function assumes that the input is batched!
    rngs = random.split(rng, len(self.layer_sizes))

    for i, (rng, out_dim) in enumerate(zip(rngs, self.layer_sizes)):
      # Pass a singly batched input to the parameter functions.
      # Don't use autobatching here because we might end up reducing
      x, reshape = self.make_singly_batched(x)

      if self.zero_init and i == len(self.layer_sizes) - 1:
        w, b = data_dependent_param_init(x,
                                         out_dim,
                                         name_suffix=f"{i}",
                                         w_init=hk.initializers.RandomNormal(stddev=0.01),
                                         b_init=jnp.zeros,
                                         is_training=is_training,
                                         update_params=update_params,
                                         max_singular_value=self.max_singular_value,
                                         max_power_iters=self.max_power_iters,
                                         parameter_norm=None)
      else:
        w, b = data_dependent_param_init(x,
                                         out_dim,
                                         name_suffix=f"{i}",
                                         w_init=self.w_init,
                                         b_init=self.b_init,
                                         is_training=is_training,
                                         update_params=update_params,
                                         max_singular_value=self.max_singular_value,
                                         max_power_iters=self.max_power_iters,
                                         parameter_norm=self.parameter_norm)
      x = reshape(x)

      z = jnp.einsum("...ij,...j->...i", w, x) + b

      if self.norm is not None:
        norm = self.auto_batch(self.norm(f"norm_{i}"), expected_depth=1)
        x = norm(x, is_training=is_training)

      if i < len(self.layer_sizes) - 1:
        z = self.nonlinearity(z)

      # Residual connection
      if self.skip_connection and x.shape[-1] == z.shape[-1]:
        x += z
      else:
        x = z

      if i < len(self.layer_sizes) - 1:
        if self.dropout_rate is not None:
          rate = self.dropout_rate if is_training else 0.0
          x = hk.dropout(rng, rate, x)

    outputs = {"x": x}
    return outputs