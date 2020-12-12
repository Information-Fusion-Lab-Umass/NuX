import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence, Any
import nux.util.weight_initializers as init

__all__ = ["MLP"]

def data_dependent_param_init(x: jnp.ndarray,
                              out_dim: int,
                              name_suffix: str="",
                              w_init: Callable=None,
                              b_init: Callable=None,
                              is_training: bool=True,
                              parameter_norm: str=None,
                              use_bias: bool=True,
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

  elif parameter_norm == "weight_norm" and x.shape[0] > 1:

    return init.weight_with_weight_norm(x=x,
                                        out_dim=out_dim,
                                        name_suffix=name_suffix,
                                        w_init=w_init,
                                        b_init=b_init,
                                        is_training=is_training,
                                        use_bias=use_bias,
                                        **kwargs)

  # elif parameter_norm is not None:
  #   assert 0, "Invalid weight choice.  Expected 'spectral_norm' or 'weight_norm'"

  in_dim, dtype = x.shape[-1], x.dtype

  w = hk.get_parameter(f"w_{name_suffix}", (out_dim, in_dim), init=w_init)
  if use_bias:
    b = hk.get_parameter(f"b_{name_suffix}", (out_dim,), init=b_init)

  if use_bias:
    return w, b
  return w

################################################################################################################

class MLP(hk.Module):

  def __init__(self,
               out_dim: Sequence[int],
               layer_sizes: Sequence[int]=[128]*4,
               nonlinearity: str="relu",
               dropout_rate: Optional[float]=None,
               parameter_norm: str=None,
               w_init: Callable=None,
               b_init: Callable=None,
               zero_init: bool=False,
               skip_connection: bool=False,
               name: str=None):
    super().__init__(name=name)
    self.out_dim         = out_dim
    self.layer_sizes     = layer_sizes + [self.out_dim]
    self.parameter_norm  = parameter_norm
    self.zero_init       = zero_init
    self.dropout_rate    = dropout_rate
    self.skip_connection = skip_connection

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

    self.w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal") if w_init is None else w_init
    self.b_init = jnp.zeros if b_init is None else b_init

  def __call__(self, x, rng, is_training=True, update_params=True, **kwargs):
    # This function assumes that the input is batched!
    batch_size, in_dim = x.shape

    rngs = random.split(rng, len(self.layer_sizes))

    for i, (rng, out_dim) in enumerate(zip(rngs, self.layer_sizes)):
      if self.zero_init and i == len(self.layer_sizes) - 1:
        w, b = data_dependent_param_init(x,
                                         out_dim,
                                         name_suffix=f"{i}",
                                         w_init=hk.initializers.RandomNormal(stddev=0.01),
                                         b_init=jnp.zeros,
                                         is_training=is_training,
                                         update_params=update_params,
                                         parameter_norm=None)
      else:
        w, b = data_dependent_param_init(x,
                                         out_dim,
                                         name_suffix=f"{i}",
                                         w_init=self.w_init,
                                         b_init=self.b_init,
                                         is_training=is_training,
                                         update_params=update_params,
                                         parameter_norm=self.parameter_norm)
      z = jnp.dot(x, w.T) + b

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

    return x
