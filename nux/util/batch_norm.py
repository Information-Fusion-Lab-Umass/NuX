import types
from typing import Optional, Sequence
import jax
import jax.numpy as jnp
import haiku as hk


class BatchNorm(hk.BatchNorm):

  def __init__(
      self,
      create_scale: bool,
      create_offset: bool,
      decay_rate: float,
      eps: float=1e-5,
      mean_only: bool=False,
      scale_init: Optional[hk.initializers.Initializer]=None,
      offset_init: Optional[hk.initializers.Initializer]=None,
      axis: Optional[Sequence[int]]=None,
      cross_replica_axis: Optional[str]=None,
      cross_replica_axis_index_groups: Optional[Sequence[Sequence[int]]]=None,
      data_format: str="channels_last",
      name: Optional[str]=None,
  ):
    """
    Same as hk.BatchNorm but has extra option to only compute the batch mean
    """
    if mean_only:
      create_scale = False
    super().__init__(create_scale=create_scale,
                     create_offset=create_offset,
                     decay_rate=decay_rate,
                     eps=eps,
                     scale_init=scale_init,
                     offset_init=offset_init,
                     axis=axis,
                     cross_replica_axis=cross_replica_axis,
                     cross_replica_axis_index_groups=cross_replica_axis_index_groups,
                     data_format=data_format,
                     name=name)
    self.mean_only = mean_only


  def __call__(
      self,
      inputs: jnp.ndarray,
      is_training: bool,
      test_local_stats: bool=False,
      scale: Optional[jnp.ndarray]=None,
      offset: Optional[jnp.ndarray]=None,
      return_lipschitz_const: bool=False,
  ) -> jnp.ndarray:
    """Computes the normalized version of the input.
    Args:
      inputs: An array, where the data format is ``[..., C]``.
      is_training: Whether this is during training.
      test_local_stats: Whether local stats are used when is_training=False.
      scale: An array up to n-D. The shape of this tensor must be broadcastable
        to the shape of ``inputs``. This is the scale applied to the normalized
        inputs. This cannot be passed in if the module was constructed with
        ``create_scale=True``.
      offset: An array up to n-D. The shape of this tensor must be broadcastable
        to the shape of ``inputs``. This is the offset applied to the normalized
        inputs. This cannot be passed in if the module was constructed with
        ``create_offset=True``.
    Returns:
      The array, normalized across all but the last dimension.
    """
    if self.create_scale and scale is not None:
      raise ValueError(
          "Cannot pass `scale` at call time if `create_scale=True`.")
    if self.create_offset and offset is not None:
      raise ValueError(
          "Cannot pass `offset` at call time if `create_offset=True`.")

    channel_index = self.channel_index
    if channel_index < 0:
      channel_index += inputs.ndim

    if self.axis is not None:
      axis = self.axis
    else:
      axis = [i for i in range(inputs.ndim) if i != channel_index]

    if is_training or test_local_stats:
      mean = jnp.mean(inputs, axis, keepdims=True)
      if self.mean_only == False:
        mean_of_squares = jnp.mean(inputs**2, axis, keepdims=True)
      if self.cross_replica_axis:
        mean = jax.lax.pmean(
            mean,
            axis_name=self.cross_replica_axis,
            axis_index_groups=self.cross_replica_axis_index_groups)
        if self.mean_only == False:
          mean_of_squares = jax.lax.pmean(
              mean_of_squares,
              axis_name=self.cross_replica_axis,
              axis_index_groups=self.cross_replica_axis_index_groups)

      if self.mean_only == False:
        var = mean_of_squares - mean ** 2
    else:
      mean = self.mean_ema.average
      if self.mean_only == False:
        var = self.var_ema.average

    if is_training:
      self.mean_ema(mean)
      if self.mean_only == False:
        self.var_ema(var)

    w_shape = [1 if i in axis else inputs.shape[i] for i in range(inputs.ndim)]
    w_dtype = inputs.dtype

    if self.mean_only == False:
      if self.create_scale:
        scale = hk.get_parameter("scale", w_shape, w_dtype, self.scale_init)
      elif scale is None:
        scale = np.ones([], dtype=w_dtype)

    if self.create_offset:
      offset = hk.get_parameter("offset", w_shape, w_dtype, self.offset_init)
    elif offset is None:
      offset = np.zeros([], dtype=w_dtype)

    if self.mean_only == False:
      eps = jax.lax.convert_element_type(self.eps, var.dtype)
      inv = scale * jax.lax.rsqrt(var + eps)
      ret = (inputs - mean) * inv + offset
      lip = jnp.max(inv)
    else:
      ret = inputs - mean + offset
      lip = 1.0

    if return_lipschitz_const:
      return ret, lip

    return ret

