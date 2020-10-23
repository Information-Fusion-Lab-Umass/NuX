import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Sequence
from nux.flows.base import *
import nux.util as util

__all__ = ["Squeeze",
           "UnSqueeze",
           "Flatten",
           "Reshape",
           "Reverse"]

class Squeeze(Layer):

  def __init__(self,
               filter_shape: Sequence[int]=(2, 2),
               dilation: Sequence[int]=(1, 1),
               name: str="squeeze",
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.filter_shape = filter_shape
    self.dilation     = dilation

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]

    @self.auto_batch
    def apply_fun(x):
      sq = util.dilated_squeeze if sample == False else util.dilated_unsqueeze
      return sq(x, self.filter_shape, self.dilation)

    z = apply_fun(x)
    outputs = {"x": z, "log_det": jnp.zeros(self.batch_shape)}

    return outputs

class UnSqueeze(Layer):

  def __init__(self,
               filter_shape: Sequence[int]=(2, 2),
               dilation: Sequence[int]=(1, 1),
               name: str="unsqueeze",
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.filter_shape = filter_shape
    self.dilation     = dilation

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]

    @self.auto_batch
    def apply_fun(x):
      sq = util.dilated_squeeze if sample == True else util.dilated_unsqueeze
      return sq(x, self.filter_shape, self.dilation)

    z = apply_fun(x)
    outputs = {"x": z, "log_det": jnp.zeros(self.batch_shape)}

    return outputs

class Flatten(Layer):

  def __init__(self, name: str="flatten", **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]

    if sample == False:
      unbatched_shape = self.unbatched_input_shapes["x"]
      flat_dim = util.list_prod(unbatched_shape)
      flat_shape = self.batch_shape + (flat_dim,)
      z = x.reshape(flat_shape)
    else:
      original_shape = self.batch_shape + self.unbatched_input_shapes["x"]
      z = x.reshape(original_shape)

    outputs = {"x": z, "log_det": jnp.zeros(self.batch_shape)}
    return outputs

class Reshape(Layer):

  def __init__(self,
               output_shape: Sequence[int],
               name: str="flatten",
               **kwargs):
    self.output_shape   = output_shape

    super().__init__(name=name, **kwargs)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]

    if sample == False:
      out_dim = util.list_prod(self.output_shape)
      expected_out_dim = util.list_prod(self.unbatched_input_shapes["x"])
      assert out_dim == expected_out_dim, f"Dimension mismatch"

      z = x.reshape(self.batch_shape + self.output_shape)
    else:
      original_shape = self.batch_shape + self.unbatched_input_shapes["x"]
      z = x.reshape(original_shape)

    outputs = {"x": z, "log_det": jnp.zeros(self.batch_shape)}
    return outputs

class Reverse(Layer):

  def __init__(self, name: str="reverse", **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    z = x[...,::-1]
    outputs = {"x": z, "log_det": jnp.zeros(self.batch_shape)}
    return outputs
