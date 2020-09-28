import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Sequence
from nux.flows.base import *
import nux.util as util

__all__ = ["WidthSqueeze",
           "WidthUnSqueeze",
           "Squeeze",
           "UnSqueeze",
           "Flatten",
           "Reshape",
           "Reverse"]

class WidthSqueeze(AutoBatchedLayer):

  def __init__(self, name: str="width_squeeze", **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    if sample == False:
      H, W, C = x.shape
      z = x.reshape((H, W//2, 2, C)).transpose((0, 1, 3, 2)).reshape((H, W//2, 2*C))
    else:
      H, W, C = x.shape
      z = x.reshape((H, W, C//2, 2)).transpose((0, 1, 3, 2)).reshape((H, 2*W, C//2))

    outputs = {"x": z, "log_det": jnp.array(0.0)}
    return outputs

class WidthUnSqueeze(AutoBatchedLayer):

  def __init__(self, name: str="width_unsqueeze", **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    if sample == True:
      H, W, C = x.shape
      z = x.reshape((H, W//2, 2, C)).transpose((0, 1, 3, 2)).reshape((H, W//2, 2*C))
    else:
      H, W, C = x.shape
      z = x.reshape((H, W, C//2, 2)).transpose((0, 1, 3, 2)).reshape((H, 2*W, C//2))

    outputs = {"x": z, "log_det": jnp.array(0.0)}
    return outputs

class Squeeze(AutoBatchedLayer):

  def __init__(self,
               filter_shape: Sequence[int]=(2, 2),
               dilation: Sequence[int]=(1, 1),
               name: str="squeeze",
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.filter_shape = filter_shape
    self.dilation     = dilation

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    if sample == False:
      z = util.dilated_squeeze(x, self.filter_shape, self.dilation)
    else:
      z = util.dilated_unsqueeze(x, self.filter_shape, self.dilation)

    outputs = {"x": z, "log_det": jnp.array(0.0)}
    return outputs

class UnSqueeze(AutoBatchedLayer):

  def __init__(self,
               filter_shape: Sequence[int]=(2, 2),
               dilation: Sequence[int]=(1, 1),
               name: str="unsqueeze",
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.filter_shape = filter_shape
    self.dilation     = dilation

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    if sample == True:
      z = util.dilated_squeeze(x, self.filter_shape, self.dilation)
    else:
      z = util.dilated_unsqueeze(x, self.filter_shape, self.dilation)

    outputs = {"x": z, "log_det": jnp.array(0.0)}
    return outputs

class Flatten(AutoBatchedLayer):

  def __init__(self, original_shape: Sequence[int], name: str="flatten", **kwargs):
    super().__init__(name=name, **kwargs)
    self.original_shape = original_shape

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]

    # TODO: Fix this!  Need to find a way to store data dependent constants in a haiku context.
    # def init_fun(shape, dtype):
    #   return jnp.array(shape)
    # original_shape = hk.get_state("original_state", x.shape, jnp.int32, init_fun)

    if sample == False:
        z = x.ravel()
    else:
        z = x.reshape(self.original_shape)

    outputs = {"x": z, "log_det": jnp.array(0.0)}
    return outputs

class Reshape(AutoBatchedLayer):

  def __init__(self,
               original_shape: Sequence[int],
               output_shape: Sequence[int],
               name: str="flatten",
               **kwargs):
    assert jnp.prod(jnp.array(original_shape)) == jnp.prod(jnp.array(output_shape))
    self.original_shape = original_shape
    self.output_shape   = output_shape

    super().__init__(name=name, **kwargs)

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]

    # TODO: Fix this!  Need to find a way to store data dependent constants in a haiku context.
    # def init_fun(shape, dtype):
    #   return jnp.array(shape)
    # original_shape = hk.get_state("original_state", x.shape, jnp.int32, init_fun)

    if sample == False:
        z = x.reshape(self.output_shape)
    else:
        z = x.reshape(self.original_shape)

    outputs = {"x": z, "log_det": jnp.array(0.0)}
    return outputs

class Reverse(AutoBatchedLayer):

  def __init__(self, name: str="reverse", **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    z = x[...,::-1]
    outputs = {"x": z, "log_det": jnp.array(0.0)}
    return outputs
