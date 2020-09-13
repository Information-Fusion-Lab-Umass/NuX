import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping
from nux.flows.base import *
import nux.util as util

__all__ = ["Squeeze",
           "UnSqueeze",
           "Flatten",
           "Reverse"]

class Squeeze(AutoBatchedLayer):

  def __init__(self, name: str="squeeze", **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    if sample == False:
      z = util.dilated_squeeze(x, (2, 2), (1, 1))
    else:
      z = util.dilated_unsqueeze(x, (2, 2), (1, 1))

    outputs = {"x": z, "log_det": jnp.array(0.0)}
    return outputs

class UnSqueeze(AutoBatchedLayer):

  def __init__(self, name: str="unsqueeze", **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    if sample == True:
      z = util.dilated_squeeze(x, (2, 2), (1, 1))
    else:
      z = util.dilated_unsqueeze(x, (2, 2), (1, 1))

    outputs = {"x": z, "log_det": jnp.array(0.0)}
    return outputs


class Flatten(AutoBatchedLayer):

  def __init__(self, original_shape, name: str="flatten", **kwargs):
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

class Reverse(AutoBatchedLayer):

  def __init__(self, name: str="reverse", **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    z = x[...,::-1]
    outputs = {"x": z, "log_det": jnp.array(0.0)}
    return outputs
