import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping
from nux.flows.base import *
import nux.util as util

__all__ = ["ActNorm",
           "FlowNorm"]

################################################################################################################

class ActNorm(Layer):

  def __init__(self, name: str="act_norm", **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    outputs = {}

    def b_init(*args, **kwargs):
      x = inputs["x"]
      axes = tuple(jnp.arange(len(x.shape) - 1))
      return jnp.mean(x, axis=axes)

    def log_s_init(*args, **kwargs):
      x = inputs["x"]
      axes = tuple(jnp.arange(len(x.shape) - 1))
      return jnp.log(jnp.std(x, axis=axes) + 1e-5)

    def const_init(*args, **kwargs):
      # We need to multiply the log determinant by the other dimensions
      x = inputs["x"]
      shape = [s for i, s in enumerate(x.shape[:-1]) if i not in Layer.batch_axes] + [1]
      return jnp.prod(jnp.array(shape))

    b     = hk.get_parameter("b", shape=(inputs["x"].shape[-1],), dtype=inputs["x"].dtype, init=b_init)
    log_s = hk.get_parameter("log_s", shape=(inputs["x"].shape[-1],), dtype=inputs["x"].dtype, init=b_init)
    const = hk.get_state("const", shape=(), dtype=jnp.float32, init=const_init)

    if sample == False:
      x = inputs["x"]
      outputs["x"] = (x - b)*jnp.exp(-log_s)
    else:
      z = inputs["x"]
      outputs["x"] = jnp.exp(log_s)*z + b

    outputs["log_det"] = -log_s.sum()*const

    return outputs

################################################################################################################

class FlowNorm(AutoBatchedLayer):

  def __init__(self, name: str="flow_norm", **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    outputs = {}

    def const_init(*args, **kwargs):
      # We need to multiply the log determinant by the other dimensions
      x = inputs["x"]
      shape = [s for i, s in enumerate(x.shape[:-1]) if i not in Layer.batch_axes] + [1]
      return jnp.prod(jnp.array(shape))

    dim, dtype = inputs["x"].shape[-1], inputs["x"].dtype
    b     = hk.get_parameter("b", shape=(dim,), dtype=dtype, init=jnp.zeros)
    log_s = hk.get_parameter("log_s", shape=(dim,), dtype=dtype, init=jnp.zeros)

    log_det = -log_s.sum()*jnp.prod(jnp.array(inputs["x"].shape[:-1]))

    outputs["log_det"] = log_det

    if sample == False:
      x = inputs["x"]
      outputs["x"] = (x - b)*jnp.exp(-log_s)

      # Add in the normalzation prior
      z = (jax.lax.stop_gradient(x) - b)*jnp.exp(-log_s)
      log_pz = -0.5*jnp.sum(z**2)
      outputs["flow_norm"] = log_pz + log_det
    else:
      z = inputs["x"]
      outputs["x"] = jnp.exp(log_s)*z + b

    return outputs
