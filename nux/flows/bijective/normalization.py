import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping
from nux.flows.base import *
import nux.util as util

__all__ = ["ActNorm"]

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

    log_det = -log_s.sum()*const
    outputs["log_det"] = log_det

    if sample == False:
      x = inputs["x"]
      flow_norm = kwargs.get("flow_norm", False)
      if(flow_norm):
        x = jax.lax.stop_gradient(x)

      z = (x - b)*jnp.exp(-log_s)

      if(flow_norm):
        axes = -1 if z.ndim < 3 else (-1, -2, -3)
        outputs["flow_norm"] = -0.5*jnp.sum(z**2, axis=axes) + log_det

      outputs["x"] = z
    else:
      z = inputs["x"]
      outputs["x"] = jnp.exp(log_s)*z + b


    return outputs
