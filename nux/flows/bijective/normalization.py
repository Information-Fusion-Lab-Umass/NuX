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

  def __init__(self,
               name: str="act_norm"
  ):
    """ Act norm.  Used in GLOW https://arxiv.org/pdf/1807.03039.pdf
    Args:
      name : Optional name for this module.
    """
    super().__init__(name=name)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    outputs = {}
    x = inputs["x"]
    x_shape = self.get_unbatched_shapes(sample)["x"]
    dtype = x.dtype

    def b_init(*args, **kwargs):
      axes = tuple(jnp.arange(len(x.shape) - 1))
      return jnp.mean(x, axis=axes)

    def log_s_init(*args, **kwargs):
      axes = tuple(jnp.arange(len(x.shape) - 1))
      return jnp.log(jnp.std(x, axis=axes) + 1e-5)

    b     = hk.get_parameter("b", shape=(x_shape[-1],), dtype=dtype, init=b_init)
    log_s = hk.get_parameter("log_s", shape=(x_shape[-1],), dtype=dtype, init=log_s_init)

    if sample == False:
      outputs["x"] = (x - b)*jnp.exp(-log_s)
    else:
      outputs["x"] = jnp.exp(log_s)*x + b

    log_det = -log_s.sum()*util.list_prod(x_shape[:-1])
    outputs["log_det"] = log_det*jnp.ones(self.batch_shape)

    return outputs
