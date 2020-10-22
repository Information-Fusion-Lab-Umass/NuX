import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence
from nux.flows.base import *
import nux.util as util
import nux.networks as net

__all__ = ["Coupling"]

from nux.flows.bijective.coupling_base import CouplingBase

class Coupling(CouplingBase):
  def __init__(self,
               create_network: Optional[Callable]=None,
               kind: Optional[str]="affine",
               axis: Optional[int]=-1,
               split_kind: str="channel",
               use_condition: bool=False,
               name: str="coupling",
               network_kwargs: Optional=None,
               **kwargs
  ):
    super().__init__(create_network=create_network,
                     axis=axis,
                     split_kind=split_kind,
                     use_condition=use_condition,
                     name=name,
                     network_kwargs=network_kwargs,
                     **kwargs)
    self.kind = kind

  def get_out_shape(self, x):
    x_shape = x.shape[len(self.batch_shape):]
    out_dim = x_shape[-1] if self.kind == "additive" else 2*x_shape[-1]
    return x_shape[:-1] + (out_dim,)

  def transform(self, x, params=None, sample=False):
    scale_init = hk.initializers.RandomNormal(stddev=0.01)
    if params is None:
      x_shape = x.shape[len(self.batch_shape):]
      if self.kind == "affine":
        log_s = hk.get_parameter("log_s", shape=x_shape, dtype=x.dtype, init=scale_init)
      t = hk.get_parameter("t", shape=x_shape, dtype=x.dtype, init=scale_init)

    else:
      if self.kind == "affine":
        scale_scale = hk.get_parameter("scale_scale", shape=(), dtype=x.dtype, init=scale_init)
      shift_scale = hk.get_parameter("shift_scale", shape=(), dtype=x.dtype, init=scale_init)

      # Split the output and bound the scaling term
      if self.kind == "affine":
        t, log_s = jnp.split(params, 2, axis=self.axis)
        log_s = jnp.tanh(log_s)
      else:
        t = params

      # Scale the parameters so that we can initialize this function to the identity
      t = shift_scale*t
      if self.kind == "affine":
        log_s = scale_scale*log_s

    if sample == False:
      z = (x - t)*jnp.exp(-log_s) if self.kind == "affine" else x - t
    else:
      z = x*jnp.exp(log_s) + t if self.kind == "affine" else x + t

    log_det = -jnp.sum(log_s) if self.kind == "affine" else jnp.zeros(self.batch_shape)

    return z, log_det
