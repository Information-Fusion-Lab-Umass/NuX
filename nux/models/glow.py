import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Sequence
from nux.flows.base import *
import nux.util as util
import nux

__all__ = ["GLOW",
           "MultiscaleGLOW"]

################################################################################################################

def GLOW(n_blocks: int,
         coupling_type: str="affine",
         actnorm=True,
         network_kwargs: Optional=None):

  layers = []
  for i in range(n_blocks):
    if actnorm:
      layers.append(nux.ActNorm())
    layers.append(nux.OneByOneConv())
    layers.append(nux.Coupling(kind=coupling_type,
                               network_kwargs=network_kwargs))

  return nux.sequential(*layers)

################################################################################################################

def MultiscaleGLOW(n_scales: int,
                   n_blocks: int,
                   coupling_type: str="affine",
                   actnorm=True,
                   network_kwargs: Optional=None):

  def build_network(flow):
    return nux.sequential(GLOW(n_blocks, coupling_type, actnorm, network_kwargs),
                          nux.multi_scale(flow))

  flow = GLOW(n_blocks, coupling_type, actnorm, network_kwargs)
  for i in range(n_scales):
    flow = build_network(flow)

  return flow
