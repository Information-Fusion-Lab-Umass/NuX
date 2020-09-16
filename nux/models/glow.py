import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping
from nux.flows.base import *
import nux.util as util
import nux

# __all__ = ["GLOW",
#            "MultiscaleGLOW"]

__all__ = ["glow",
           "multiscale_glow"]

def glow(n_blocks: int,
         n_channels: int=64,
         coupling_type: str='affine'):

  layers = []
  for i in range(n_blocks):
    layers.append(nux.ActNorm())
    layers.append(nux.OneByOneConv())
    layers.append(nux.Coupling(n_channels=n_channels, kind=coupling_type))

  flow = nux.sequential(*layers)
  return flow

def multiscale_glow(n_scales: int=2,
                    n_blocks: int=7,
                    n_channels: int=64,
                    coupling_type: str='affine'):

  def build_network(flow):
    return nux.sequential(glow(n_blocks, n_channels=n_channels, coupling_type=coupling_type),
                          nux.multi_scale(flow))

  flow = glow(n_blocks, n_channels=n_channels, coupling_type=coupling_type)
  for i in range(n_scales):
    flow = build_network(flow)

  return flow

################################################################################################################

class GLOW(Layer):

  def __init__(self,
               n_blocks: int,
               n_channels: int=64,
               name: str="glow",
               **kwargs
  ):
    super().__init__(name=name, **kwargs)
    self.n_blocks   = n_blocks
    self.n_channels = n_channels

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    layers = []
    for i in range(self.n_blocks):
      layers.append(nux.ActNorm())
      layers.append(nux.OneByOneConv())
      layers.append(nux.Coupling(n_channels=self.n_channels, kind='additive'))

    flow = nux.sequential(*layers)
    return flow(inputs, sample=sample, **kwargs)

class MultiscaleGLOW(Layer):

  def __init__(self,
               n_scales: int,
               n_blocks_per_scale: int,
               n_channels: int=64,
               name: str="multiscale_glow",
               **kwargs
  ):
    super().__init__(name=name, **kwargs)
    self.n_scales           = n_scales
    self.n_blocks_per_scale = n_blocks_per_scale
    self.n_channels         = n_channels

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    def build_network(flow):
      return nux.sequential(GLOW(self.n_blocks_per_scale, self.n_channels),
                            nux.multi_scale(flow))

    flow = GLOW(self.n_blocks_per_scale)
    for i in range(self.n_scales):
      flow = build_network(flow)

    return flow(inputs, sample=sample, **kwargs)
