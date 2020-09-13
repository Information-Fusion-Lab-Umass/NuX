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

__all__ = ["GLOW",
           "MultiscaleGLOW"]

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
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    def reducer(outputs):
      return {"mean": outputs["x"].mean(), "std": outputs["x"].std()}

    layers = []
    for i in range(self.n_blocks):
      layers.append(nux.track(nux.FlowNorm(), f"flow_norm_{i}_{self.name}", reducer))
      layers.append(nux.track(nux.OneByOneConv(), f"conv_{i}_{self.name}", reducer))
      layers.append(nux.track(nux.Coupling(n_channels=self.n_channels), f"coupling_{i}_{self.name}", reducer))

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
