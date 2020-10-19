import nux
from typing import Optional, Mapping, Callable, Sequence

__all__ = ["RealNVP",
           "MultiscaleRealNVP"]

################################################################################################################

def RealNVP(n_blocks: int=10,
            coupling_kind="affine",
            actnorm=True,
            network_kwargs: Optional=None):

  layers = []
  for i in range(n_blocks):
    if actnorm:
      layers.append(nux.ActNorm())
    layers.append(nux.Reverse())
    layers.append(nux.Coupling(kind=coupling_kind, network_kwargs=network_kwargs))

  return nux.sequential(*layers)

################################################################################################################

def MultiscaleRealNVP(n_scales: int,
                      n_blocks: int,
                      coupling_type: str="affine",
                      actnorm=True,
                      network_kwargs: Optional=None):

  def build_network(flow):
    return nux.sequential(RealNVP(n_blocks, coupling_type, actnorm, network_kwargs),
                          nux.multi_scale(flow))

  flow = RealNVP(n_blocks, coupling_type, actnorm, network_kwargs)
  for i in range(n_scales):
    flow = build_network(flow)

  return flow
