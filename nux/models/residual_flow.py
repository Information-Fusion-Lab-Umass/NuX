import nux
from nux.flows.base import *
from typing import Optional, Mapping, Callable, Sequence

__all__ = ["residual_flow",
           "multiscale_residual_flow"]

def residual_flow(n_layers: int=10,
                  layer_sizes: Sequence[int]=[1024]*4,
                  n_channels: Optional[int]=256,
                  **kwargs):
    layers = []
    for i in range(n_layers):
        layers.append(nux.ResidualFlow(layer_sizes=layer_sizes, n_channels=n_channels, **kwargs))
    return nux.sequential(*layers)

################################################################################################################

def multiscale_residual_flow(n_scales: int=2,
                             n_layers: int=10,
                             n_channels: Optional[int]=256):

  def build_network(flow):
    return nux.sequential(residual_flow(n_layers, n_channels=n_channels),
                          nux.multi_scale(flow))

  flow = residual_flow(n_layers, n_channels=n_channels)
  for i in range(n_scales):
    flow = build_network(flow)

  return flow