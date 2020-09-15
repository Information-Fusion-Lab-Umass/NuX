import nux
from typing import Optional, Mapping, Callable, Sequence

__all__ = ["residual_flow",
           "multiscale_residual_flow"]

def residual_flow(n_layers: int=10,
                  n_hidden_layers: Sequence[int]=[1024]*4,
                  n_channels: Optional[int]=256):
    layers = []
    for i in range(n_layers):
        layers.append(nux.ResidualFlow(n_hidden_layers=n_hidden_layers, n_channels=n_channels))
    return nux.sequential(*layers)

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