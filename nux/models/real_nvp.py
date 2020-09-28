import nux
from typing import Optional, Mapping, Callable, Sequence

__all__ = ["RealNVP"]

def RealNVP(n_layers: int=10,
            layer_sizes: Sequence[int]=[1024]*4,
            coupling_kind="affine",
            actnorm=True):

  layers = []
  for i in range(n_layers):
    if actnorm:
      layers.append(nux.ActNorm())
    layers.append(nux.Reverse())
    layers.append(nux.Coupling(layer_sizes=layer_sizes, kind=coupling_kind))

  return nux.sequential(*layers)