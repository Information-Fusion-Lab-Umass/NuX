import nux
from typing import Optional, Mapping, Callable, Sequence

__all__ = ["RealNVP"]

def RealNVP(n_layers: int=10,
            hidden_layer_sizes: Sequence[int]=[1024]*4):

    layers = []
    for i in range(n_layers):
        layers.append(nux.ActNorm())
        layers.append(nux.Reverse())
        layers.append(nux.Coupling(hidden_layer_sizes=hidden_layer_sizes, kind='additive'))

    return nux.sequential(*layers)