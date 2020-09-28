import nux
from typing import Optional, Mapping, Callable, Sequence

__all__ = ["RQSpline"]

def RQSpline(n_layers: int=10,
             n_bins: int=10,
             layer_sizes: Sequence[int]=[1024]*4,
             bounds: Sequence[float]=((-4.0, 4.0), (-4.0, 4.0))):

    layers = []
    for i in range(n_layers):
        layers.append(nux.ActNorm())
        layers.append(nux.AffineLDU())
        layers.append(nux.NeuralSpline(n_bins, layer_sizes=layer_sizes, bounds=bounds))

    return nux.sequential(*layers)