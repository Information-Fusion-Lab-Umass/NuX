import nux
from nux.flows.base import *
from typing import Optional, Mapping, Callable, Sequence

__all__ = ["FlowPP",
           "MultscaleFlowPP"]

def FlowPP(n_blocks: int=6,
           n_components: int=8,
           mixture_type="logistic",
           actnorm=True,
           one_dim=False,
           use_condition=False,
           reverse=False,
           network_kwargs: Optional=None):

    mixture_kwargs = {}

    layers = []
    for i in range(n_blocks):
      if actnorm:
        layers.append(nux.ActNorm())
      if one_dim:
        layers.append(nux.AffineLDU())
      else:
        layers.append(nux.OneByOneConv())
      if(mixture_type == "logistic"):
        layers.append(nux.CouplingLogitsticMixtureLogit(n_components=n_components,
                                                        network_kwargs=network_kwargs,
                                                        use_condition=use_condition,
                                                        reverse=reverse))
      elif(mixture_type == "gaussian"):
        layers.append(nux.CouplingGaussianMixtureLogit(n_components=n_components,
                                                       network_kwargs=network_kwargs,
                                                       use_condition=use_condition,
                                                       reverse=reverse))
      else:
        assert 0

    return nux.sequential(*layers)

################################################################################################################

def MultscaleFlowPP(n_scales: int=2,
                    n_blocks: int=10,
                    n_components: int=8,
                    mixture_type="logistic",
                    actnorm=True,
                    network_kwargs: Optional=None):

  def build_network(flow):
    return nux.sequential(FlowPP(n_blocks, n_components, mixture_type, actnorm, one_dim=False, network_kwargs=network_kwargs),
                          nux.multi_scale(flow))

  flow = FlowPP(n_blocks, n_components, mixture_type, actnorm, one_dim=False, network_kwargs=network_kwargs)
  for i in range(n_scales):
    flow = build_network(flow)

  return flow