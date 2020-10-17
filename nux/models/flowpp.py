import nux
from nux.flows.base import *
from typing import Optional, Mapping, Callable, Sequence

__all__ = ["FlowPP",
           "MultscaleFlowPP"]

def FlowPP(n_blocks: int=6,
           n_components: int=8,
           mixture_type="logistic",
           one_dim=False,
           **kwargs):

    mixture_kwargs = {}

    layers = []
    for i in range(n_blocks):
      layers.append(nux.ActNorm())
      if one_dim:
        layers.append(nux.AffineLDU())
      else:
        layers.append(nux.OneByOneConv())
      if(mixture_type == "logistic"):
        layers.append(nux.CouplingLogitsticMixtureLogit(n_components=n_components, **kwargs))
      elif(mixture_type == "logistic_linear"):
        layers.append(nux.CoupingLogisticMixtureCDFWithLogitLinear(n_components=n_components, **kwargs))
      elif(mixture_type == "gaussian"):
        layers.append(nux.CoupingGaussianMixtureCDFWithLogitLinear(n_components=n_components, **kwargs))
      else:
        assert 0

    return nux.sequential(*layers)

################################################################################################################

def MultscaleFlowPP(n_scales: int=2,
                    n_blocks: int=10,
                    n_components: int=8,
                    mixture_type="logistic",
                    **kwargs):

  return nux.sequential(nux.Squeeze(),
                        FlowPP(n_blocks, n_components, mixture_type, **kwargs),
                        nux.factored(FlowPP(n_blocks, n_components, mixture_type, **kwargs),
                                     nux.Identity()),
                        nux.Squeeze(),
                        nux.factored(FlowPP(n_blocks, n_components, mixture_type, **kwargs),
                                     nux.Identity()),
                        nux.UnSqueeze())

# def MultscaleFlowPP(n_scales: int=2,
#                     n_blocks: int=10,
#                     n_components: int=8,
#                     mixture_type="logistic",
#                     **kwargs):

#   def build_network(flow):
#     return nux.sequential(FlowPP(n_blocks, n_components, mixture_type, **kwargs),
#                           nux.multi_scale(flow))

#   flow = FlowPP(n_blocks, n_components, mixture_type, **kwargs)
#   for i in range(n_scales):
#     flow = build_network(flow)

#   return flow