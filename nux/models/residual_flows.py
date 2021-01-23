import nux
from typing import Optional, Mapping, Callable, Sequence
from functools import partial
import nux.networks as net

def ResidualFlowArchitecture(*,
                             hidden_channel_size,
                             actnorm,
                             one_by_one_conv,
                             repititions):

  if isinstance(repititions, int):
    repititions = [repititions]

  def create_resnet_network(out_shape):
    return net.ReverseBottleneckConv(out_channel=out_shape[-1],
                                     hidden_channel=hidden_channel_size,
                                     nonlinearity="lipswish",
                                     normalization=None,
                                     parameter_norm="differentiable_spectral_norm",
                                     use_bias=True,
                                     dropout_rate=None,
                                     gate=False,
                                     activate_last=False,
                                     max_singular_value=0.999,
                                     max_power_iters=1)

  def block():
    layers = []
    if actnorm:
      layers.append(nux.ActNorm(axis=(-3, -2, -1)))
    if one_by_one_conv:
      layers.append(nux.OneByOneConv())
    layers.append(nux.ResidualFlow(create_network=create_resnet_network))
    return nux.sequential(*layers)

  layers = []
  for i, r in enumerate(repititions):
    if i > 0:
      layers.append(nux.Squeeze())

    layers.append(nux.repeat(block, n_repeats=r))

  return nux.sequential(*layers)
