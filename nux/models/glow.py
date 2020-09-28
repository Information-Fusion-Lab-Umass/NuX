import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Sequence
from nux.flows.base import *
import nux.util as util
import nux

__all__ = ["GLOW",
           "MultiscaleGLOW"]

################################################################################################################

def GLOW(n_blocks: int,
         n_channels: int=64,
         coupling_type: str="affine",
         parameter_norm: Optional[str]=None,
         split_kind: str="channel"):

  layers = []
  for i in range(n_blocks):
    layers.append(nux.ActNorm())
    layers.append(nux.OneByOneConv())
    layers.append(nux.Coupling(n_channels=n_channels,
                               kind=coupling_type,
                               parameter_norm=parameter_norm,
                               split_kind=split_kind))

  return nux.sequential(*layers)

def GLOW2(n_blocks: int,
          n_channels: int=64,
          coupling_type: str="affine",
          parameter_norm: Optional[str]=None,
          filter_shape: Sequence[int]=(2, 2),
          dilation: Sequence[int]=(1, 1)):

  layers = []
  for i in range(n_blocks):
    layers.append(nux.Squeeze(filter_shape=filter_shape, dilation=dilation))
    layers.append(nux.ActNorm())
    layers.append(nux.OneByOneConv())
    layers.append(nux.UnSqueeze(filter_shape=filter_shape, dilation=dilation))
    layers.append(nux.Coupling(n_channels=n_channels,
                               kind=coupling_type,
                               parameter_norm=parameter_norm,
                               split_kind="checkerboard"))

  return nux.sequential(*layers)

################################################################################################################

def multiscale_glow(n_blocks: int,
                    n_channels: int=64,
                    coupling_type: str="affine",
                    parameter_norm: Optional[str]=None,
                    split_kind="checkerboard"):

  def create_block():
    transform = nux.sequential(nux.ActNorm(),
                               nux.OneByOneConv(),
                               nux.Coupling(n_channels=n_channels, kind=coupling_type, parameter_norm=parameter_norm))
    return nux.multi_scale(transform, split_kind=split_kind)

  layers = []
  for i in range(n_blocks):
    layers.append(create_block())
    layers.append(nux.Reverse())

  return nux.sequential(*layers)

################################################################################################################

def MultiscaleGLOW(n_scales: int,
                   n_blocks: int,
                   n_channels: int=64,
                   coupling_type: str="affine",
                   parameter_norm: Optional[str]=None):

  def build_network(flow):
    return nux.sequential(GLOW(n_blocks, n_channels, coupling_type, parameter_norm),
                          nux.multi_scale(flow))

  flow = GLOW(n_blocks, n_channels, coupling_type, parameter_norm)
  for i in range(n_scales):
    flow = build_network(flow)

  return flow
