# from jax.config import config
# config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
import nux.util as util
from nux.flows.base import Sequential, Repeat
from nux.flows.bijective.residual_flow import ResidualFlow
from nux.flows.bijective.affine import ShiftScale, PLUMVP
from nux.flows.bijective.conv import OneByOneConv
from nux.nn.lipschitz import LipschitzDenseResBlock, LipschitzConvResBlock

__all__ = ["ResidualFlowModel",
           "ResidualFlowImageModel"]

class ResidualFlowModel(Repeat):

  def __init__(self,
               hidden_dim=32,
               n_resnet_layers=1,
               dropout_prob=0.0,
               n_layers=3,
               norm="l2",
               with_actnorm=True,
               glow=False,
               nonlinearity="lipswish",
               sn_iters=3,
               sn_scale=0.9):
    res_block = LipschitzDenseResBlock(hidden_dim=hidden_dim,
                                       n_layers=n_resnet_layers,
                                       dropout_prob=dropout_prob,
                                       norm=norm,
                                       nonlinearity=nonlinearity,
                                       sn_iters=sn_iters,
                                       sn_scale=sn_scale)
    if glow:
      self.flow = Sequential([PLUMVP(), ShiftScale(), ResidualFlow(res_block)])
    elif with_actnorm:
      self.flow = Sequential([ShiftScale(), ResidualFlow(res_block)])
    else:
      self.flow = ResidualFlow(res_block)
    super().__init__(flow=self.flow, n_repeats=n_layers, checkerboard=False)

class ResidualFlowImageModel(Repeat):

  def __init__(self,
               hidden_channel=32,
               n_resnet_layers=1,
               dropout_prob=0.0,
               n_layers=3,
               norm="l2",
               with_actnorm=True,
               glow=False,
               nonlinearity="lipswish",
               scaled_ws=False,
               sn_iters=3,
               sn_scale=0.9):
    res_block = LipschitzConvResBlock(filter_shape=(1, 1),
                                      hidden_channel=hidden_channel,
                                      n_layers=n_resnet_layers,
                                      dropout_prob=dropout_prob,
                                      norm=norm,
                                      nonlinearity=nonlinearity,
                                      scaled_ws=scaled_ws,
                                      sn_iters=sn_iters,
                                      sn_scale=sn_scale)
    if glow:
      self.flow = Sequential([OneByOneConv(), ShiftScale(), ResidualFlow(res_block)])
    elif with_actnorm:
      self.flow = Sequential([ShiftScale(), ResidualFlow(res_block)])
    else:
      self.flow = ResidualFlow(res_block)
    super().__init__(flow=self.flow, n_repeats=n_layers, checkerboard=False)

################################################################################################################

if __name__ == "__main__":
  from debug import *
  import nux

  rng_key = random.PRNGKey(1)
  # x_shape = (16, 4, 4, 3)
  x_shape = (16, 3)
  x, aux = random.normal(rng_key, (2,)+x_shape)

  filter_shape    = (3, 3)
  hidden_channel  = 16
  dropout_prob    = 0.2
  n_layers        = 4
  # res_block = nux.LipschitzConvResBlock(filter_shape,
  #                                           hidden_channel,
  #                                           n_layers,
  #                                           dropout_prob)

  res_block = nux.LipschitzDenseResBlock(hidden_channel,
                                             n_layers,
                                             dropout_prob)

  flow = ResidualFlow(res_block)

  z, log_det = flow(x, rng_key=rng_key)
  params = flow.get_params()

  import pdb; pdb.set_trace()
