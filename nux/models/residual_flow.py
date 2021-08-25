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
from nux.nn.lipschitz import LinfLipschitzDenseResBlock, LinfLipschitzConvResBlock

__all__ = ["ResidualFlowModel",
           "ResidualFlowImageModel"]

class ResidualFlowModel(Repeat):

  def __init__(self,
               hidden_dim=32,
               n_resnet_layers=1,
               dropout_prob=0.0,
               n_layers=3):
    res_block = LinfLipschitzDenseResBlock(hidden_dim=hidden_dim,
                                           n_layers=n_resnet_layers,
                                           dropout_prob=dropout_prob)
    self.flow = ResidualFlow(res_block)
    super().__init__(flow=self.flow, n_repeats=n_layers, checkerboard=False)

class ResidualFlowImageModel(Repeat):

  def __init__(self,
               filter_shape=(3, 3),
               hidden_channel=32,
               n_resnet_layers=1,
               dropout_prob=0.0,
               n_layers=3):
    res_block = LinfLipschitzConvResBlock(filter_shape=filter_shape,
                                          hidden_channel=hidden_channel,
                                          n_layers=n_resnet_layers,
                                          dropout_prob=dropout_prob)
    self.flow = ResidualFlow(res_block)
    super().__init__(flow=self.flow, n_repeats=n_layers, checkerboard=False)
