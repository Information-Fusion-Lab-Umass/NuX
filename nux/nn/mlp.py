import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import nux.util as util
from .layers import WeightNormDense, GatedDense

__all__ = ["ResBlock1D",
           "ResNet1D",
           "CouplingResNet1D"]

class ResBlock1D():
  def __init__(self, hidden_dim, nonlinearity, dropout_prob):
    self.hidden_dim   = hidden_dim
    self.nonlinearity = nonlinearity
    self.dropout_prob = dropout_prob

  def get_params(self):
    return dict(gated_dense=self.gc.get_params())

  def __call__(self, x, params=None, aux=None, rng_key=None, is_training=True):

    if params is None:
      self.gc_params = None
    else:
      self.gc_params = params["gated_dense"]

    k1, k2 = random.split(rng_key, 2)

    self.gc = GatedDense(hidden_dim=self.hidden_dim,
                         nonlinearity=self.nonlinearity,
                         dropout_prob=self.dropout_prob)
    gx = self.gc(x, aux=aux, params=self.gc_params, rng_key=k1, is_training=is_training)
    x += gx

    return x

class ResNet1D():
  def __init__(self, hidden_dim, nonlinearity, dropout_prob, n_layers, unroll=1):
    self.hidden_dim   = hidden_dim
    self.nonlinearity = nonlinearity
    self.dropout_prob = dropout_prob
    self.n_layers     = n_layers
    self.params = None
    self.unroll = unroll

  def get_params(self):
    return self.params

  def __call__(self, x, params=None, aux=None, rng_key=None, is_training=True, **kwargs):

    if params is None:
      self.params = None
    else:
      self.params = params

    def scan_block(carry, inputs):
      x = carry
      key, params = inputs
      res_block = ResBlock1D(self.hidden_dim, self.nonlinearity, self.dropout_prob)
      x = res_block(x, aux=aux, params=params, rng_key=key, is_training=is_training)
      return x, res_block.get_params()

    keys = random.split(rng_key, self.n_layers)

    if self.params is None:
      init_params = []
      for i, key in enumerate(keys):
        x, block_params = scan_block(x, (key, None))
        init_params.append(block_params)
      self.params = jax.tree_util.tree_map(lambda *xs: jnp.array(xs), *init_params)
    else:
      if False:
        params = []
        for i, key in enumerate(keys):
          x, block_params = scan_block(x, (key, jax.tree_util.tree_map(lambda x: x[i], self.params)))
          params.append(block_params)
        self.params = jax.tree_util.tree_map(lambda *xs: jnp.array(xs), *params)
      else:
        x, self.params = jax.lax.scan(scan_block, x, (keys, self.params), unroll=self.unroll)
    return x

class CouplingResNet1D():
  def __init__(self,
               out_dim,
               working_dim,
               hidden_dim,
               nonlinearity,
               dropout_prob,
               n_layers,
               unroll=1):
    self.out_dim      = out_dim
    self.working_dim  = working_dim
    self.hidden_dim   = hidden_dim
    self.nonlinearity = nonlinearity
    self.dropout_prob = dropout_prob
    self.n_layers     = n_layers
    self.unroll       = unroll

  def get_params(self):
    return dict(wn_in=self.wn_in.get_params(),
                resnet=self.resnet.get_params(),
                wn_out=self.wn_out.get_params())

  def __call__(self, x, params=None, aux=None, rng_key=None, is_training=True, **kwargs):
    k1, k2, k3 = random.split(rng_key, 3)

    if params is None:
      self.wn_in_params, self.resnet_params, self.wn_out_params = [None]*3
    else:
      self.wn_in_params, self.resnet_params, self.wn_out_params = params["wn_in"], params["resnet"], params["wn_out"]

    # Project to working dim
    self.wn_in = WeightNormDense(out_dim=self.working_dim)
    x = self.wn_in(x, params=self.wn_in_params, rng_key=k1)

    # ResNet
    self.resnet = ResNet1D(hidden_dim=self.hidden_dim,
                           nonlinearity=self.nonlinearity,
                           dropout_prob=self.dropout_prob,
                           n_layers=self.n_layers,
                           unroll=self.unroll)
    x = self.resnet(x, aux=aux, params=self.resnet_params, rng_key=k2, is_training=is_training)

    # Project to output channel
    self.wn_out = WeightNormDense(out_dim=self.out_dim)
    x = self.wn_out(x, params=self.wn_out_params, rng_key=k3)

    return x

################################################################################################################

if __name__ == "__main__":
  from debug import *

  rng_key = random.PRNGKey(1)
  x_shape = (16, 4)
  x, aux = random.normal(rng_key, (2,)+x_shape)

  out_dim      = 6
  working_dim  = 16
  hidden_dim   = 16
  nonlinearity = util.square_swish
  dropout_prob = 0.2
  n_layers     = 1
  net = CouplingResNet1D(out_dim,
                         working_dim,
                         hidden_dim,
                         nonlinearity,
                         dropout_prob,
                         n_layers)

  z = net(x, aux=aux, rng_key=rng_key, is_training=False)
  params = net.get_params()

  z2 = net(x, aux=aux, params=params, rng_key=rng_key, is_training=False)
  z3 = net(x[:4], aux=aux[:4], params=params, rng_key=rng_key, is_training=False)

  param_diff = jax.tree_util.tree_map(lambda x,y: jnp.linalg.norm(x-y), params, net.get_params())

  import pdb; pdb.set_trace()
