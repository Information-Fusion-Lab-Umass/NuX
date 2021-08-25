import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import nux.util as util
import einops

__all__ = ["LinfLipschitzDenseResBlock",
           "LinfLipschitzConvResBlock"]

class LinfLipschitzDense():

  def __init__(self, out_dim):
    self.dim_out = out_dim

  def get_params(self):
    return dict(w=self.w, b=self.b)

  def __call__(self, x, params=None, rng_key=None):
    dim_in = x.shape[-1]

    if params is None:
      self.w = random.normal(rng_key, shape=(self.dim_out, dim_in))*0.05
    else:
      self.w, self.b = params["w"], params["b"]

    linf_norm = jnp.abs(self.w).sum(axis=-1).max(axis=-1)
    x = jnp.einsum("ij,bj->bi", self.w, x)
    x /= linf_norm

    if params is None:
      mean = jnp.mean(x.reshape((-1, x.shape[-1])), axis=0)
      self.b = -mean
    x += self.b

    return x

class LinfLipschitzDenseBlock():
  def __init__(self, out_dim, dropout_prob):
    self.out_dim  = out_dim
    self.nonlinearity = util.str_to_nonlinearity("lipswish")
    self.dropout_prob = dropout_prob

    self.dense = LinfLipschitzDense(out_dim=self.out_dim)

  def get_params(self):
    return dict(dense=self.dense.get_params())

  def __call__(self, x, params=None, aux=None, rng_key=None, is_training=True):

    if params is None:
      self.dense_params = None
    else:
      self.dense_params = params["dense"]

    k1, k2 = random.split(rng_key, 2)

    # Dense
    gx = self.dense(x, params=self.dense_params, rng_key=k1)

    # Nonlinearity
    gx = self.nonlinearity(gx)

    # Dropout
    if is_training == True and self.dropout_prob > 0:
      keep_rate = 1.0 - self.dropout_prob
      mask = jax.random.bernoulli(k2, keep_rate, shape=gx.shape)
      gx = mask*gx/keep_rate

    return gx

################################################################################################################

class LinfLipschitzConv():

  def __init__(self, filter_shape, out_channel):
    self.filter_shape = filter_shape
    self.C_out = out_channel

  def get_params(self):
    return dict(w=self.w, b=self.b)

  def __call__(self, x, params=None, rng_key=None):
    C_in = x.shape[-1]

    if params is None:
      self.w = random.normal(rng_key, shape=self.filter_shape + (C_in, self.C_out))*0.05
    else:
      self.w, self.b = params["w"], params["b"]

    linf_norm = util.conv(jnp.abs(self.w), jnp.ones(x.shape[1:])).max()
    x = util.conv(self.w, x)
    x /= linf_norm

    if params is None:
      mean = jnp.mean(x.reshape((-1, x.shape[-1])), axis=0) + 1e-5
      self.b = -mean
    x += self.b

    return x

class LinfLipschitzConvBlock():
  def __init__(self, filter_shape, out_channel, dropout_prob):
    self.filter_shape = filter_shape
    self.out_channel  = out_channel
    self.nonlinearity = util.str_to_nonlinearity("lipswish")
    self.dropout_prob = dropout_prob

    self.conv = LinfLipschitzConv(filter_shape=self.filter_shape,
                                  out_channel=self.out_channel)

  def get_params(self):
    return dict(conv=self.conv.get_params())

  def __call__(self, x, params=None, aux=None, rng_key=None, is_training=True):

    if params is None:
      self.conv_params = None
    else:
      self.conv_params = params["conv"]

    k1, k2 = random.split(rng_key, 2)

    # Conv
    gx = self.conv(x, params=self.conv_params, rng_key=k1)

    # Nonlinearity
    gx = self.nonlinearity(gx)

    # Dropout
    if is_training == True and self.dropout_prob > 0:
      keep_rate = 1.0 - self.dropout_prob
      mask = jax.random.bernoulli(k2, keep_rate, shape=gx.shape)
      gx = mask*gx/keep_rate

    return gx

################################################################################################################

class LinfLipschitzDenseResBlock():
  def __init__(self, hidden_dim, n_layers, dropout_prob):
    self.hidden_dim = hidden_dim
    self.nonlinearity   = util.str_to_nonlinearity("lipswish")
    self.n_layers       = n_layers
    self.dropout_prob   = dropout_prob

  def get_params(self):
    return dict(res_block=self.res_params,
                in_proj=self.in_projection.get_params(),
                out_proj=self.out_projection.get_params())

  def __call__(self, x, params=None, aux=None, rng_key=None, is_training=True, **kwargs):

    if params is None:
      self.res_params, self.in_proj_params, self.out_proj_params = [None]*3
    else:
      self.res_params, self.in_proj_params, self.out_proj_params = params["res_block"], params["in_proj"], params["out_proj"]

    in_proj_key, out_proj_key, *keys = random.split(rng_key, self.n_layers + 2)
    keys = jnp.array(keys)

    # Apply the input projection
    self.in_projection = LinfLipschitzDenseBlock(out_dim=self.hidden_dim,
                                                 dropout_prob=self.dropout_prob)
    gx = self.in_projection(x, aux=aux, params=self.in_proj_params, rng_key=in_proj_key, is_training=is_training)

    def scan_block(carry, inputs):
      x = carry
      key, params = inputs
      transform = LinfLipschitzDenseBlock(out_dim=self.hidden_dim,
                                          dropout_prob=self.dropout_prob)
      x = transform(x, aux=aux, params=params, rng_key=key, is_training=is_training)
      return x, transform.get_params()

    if self.res_params is None:
      init_params = []
      for i, key in enumerate(keys):
        gx, block_params = scan_block(gx, (key, None))
        init_params.append(block_params)
      self.res_params = jax.tree_multimap(lambda *xs: jnp.array(xs), *init_params)
    else:
      gx, self.res_params = jax.lax.scan(scan_block, gx, (keys, self.res_params), unroll=10)

    # Apply the output projection
    self.out_projection = LinfLipschitzDense(out_dim=x.shape[-1])
    gx = self.out_projection(gx, params=self.out_proj_params, rng_key=out_proj_key)

    return gx

class LinfLipschitzConvResBlock():
  def __init__(self, filter_shape, hidden_channel, n_layers, dropout_prob):
    self.filter_shape   = filter_shape
    self.hidden_channel = hidden_channel
    self.nonlinearity   = util.str_to_nonlinearity("lipswish")
    self.n_layers       = n_layers
    self.dropout_prob   = dropout_prob

  def get_params(self):
    return dict(res_block=self.res_params,
                in_proj=self.in_projection.get_params(),
                out_proj=self.out_projection.get_params())

  def __call__(self, x, params=None, aux=None, rng_key=None, is_training=True, **kwargs):

    if params is None:
      self.res_params, self.in_proj_params, self.out_proj_params = [None]*3
    else:
      self.res_params, self.in_proj_params, self.out_proj_params = params["res_block"], params["in_proj"], params["out_proj"]

    in_proj_key, out_proj_key, *keys = random.split(rng_key, self.n_layers + 2)
    keys = jnp.array(keys)

    # Apply the input projection
    self.in_projection = LinfLipschitzConvBlock(filter_shape=self.filter_shape,
                                                out_channel=self.hidden_channel,
                                                dropout_prob=self.dropout_prob)
    gx = self.in_projection(x, aux=aux, params=self.in_proj_params, rng_key=in_proj_key, is_training=is_training)

    def scan_block(carry, inputs):
      x = carry
      key, params = inputs
      transform = LinfLipschitzConvBlock(filter_shape=self.filter_shape,
                                         out_channel=self.hidden_channel,
                                         dropout_prob=self.dropout_prob)
      x = transform(x, aux=aux, params=params, rng_key=key, is_training=is_training)
      return x, transform.get_params()

    if self.res_params is None:
      init_params = []
      for i, key in enumerate(keys):
        gx, block_params = scan_block(gx, (key, None))
        init_params.append(block_params)
      self.res_params = jax.tree_multimap(lambda *xs: jnp.array(xs), *init_params)
    else:
      gx, self.res_params = jax.lax.scan(scan_block, gx, (keys, self.res_params), unroll=10)

    # Apply the output projection
    self.out_projection = LinfLipschitzConv(filter_shape=self.filter_shape,
                                            out_channel=x.shape[-1])
    gx = self.out_projection(gx, params=self.out_proj_params, rng_key=out_proj_key)

    return gx

################################################################################################################

if __name__ == "__main__":
  from debug import *

  rng_key = random.PRNGKey(1)
  # x_shape = (16, 4, 4, 3)
  x_shape = (16, 3)
  x, aux = random.normal(rng_key, (2,)+x_shape)

  filter_shape    = (3, 3)
  hidden_channel  = 16
  dropout_prob    = 0.2
  n_layers        = 4
  # net = LinfLipschitzConvResBlock(filter_shape,
  #                                 hidden_channel,
  #                                 n_layers,
  #                                 dropout_prob)

  net = LinfLipschitzDenseResBlock(hidden_channel,
                                   n_layers,
                                   dropout_prob)

  z = net(x, aux=aux, rng_key=rng_key, is_training=False)
  params = net.get_params()


  def apply_fun(params, x):
    x = net(x, params=params, rng_key=rng_key)
    return x

  z2 = net(x, aux=aux, params=params, rng_key=rng_key, is_training=False)
  z3 = net(x[:4], aux=aux[:4], params=params, rng_key=rng_key, is_training=False)

  param_diff = jax.tree_multimap(lambda x,y: jnp.linalg.norm(x-y), params, net.get_params())

  import pdb; pdb.set_trace()
