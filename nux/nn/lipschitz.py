import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import nux.util as util
import einops

__all__ = ["LipschitzDenseResBlock",
           "LipschitzConvResBlock"]

def mvp(w, x):
  return jnp.dot(x, w.T)

class L2LipschitzDense():

  def __init__(self, out_dim, sn_iters=3, sn_scale=0.9):
    self.dim_out = out_dim
    self.sn_iters = sn_iters
    self.sn_scale = sn_scale

  def get_params(self):
    return dict(w=self.w, b=self.b, v=self.v)

  def __call__(self, x, params=None, rng_key=None, sv_update=True, max_sv_update=False, **kwargs):
    dim_in = x.shape[-1]

    if params is None:
      self.w = random.normal(rng_key, shape=(self.dim_out, dim_in))*0.05

      # Initialize v with the correct value
      self.v = random.normal(rng_key, shape=x.shape[1:])
      n_iters = -1
    else:
      self.w, self.b, self.v = params["w"], params["b"], params["v"]
      n_iters = self.sn_iters

    if max_sv_update:
      n_iters = -1

    if sv_update == False:
      n_iters = 0

    sigma, v = util.max_singular_value(partial(mvp, self.w), self.v, n_iters=n_iters)
    if sv_update == True:
      self.v = jax.lax.stop_gradient(v)

    x = jnp.einsum("ij,bj->bi", self.w, x)
    factor = jnp.where(self.sn_scale < sigma, self.sn_scale/sigma, 1.0)
    x *= factor

    if params is None:
      mean = jnp.mean(x.reshape((-1, x.shape[-1])), axis=0)
      self.b = -mean
    x += self.b

    return x

class L2LipschitzConv():

  def __init__(self, filter_shape, out_channel, scaled_ws=False, sn_iters=3, sn_scale=0.9):
    self.filter_shape = filter_shape
    self.C_out = out_channel
    self.scaled_ws = scaled_ws
    self.sn_iters = sn_iters
    self.sn_scale = sn_scale

  def get_params(self):
    return dict(w=self.w, b=self.b, v=self.v)

  def __call__(self, x, params=None, rng_key=None, sv_update=True, max_sv_update=False, **kwargs):
    C_in = x.shape[-1]

    if params is None:
      # self.w = random.normal(rng_key, shape=self.filter_shape + (C_in, self.C_out))*0.05
      w_init = jax.nn.initializers.glorot_normal(in_axis=-2, out_axis=-1, dtype=x.dtype)
      self.w = w_init(rng_key, shape=self.filter_shape + (C_in, self.C_out))

      if self.scaled_ws:
        # No gain because we need to divide by spectral norm
        self.w = util.scaled_weight_standardization_conv(self.w)

      # Initialize v with the correct value
      if self.filter_shape != (1, 1):
        self.v = random.normal(rng_key, shape=x.shape[1:])
        matrix_prod = partial(util.conv, self.w)
      else:
        self.v = random.normal(rng_key, shape=self.w.shape[-1:])
        matrix_prod = partial(mvp, self.w[0,0])
      n_iters = -1
    else:
      self.w, self.b, self.v = params["w"], params["b"], params["v"]

      if self.scaled_ws:
        # No gain because we need to divide by spectral norm
        self.w = util.scaled_weight_standardization_conv(self.w)

      if self.filter_shape != (1, 1):
        matrix_prod = partial(util.conv, self.w)
      else:
        matrix_prod = partial(mvp, self.w[0,0])
      n_iters = self.sn_iters

    if max_sv_update:
      n_iters = -1

    if sv_update == False:
      n_iters = 0

    sigma, v = util.max_singular_value(matrix_prod, self.v, n_iters=n_iters)
    if sv_update == True:
      self.v = jax.lax.stop_gradient(v)

    x = util.conv(self.w, x)
    factor = jnp.where(self.sn_scale < sigma, self.sn_scale/sigma, 1.0)
    x *= factor

    if params is None:
      mean = jnp.mean(x.reshape((-1, x.shape[-1])), axis=0) + 1e-5
      self.b = -mean
    x += self.b

    return x

################################################################################################################

class LinfLipschitzDense():

  def __init__(self, out_dim):
    self.dim_out = out_dim

  def get_params(self):
    return dict(w=self.w, b=self.b)

  def __call__(self, x, params=None, rng_key=None, **kwargs):
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

class LinfLipschitzConv():

  def __init__(self, filter_shape, out_channel):
    self.filter_shape = filter_shape
    self.C_out = out_channel

  def get_params(self):
    return dict(w=self.w, b=self.b)

  def __call__(self, x, params=None, rng_key=None, **kwargs):
    C_in = x.shape[-1]

    if params is None:
      init = jax.nn.initializers.glorot_normal()
      self.w = init(rng_key, shape=self.filter_shape + (C_in, self.C_out))
      # self.w = random.normal(rng_key, shape=self.filter_shape + (C_in, self.C_out))*0.05
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

################################################################################################################

class LipschitzDenseBlock():
  def __init__(self,
               out_dim,
               dropout_prob,
               norm="l2",
               nonlinearity="lipswish",
               sn_iters=3,
               sn_scale=0.9):
    self.out_dim  = out_dim
    if isinstance(nonlinearity, str):
      assert nonlinearity in ["lipswish", "square_lipswish"]
      self.nonlinearity = util.str_to_nonlinearity(nonlinearity)
    else:
      self.nonlinearity = nonlinearity
    self.dropout_prob = dropout_prob
    self.norm = norm
    self.sn_iters = sn_iters
    self.sn_scale = sn_scale

    if self.norm == "l2":
      self.dense = L2LipschitzDense(out_dim=self.out_dim, sn_iters=self.sn_iters, sn_scale=self.sn_scale)
    elif self.norm == "linf":
      self.dense = LinfLipschitzDense(out_dim=self.out_dim)
    else:
      assert 0, "Invalid norm"

  def get_params(self):
    return dict(dense=self.dense.get_params())

  def __call__(self, x, params=None, aux=None, rng_key=None, is_training=True, sv_update=True, **kwargs):

    if params is None:
      self.dense_params = None
    else:
      self.dense_params = params["dense"]

    k1, k2 = random.split(rng_key, 2)

    # Dense
    gx = self.dense(x, params=self.dense_params, rng_key=k1, sv_update=sv_update, **kwargs)

    # Nonlinearity
    gx = self.nonlinearity(gx)

    # Dropout
    if is_training == True and self.dropout_prob > 0:
      keep_rate = 1.0 - self.dropout_prob
      mask = jax.random.bernoulli(k2, keep_rate, shape=gx.shape)
      gx = mask*gx/keep_rate

    return gx

################################################################################################################

class LipschitzConvBlock():
  def __init__(self,
               filter_shape,
               out_channel,
               dropout_prob,
               norm="l2",
               nonlinearity="lipswish",
               scaled_ws=False,
               sn_iters=3,
               sn_scale=0.9):
    self.filter_shape = filter_shape
    self.out_channel  = out_channel
    if isinstance(nonlinearity, str):
      assert nonlinearity in ["lipswish", "square_lipswish"]
      self.nonlinearity = util.str_to_nonlinearity(nonlinearity)
    else:
      self.nonlinearity = nonlinearity
    self.dropout_prob = dropout_prob
    self.norm = norm
    self.scaled_ws = scaled_ws
    self.sn_iters = sn_iters
    self.sn_scale = sn_scale

    if self.norm == "l2":
      self.conv = L2LipschitzConv(filter_shape=self.filter_shape,
                                  out_channel=self.out_channel,
                                  scaled_ws=self.scaled_ws,
                                  sn_iters=self.sn_iters,
                                  sn_scale=self.sn_scale)
    else:
      self.conv = LinfLipschitzConv(filter_shape=self.filter_shape,
                                    out_channel=self.out_channel)

  def get_params(self):
    return dict(conv=self.conv.get_params())

  def __call__(self, x, params=None, aux=None, rng_key=None, is_training=True, sv_update=True, **kwargs):

    if params is None:
      self.conv_params = None
    else:
      self.conv_params = params["conv"]

    k1, k2 = random.split(rng_key, 2)

    # Conv
    gx = self.conv(x, params=self.conv_params, rng_key=k1, sv_update=sv_update, **kwargs)

    # Nonlinearity
    gx = self.nonlinearity(gx)

    # Dropout
    if is_training == True and self.dropout_prob > 0:
      keep_rate = 1.0 - self.dropout_prob
      mask = jax.random.bernoulli(k2, keep_rate, shape=gx.shape)
      gx = mask*gx/keep_rate

    return gx

################################################################################################################

class LipschitzDenseResBlock():
  def __init__(self,
               hidden_dim,
               n_layers,
               dropout_prob,
               preactivation=True,
               norm="l2",
               nonlinearity="lipswish",
               sn_iters=3,
               sn_scale=0.9):
    self.hidden_dim = hidden_dim
    assert nonlinearity in ["lipswish", "square_lipswish"]
    self.nonlinearity   = util.str_to_nonlinearity(nonlinearity)
    self.n_layers       = n_layers
    self.dropout_prob   = dropout_prob
    self.preactivation = preactivation
    self.norm = norm
    self.sn_iters = sn_iters
    self.sn_scale = sn_scale

  def get_params(self):
    return dict(res_block=self.res_params,
                in_proj=self.in_projection.get_params(),
                out_proj=self.out_projection.get_params())

  def __call__(self, x, params=None, aux=None, rng_key=None, is_training=True, update_params=True, **kwargs):

    if params is None:
      self.res_params, self.in_proj_params, self.out_proj_params = [None]*3
    else:
      self.res_params, self.in_proj_params, self.out_proj_params = params["res_block"], params["in_proj"], params["out_proj"]

    in_proj_key, out_proj_key, *keys = random.split(rng_key, self.n_layers + 2)
    keys = jnp.array(keys)

    gx = x

    if self.preactivation:
      gx = self.nonlinearity(gx)

    # Apply the input projection
    self.in_projection = LipschitzDenseBlock(out_dim=self.hidden_dim,
                                             dropout_prob=self.dropout_prob,
                                             norm=self.norm,
                                             sn_iters=self.sn_iters,
                                             sn_scale=self.sn_scale)
    gx = self.in_projection(gx, aux=aux, params=self.in_proj_params, rng_key=in_proj_key, is_training=is_training, **kwargs)

    def scan_block(carry, inputs):
      x = carry
      key, params = inputs
      transform = LipschitzDenseBlock(out_dim=self.hidden_dim,
                                      dropout_prob=self.dropout_prob,
                                      norm=self.norm,
                                      sn_iters=self.sn_iters,
                                      sn_scale=self.sn_scale)
      x = transform(x, aux=aux, params=params, rng_key=key, is_training=is_training, **kwargs)
      return x, transform.get_params()

    if self.res_params is None:
      init_params = []
      for i, key in enumerate(keys):
        gx, block_params = scan_block(gx, (key, None))
        init_params.append(block_params)
      if self.n_layers > 0:
        self.res_params = jax.tree_util.tree_map(lambda *xs: jnp.array(xs), *init_params)
      else:
        self.res_params = ()
    else:
      if self.n_layers > 0:
        gx, res_params = jax.lax.scan(scan_block, gx, (keys, self.res_params), unroll=1)
        if update_params:
          self.res_params = res_params
      else:
        self.res_params = ()

    # Apply the output projection
    if self.norm == "linf":
      self.out_projection = LinfLipschitzDense(out_dim=x.shape[-1])
    elif self.norm == "l2":
      self.out_projection = L2LipschitzDense(out_dim=x.shape[-1], sn_iters=self.sn_iters, sn_scale=self.sn_scale)
    else:
      assert 0, "Invalid norm"
    gx = self.out_projection(gx, params=self.out_proj_params, rng_key=out_proj_key, **kwargs)

    return gx

class LipschitzConvResBlock():
  def __init__(self,
               filter_shape,
               hidden_channel,
               n_layers,
               dropout_prob,
               preactivation=True,
               norm="l2",
               nonlinearity="lipswish",
               scaled_ws=False,
               sn_iters=3,
               sn_scale=0.9):
    self.filter_shape   = filter_shape
    self.hidden_channel = hidden_channel

    assert nonlinearity in ["lipswish", "square_lipswish"]
    self.nonlinearity = util.str_to_nonlinearity(nonlinearity)
    self.n_layers       = n_layers
    self.dropout_prob   = dropout_prob
    self.preactivation = preactivation
    self.norm = norm
    self.scaled_ws = scaled_ws
    self.sn_iters = sn_iters
    self.sn_scale = sn_scale

  def get_params(self):
    return dict(res_block=self.res_params,
                in_proj=self.in_projection.get_params(),
                out_proj=self.out_projection.get_params())

  def __call__(self, x, params=None, aux=None, rng_key=None, is_training=True, update_params=True, **kwargs):

    if params is None:
      self.res_params, self.in_proj_params, self.out_proj_params = [None]*3
    else:
      self.res_params, self.in_proj_params, self.out_proj_params = params["res_block"], params["in_proj"], params["out_proj"]

    in_proj_key, out_proj_key, *keys = random.split(rng_key, self.n_layers + 2)
    keys = jnp.array(keys)

    gx = x

    if self.preactivation:
      gx = self.nonlinearity(gx)

    # Apply the input projection
    self.in_projection = LipschitzConvBlock(filter_shape=(3, 3),
                                            out_channel=self.hidden_channel,
                                            dropout_prob=self.dropout_prob,
                                            norm=self.norm,
                                            nonlinearity=self.nonlinearity,
                                            scaled_ws=self.scaled_ws,
                                            sn_iters=self.sn_iters,
                                            sn_scale=self.sn_scale)
    gx = self.in_projection(gx, aux=aux, params=self.in_proj_params, rng_key=in_proj_key, is_training=is_training, **kwargs)

    def scan_block(carry, inputs):
      x = carry
      key, params = inputs
      transform = LipschitzConvBlock(filter_shape=self.filter_shape,
                                     out_channel=self.hidden_channel,
                                     dropout_prob=self.dropout_prob,
                                     norm=self.norm,
                                     nonlinearity=self.nonlinearity,
                                     scaled_ws=self.scaled_ws,
                                     sn_iters=self.sn_iters,
                                     sn_scale=self.sn_scale)
      x = transform(x, aux=aux, params=params, rng_key=key, is_training=is_training, **kwargs)
      return x, transform.get_params()

    if self.res_params is None:
      if self.n_layers > 0:
        init_params = []
        for i, key in enumerate(keys):
          gx, block_params = scan_block(gx, (key, None))
          init_params.append(block_params)
        self.res_params = jax.tree_util.tree_map(lambda *xs: jnp.array(xs), *init_params)
      else:
        self.res_params = ()
    else:
      if self.n_layers > 0:
        gx, res_params = jax.lax.scan(scan_block, gx, (keys, self.res_params), unroll=1)
        if update_params:
          self.res_params = res_params
      else:
        self.res_params = ()

    # Apply the output projection
    if self.norm == "linf":
      self.out_projection = LinfLipschitzConv(filter_shape=(3, 3),
                                              out_channel=x.shape[-1])
    elif self.norm == "l2":
      self.out_projection = L2LipschitzConv(filter_shape=(3, 3),
                                            out_channel=x.shape[-1],
                                            scaled_ws=self.scaled_ws,
                                            sn_iters=self.sn_iters,
                                            sn_scale=self.sn_scale)
    else:
      assert 0, "Invalid norm"
    gx = self.out_projection(gx, params=self.out_proj_params, rng_key=out_proj_key, **kwargs)

    return gx

################################################################################################################

if __name__ == "__main__":
  from debug import *

  rng_key = random.PRNGKey(1)
  x_shape = (7, 4, 4, 3)
  # x_shape = (16, 3)
  x, aux = random.normal(rng_key, (2,)+x_shape)

  filter_shape    = (3, 3)
  hidden_channel  = 16
  dropout_prob    = 0.2
  n_layers        = 4
  net = LipschitzConvResBlock(filter_shape,
                              hidden_channel,
                              n_layers,
                              dropout_prob)

  # net = LipschitzDenseResBlock(hidden_channel,
  #                              n_layers,
  #                              dropout_prob)

  z = net(x, aux=aux, rng_key=rng_key, is_training=False)
  params = net.get_params()


  def apply_fun(params, x):
    x = net(x, params=params, rng_key=rng_key)
    return x

  z2 = net(x, aux=aux, params=params, rng_key=rng_key, is_training=False)
  z3 = net(x[:4], aux=aux[:4], params=params, rng_key=rng_key, is_training=False)

  param_diff = jax.tree_util.tree_map(lambda x,y: jnp.linalg.norm(x-y), params, net.get_params())

  import pdb; pdb.set_trace()
