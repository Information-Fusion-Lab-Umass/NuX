import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import nux.util as util
import einops

__all__ = ["WeightNormDense",
           "GatedDense",
           "WeightNormConv",
           "LayerNorm",
           "GatedConv"]

class WeightNormDense():

  def __init__(self, out_dim, positive=False, before_square_plus=False):
    self.dim_out = out_dim
    self.positive = positive
    self.before_square_plus = before_square_plus

  def get_params(self):
    return dict(w=self.w, g=self.g, b=self.b)

  def __call__(self, x, params=None, rng_key=None):
    dim_in = x.shape[-1]

    if params is None:
      self.w = random.normal(rng_key, shape=(self.dim_out, dim_in))*0.05
    else:
      self.w, self.g, self.b = params["w"], params["g"], params["b"]

    w = self.w*jax.lax.rsqrt((self.w**2).sum(axis=1))[:,None]
    if self.positive:
      w = util.square_plus(w)
    x = jnp.einsum("ij,bj->bi", w, x)
    if self.positive:
      x /= w.shape[-1]

    if params is None:
      if x.shape[0] == 1:
        std = 1.0
      else:
        std = jnp.std(x.reshape((-1, x.shape[-1])), axis=0) + 1e-5

      if self.before_square_plus:
        std = std - 1/std

      self.g = 1/std
    g = self.g
    if self.positive:
      g = util.square_plus(g)
    x *= g

    if params is None:
      mean = jnp.mean(x.reshape((-1, x.shape[-1])), axis=0)
      self.b = -mean
    x += self.b

    return x

class GatedDense():
  def __init__(self, hidden_dim, nonlinearity, dropout_prob):
    self.hidden_dim   = hidden_dim
    self.nonlinearity = nonlinearity
    self.dropout_prob = dropout_prob
    self.wn_aux       = None

  def get_params(self):
    params = dict(wn_1=self.wn_1.get_params(),
                  wn_2=self.wn_2.get_params())
    if self.wn_aux is not None:
      params["wn_aux"] = self.wn_aux.get_params()
    return params

  def __call__(self, x, params=None, aux=None, rng_key=None, is_training=True):
    k1, k2, k3, k4 = random.split(rng_key, 4)

    if params is None:
      self.wn_1_params, self.wn_aux_params, self.wn_2_params = [None]*3
    else:
      self.wn_1_params, self.wn_2_params = params["wn_1"], params["wn_2"]
      if aux is not None:
        self.wn_aux_params = params["wn_aux"]

    gx = self.nonlinearity(x)

    # Conv
    self.wn_1 = WeightNormDense(out_dim=self.hidden_dim)
    gx = self.wn_1(gx, params=self.wn_1_params, rng_key=k1)

    # Auxiliary input
    if aux is not None:
      aux = self.nonlinearity(aux)

      self.wn_aux = WeightNormDense(out_dim=self.hidden_dim)
      gx += self.wn_aux(aux, params=self.wn_aux_params, rng_key=k2)

    gx = self.nonlinearity(gx)

    # Dropout
    if is_training == True and self.dropout_prob > 0:
      keep_rate = 1.0 - self.dropout_prob
      mask = jax.random.bernoulli(k3, keep_rate, shape=gx.shape)
      gx = mask*gx/keep_rate

    # Conv
    self.wn_2 = WeightNormDense(out_dim=2*x.shape[-1])
    gx = self.wn_2(gx, params=self.wn_2_params, rng_key=k4)

    a, b = jnp.split(gx, 2, axis=-1)
    gx = a*util.square_sigmoid(b)
    return gx

################################################################################################################

class WeightNormConv():

  def __init__(self, filter_shape, out_channel):
    self.filter_shape = filter_shape
    self.C_out = out_channel

  def get_params(self):
    return dict(w=self.w, g=self.g, b=self.b)

  def __call__(self, x, params=None, rng_key=None):
    C_in = x.shape[-1]

    if params is None:
      self.w = random.normal(rng_key, shape=self.filter_shape + (C_in, self.C_out))*0.05
    else:
      self.w, self.g, self.b = params["w"], params["g"], params["b"]

    w = self.w*jax.lax.rsqrt((self.w**2).sum(axis=(0, 1, 2)))[None,None,None,:]
    x = util.conv(w, x)

    if params is None:
      std = jnp.std(x.reshape((-1, x.shape[-1])), axis=0) + 1e-5
      self.g = 1/std
    x *= self.g

    if params is None:
      mean = jnp.mean(x.reshape((-1, x.shape[-1])), axis=0) + 1e-5
      self.b = -mean
    x += self.b

    return x

class LayerNorm():

  def __init__(self):
    pass

  def get_params(self):
    return dict(gamma=self.gamma, beta=self.beta)

  def __call__(self, x, params=None):
    if params is not None:
      self.beta, self.gamma = params["beta"], params["gamma"]
    else:
      self.beta = jnp.zeros(x.shape[-1:])
      self.gamma = jnp.ones(x.shape[-1:])

    x_spatial_flat = einops.rearrange(x, "b h w c -> b (h w) c")
    mean, inv_std = util.mean_and_inverse_std(x_spatial_flat, axis=1)
    mean, inv_std = mean[:,None,None,:], inv_std[:,None,None,:]

    y = (x - mean)*inv_std*self.gamma + self.beta
    return y

class GatedConv():
  def __init__(self, filter_shape, hidden_channel, nonlinearity, dropout_prob):
    self.filter_shape   = filter_shape
    self.hidden_channel = hidden_channel
    self.nonlinearity   = nonlinearity
    self.dropout_prob   = dropout_prob
    self.wn_aux = None

  def get_params(self):
    params = dict(wn_1=self.wn_1.get_params(),
                  wn_2=self.wn_2.get_params())
    if self.wn_aux is not None:
      params["wn_aux"] = self.wn_aux.get_params()
    return params

  def __call__(self, x, params=None, aux=None, rng_key=None, is_training=True):
    k1, k2, k3, k4 = random.split(rng_key, 4)

    if params is None:
      self.wn_1_params, self.wn_aux_params, self.wn_2_params = [None]*3
    else:
      self.wn_1_params, self.wn_2_params = params["wn_1"], params["wn_2"]
      if aux is not None:
        self.wn_aux_params = params["wn_aux"]

    gx = self.nonlinearity(x)

    # Conv
    self.wn_1 = WeightNormConv(filter_shape=self.filter_shape, out_channel=self.hidden_channel)
    gx = self.wn_1(gx, params=self.wn_1_params, rng_key=k1)

    # Auxiliary input
    if aux is not None:
      aux = self.nonlinearity(aux)

      self.wn_aux = WeightNormConv(filter_shape=(1, 1), out_channel=self.hidden_channel)
      gx += self.wn_aux(aux, params=self.wn_aux_params, rng_key=k2)

    gx = self.nonlinearity(gx)

    # Dropout
    if is_training == True and self.dropout_prob > 0:
      keep_rate = 1.0 - self.dropout_prob
      mask = jax.random.bernoulli(k3, keep_rate, shape=gx.shape)
      gx = mask*gx/keep_rate

    # Conv
    self.wn_2 = WeightNormConv(filter_shape=(1, 1), out_channel=2*x.shape[-1])
    gx = self.wn_2(gx, params=self.wn_2_params, rng_key=k4)

    a, b = jnp.split(gx, 2, axis=-1)
    gx = a*util.square_sigmoid(b)
    return gx