from abc import ABC, abstractmethod
import jax
from jax import random
import jax.numpy as jnp
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
from collections import OrderedDict
import nux.util as util
import einops

__all__ = ["Flow",
           "Sequential",
           "Invert",
           "NoOp",
           "Identity",
           "Repeat",
           "AsFlat"]

################################################################################################################

class Flow(ABC):

  def get_param_dim(self, dim):
    raise NotImplementedError

  def get_params(self):
    raise NotImplementedError

  @property
  def coupling_param_keys(self):
    raise NotImplementedError

  def extract_coupling_params(self, theta):
    raise NotImplementedError

  @abstractmethod
  def get_params(self):
    pass

  @abstractmethod
  def __call__(self, x, params=None, aux=None, inverse=False, **kwargs):
    pass

################################################################################################################

class Sequential():
  def __init__(self, layers):
    self.layers = layers

  def get_params(self):
    return [layer.get_params() for layer in self.layers]

  def __call__(self, x, params=None, aux=None, rng_key=None, is_training=True, inverse=False, **kwargs):
    n_layers = len(self.layers)

    if params is None:
      self.params = [None]*n_layers
    else:
      assert len(params) == n_layers
      self.params = params

    keys = [None]*n_layers if rng_key is None else random.split(rng_key, n_layers)
    iterable = list(zip(keys, self.params, self.layers))

    if inverse:
      iterable = iterable[::-1]

    log_det = 0.0
    for i, (key, p, layer) in enumerate(iterable):
      x, llc = layer(x, aux=aux, params=p, rng_key=key, is_training=is_training, inverse=inverse, **kwargs)
      log_det += llc

    return x, log_det

################################################################################################################

class Invert():
  def __init__(self, flow):
    self.flow = flow

  def get_params(self):
    return self.flow.get_params()

  def __call__(self, *args, inverse=False, **kwargs):
    z, log_det = self.flow(*args, inverse=not inverse, **kwargs)
    return z, -log_det

################################################################################################################

class NoOp():
  def __init__(self):
    pass

  def get_params(self):
    return {}

  def __call__(self, x, *args, **kwargs):
    return x

################################################################################################################

class Identity():
  def __init__(self):
    pass

  def get_params(self):
    return {}

  def __call__(self, x, *args, **kwargs):
    return x, jnp.zeros(x.shape[:1])

################################################################################################################
import copy
class Repeat():

  def __init__(self, flow, n_repeats, checkerboard=False, unroll=1):
    self.make_flow = lambda : copy.deepcopy(flow)
    self.flow = flow
    self.n_repeats = n_repeats
    self.checkerboard = checkerboard
    self.unroll = unroll

  def get_params(self):
    return dict(repeated=self.repeated_params)

  def __call__(self, x, params=None, aux=None, rng_key=None, is_training=True, inverse=False, **kwargs):
    if params is None:
      self.repeated_params = None
    else:
      self.repeated_params = params["repeated"]

    if self.checkerboard:
      x = einops.rearrange(x, "b h (w k) c -> b h w (k c)", k=2)

    def scan_block(carry, inputs):
      x = carry
      key, params = inputs
      z, log_det = self.flow(x, params=params, aux=aux, rng_key=key, is_training=is_training, inverse=inverse, **kwargs)
      return z, (log_det, self.flow.get_params())

    keys = random.split(rng_key, self.n_repeats)
    if inverse:
      keys = keys[::-1]

    if self.repeated_params is None:
      init_params = []
      log_det = 0.0
      for i, key in enumerate(keys):
        in_shape = x.shape
        x, (_log_det, block_params) = scan_block(x, (key, None))
        assert x.shape == in_shape
        log_det += _log_det
        init_params.append(block_params)
      self.repeated_params = jax.tree_multimap(lambda *xs: jnp.array(xs), *init_params)
    else:
      if self.unroll == -1:
        init_params = []
        log_det = 0.0
        for i, key in enumerate(keys):
          if inverse:
            k = self.n_repeats - i - 1
          else:
            k = i
          in_shape = x.shape
          x, (_log_det, block_params) = scan_block(x, (key, jax.tree_map(lambda x: x[k], self.repeated_params)))
          assert x.shape == in_shape
          log_det += _log_det
          init_params.append(block_params)
        self.repeated_params = jax.tree_multimap(lambda *xs: jnp.array(xs), *init_params)
      else:
        # There is a leaked tracer here because self.flow.get_params() will contain the traced values!
        # This won't affect anything though because we never need those values.
        # TODO: clean up unused variables.
        x, (log_dets, self.repeated_params) = jax.lax.scan(scan_block, x, (keys, self.repeated_params), unroll=self.unroll, reverse=inverse)
        log_det = log_dets.sum(axis=0)

    if self.checkerboard:
      x = einops.rearrange(x, "b h w (k c) -> b h (w k) c", k=2)

    return x, log_det

################################################################################################################

class AsFlat():
  def __init__(self, flow):
    self.flow = flow

  def get_params(self):
    return self.flow.get_params()

  def __call__(self, x, *args, **kwargs):
    x_flat = x.reshape(x.shape[:1] + (-1,))
    z_flat, llc = self.flow(x_flat, *args, **kwargs)
    z = z_flat.reshape(x.shape)
    return z, llc
