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
           "AsFlat",
           "Flatten",
           "Vmap",
           "ZeroInitWrapper"]

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
  def __init__(self, layers, prior=None, is_transform=False):
    if prior is not None:
      if isinstance(layers, tuple):
        layers = layers + (prior,)
      elif isinstance(layers, list):
        layers = layers + [prior]
      else:
        layers = [layers, prior]

    self.layers = layers
    self.is_transform = is_transform
    if is_transform == False:
      self.transform = Sequential(self.layers[:-1], is_transform=True)
      self.prior = self.layers[-1]

  def get_params(self):
    return [layer.get_params() for layer in self.layers]

  def __call__(self, x, params=None, aux=None, rng_key=None, is_training=True, inverse=False, **kwargs):
    n_layers = len(self.layers)

    if params is None:
      self.params = [None]*n_layers
    else:
      if self.is_transform and len(params) == n_layers + 1:
        # The last parameter is for the prior
        self.params = params[:-1]
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

  def __init__(self, flow, n_repeats, checkerboard=False, unroll=1, **kwargs):
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
      self.repeated_params = jax.tree_util.tree_map(lambda *xs: jnp.array(xs), *init_params)
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
          x, (_log_det, block_params) = scan_block(x, (key, jax.tree_util.tree_map(lambda x: x[k], self.repeated_params)))
          assert x.shape == in_shape
          log_det += _log_det
          init_params.append(block_params)
        self.repeated_params = jax.tree_util.tree_map(lambda *xs: jnp.array(xs), *init_params)
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

class Flatten():
  def __init__(self):
    pass

  def get_params(self):
    return ()

  def __call__(self, x, inverse=False, **kwargs):
    if inverse == False:
      self.x_shape = x.shape[1:]
      z = x.reshape(x.shape[:1] + (-1,))
    else:
      z = x.reshape(x.shape[:1] + self.x_shape)

    llc = jnp.zeros(x.shape[:1])
    return z, llc

################################################################################################################

class Vmap():
  """ Vectorize a flow """
  def __init__(self, flow):
    self.flow = flow

  def get_params(self):
    return self.params

  def __call__(self, x, params=None, rng_key=None, inverse=False, **kwargs):
    n_vmaps = x.shape[-1]
    keys = random.split(rng_key, n_vmaps)

    if params is None:

      def apply_fun_init(x, rng_key):
        z, log_det = self.flow(x[...,None], params=None, rng_key=rng_key, inverse=inverse, **kwargs)
        params = self.flow.get_params()
        return params

      self.params = jax.vmap(apply_fun_init, in_axes=(-1, 0), out_axes=0)(x, keys)
    else:
      self.params = params

    def apply_fun(x, params, rng_key):
      z, log_det = self.flow(x, params=params, rng_key=rng_key, inverse=inverse, **kwargs)
      return z.squeeze(), log_det

    z, log_dets = jax.vmap(apply_fun, in_axes=(-1, 0, 0), out_axes=-1)(x, self.params, keys)
    log_det = log_dets.sum(axis=-1)
    return z, log_det

################################################################################################################

class ZeroInitWrapper():
  def __init__(self, net):
    self.net = net

  def get_params(self):
    return dict(net=self.net.get_params(),
                scale=self.scale)

  def __call__(self, x, params=None, rng_key=None, **kwargs):
    if params is None:
      self.scale = random.normal(rng_key, (1,))*0.001
      params = dict(scale=self.scale, net=None)
    else:
      self.scale = params["scale"]

    out = self.net(x, params=params["net"], rng_key=rng_key, **kwargs)
    out *= self.scale
    return out

################################################################################################################

if __name__ == "__main__":
  from debug import *
  import nux
  from nux.tests.basic_unit_test import exact_test

  rng_key = random.PRNGKey(0)
  x = random.normal(rng_key, shape=(5, 4))
  flow = Vmap(nux.LogisticCDFMixtureLogit(K=8))

  # Initialize the flow
  flow(x, rng_key=rng_key)
  params = flow.get_params()

  # Scramble the parameters to undo the data dependent init
  flat_params, unflatten = jax.flatten_util.ravel_pytree(params)
  flat_params = random.normal(rng_key, flat_params.shape)
  params = unflatten(flat_params)

  # Compute the log likelihood contribution of flow
  z, log_det = flow(x, params=params, rng_key=rng_key)

  # Reconstruct x
  x_reconstr, log_det2 = flow(z, params=params, rng_key=rng_key, inverse=True)
  assert jnp.allclose(x, x_reconstr)
  assert jnp.allclose(log_det, log_det2)

  # Compute the exact jacobian
  def unbatched_apply_fun(x):
    z, _ = flow(x[None], params=params, rng_key=rng_key)
    return z[0]

  J = jax.vmap(jax.jacobian(unbatched_apply_fun))(x)
  total_dim = util.list_prod(x.shape[1:])
  J_flat = J.reshape((-1, total_dim, total_dim))
  log_det_exact = jnp.linalg.slogdet(J_flat)[1]

  assert jnp.allclose(log_det_exact, log_det)
  print(f"{str(flow)} passed the reconstruction and log det test")

  import pdb; pdb.set_trace()