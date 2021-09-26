import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import nux.util as util
import einops
from nux.flows.base import Flow

__all__ = ["CenterAndScale",
           "DiscreteBias",
           "Bias",
           "StaticScale",
           "Scale",
           "ShiftScale",
           "ShiftScalarScale",
           "StaticShiftScale",
           "DenseMVP",
           "CaleyOrthogonalMVP",
           "PLUMVP"]

class CenterAndScale(Flow):

  def __init__(self, s):
    assert s < 1.0
    self.s = s
    self.b = (1.0 - s)/2

  def get_params(self):
    return {}

  def __call__(self, x, params=None, inverse=False, **kwargs):
    s = jnp.broadcast_to(self.s, x.shape)
    log_s = jnp.log(s)

    if inverse == False:
      z = x*s + self.b
    else:
      z = (x - self.b)/s

    sum_axes = util.last_axes(x.shape[1:])
    log_det = log_s.sum(axis=sum_axes)
    return z, log_det

class DiscreteBias(Flow):

  def __init__(self):
    self.b = None

  def get_param_dim(self, dim):
    return 1*dim

  def get_params(self):
    return {"b": self.b}

  @property
  def coupling_param_keys(self):
    return ("b",)

  def extract_coupling_params(self, theta):
    return (theta,)

  def __call__(self, x, params=None, inverse=False, **kwargs):
    if params is None:
      reduce_axes = list(range(0, x.ndim - 1))
      self.b = jnp.mean(x, axis=reduce_axes)
    else:
      self.b = params["b"]

    b = util.st_round(self.b)

    if inverse == False:
      z = x - b
    else:
      z = x + b
    return z, jnp.zeros(x.shape[:1])

class Bias(Flow):

  def __init__(self):
    self.b = None

  def get_param_dim(self, dim):
    return 1*dim

  def get_params(self):
    return {"b": self.b}

  @property
  def coupling_param_keys(self):
    return ("b",)

  def extract_coupling_params(self, theta):
    return (theta,)

  def __call__(self, x, params=None, inverse=False, **kwargs):
    if params is None:
      reduce_axes = list(range(0, x.ndim - 1))
      self.b = jnp.mean(x, axis=reduce_axes)
    else:
      self.b = params["b"]

    if inverse == False:
      z = x - self.b
    else:
      z = x + self.b
    return z, jnp.zeros(x.shape[:1])

class StaticScale(Flow):

  def __init__(self, s):
    self.s = s

  def get_params(self):
    return {}

  def __call__(self, x, params=None, inverse=False, **kwargs):
    s = jnp.broadcast_to(self.s, x.shape)
    log_s = jnp.log(s)

    if inverse == False:
      z = x/s
    else:
      z = x*s

    sum_axes = util.last_axes(x.shape[1:])
    log_det = -log_s.sum(axis=sum_axes)
    return z, log_det

class Scale(Flow):

  def __init__(self):
    self.s = None

  def get_param_dim(self, dim):
    return 1*dim

  def get_params(self):
    return {"s": self.s}

  @property
  def coupling_param_keys(self):
    return ("s",)

  def extract_coupling_params(self, theta):
    return (theta,)

  def __call__(self, x, params=None, inverse=False, **kwargs):
    if params is None:
      reduce_axes = list(range(0, x.ndim - 1))
      std = jnp.std(x, axis=reduce_axes)
      self.s = std - 1/std
    else:
      self.s = params["s"]

    s = util.square_plus(self.s, gamma=1.0) + 1e-4
    s = jnp.broadcast_to(s, x.shape)
    log_s = jnp.log(s)

    if inverse == False:
      z = x/s
    else:
      z = x*s

    sum_axes = util.last_axes(x.shape[1:])
    log_det = -log_s.sum(axis=sum_axes)
    return z, log_det

class ShiftScale(Flow):

  def __init__(self, center_init=True):
    """ Elementwise shift + scale.  This is RealNVP https://arxiv.org/pdf/1605.08803.pdf
    """
    self.s = None
    self.b = None
    self.center_init = center_init

  def get_param_dim(self, dim):
    return 2*dim

  def get_params(self):
    return {"s": self.s, "b": self.b}

  @property
  def coupling_param_keys(self):
    return ("s", "b")

  def extract_coupling_params(self, theta):
    return jnp.split(theta, 2, axis=-1)

  def __call__(self, x, params=None, rng_key=None, inverse=False, **kwargs):
    if params is None:
      if self.center_init:
        reduce_axes = list(range(0, x.ndim - 1))
        mean, std = util.mean_and_std(x, axis=reduce_axes)
        std += 1e-5
        self.b = mean
        self.s = std - 1/std
      else:
        self.b, self.s = random.normal(rng_key, (2, *x.shape[1:]))
    else:
      self.b, self.s = params["b"], params["s"]

    s = util.square_plus(self.s, gamma=1.0) + 1e-4
    s = jnp.broadcast_to(s, x.shape)
    log_s = jnp.log(s)

    if inverse == False:
      z = (x - self.b)/s
    else:
      z = x*s + self.b

    sum_axes = util.last_axes(x.shape[1:])
    log_det = -log_s.sum(axis=sum_axes)
    return z, log_det

class ShiftScalarScale(Flow):

  def __init__(self, unit_norm=False):
    """ Elementwise shift + scalar scale
    """
    self.s = None
    self.b = None
    self.unit_norm = unit_norm

  def get_params(self):
    return {"s": self.s, "b": self.b}

  def __call__(self, x, params=None, rng_key=None, inverse=False, **kwargs):
    if params is None:
      reduce_axes = list(range(0, x.ndim - 1))
      mean = jnp.mean(x, axis=reduce_axes)
      self.b = mean
      if self.unit_norm == False:
        std = jnp.std(x) + 1e-5
        self.s = std - 1/std
      else:
        x_norm = jnp.sum(x**2, axis=util.last_axes(x.shape[1:]))
        x_norm = jnp.sqrt(x_norm).mean()
        self.s = x_norm - 1/x_norm

    else:
      self.b, self.s = params["b"], params["s"]

    s = util.square_plus(self.s, gamma=1.0) + 1e-4
    s = jnp.broadcast_to(s, x.shape)
    log_s = jnp.log(s)

    if inverse == False:
      z = (x - self.b)/s
    else:
      z = x*s + self.b

    sum_axes = util.last_axes(x.shape[1:])
    log_det = -log_s.sum(axis=sum_axes)
    return z, log_det

class StaticShiftScale(Flow):

  def __init__(self, s, b):
    self.s = s
    self.b = b

  def get_params(self):
    return {}

  def __call__(self, x, params=None, inverse=False, **kwargs):
    s = jnp.broadcast_to(self.s, x.shape)
    log_s = jnp.log(s)

    if inverse == False:
      z = (x - self.b)/s
    else:
      z = x*s + self.b

    sum_axes = util.last_axes(x.shape[1:])
    log_det = -log_s.sum(axis=sum_axes)
    return z, log_det

################################################################################################################

class DenseMVP(Flow):

  def __init__(self):
    """ Dense
    """
    pass

  def get_params(self):
    return {"A": self.A}

  def __call__(self, x, params=None, inverse=False, rng_key=None, **kwargs):
    x_shape = x.shape[1:]
    dim = x_shape[-1]

    if params is None:
      self.A = random.normal(rng_key, shape=(dim, dim))
    else:
      self.A = params["A"]

    if inverse == False:
      z = jnp.einsum("ij,...j->...i", self.A, x)
    else:
      A_inv = jnp.linalg.inv(self.A)
      z = jnp.einsum("ij,...j->...i", A_inv, x)

    log_det = jnp.linalg.slogdet(self.A)[1]*util.list_prod(x.shape[1:-1])
    log_det = log_det*jnp.ones(x.shape[:1])
    return z, log_det

################################################################################################################

class CaleyOrthogonalMVP(Flow):

  def __init__(self):
    """ Dense
    """
    pass

  def get_params(self):
    return {"W": self.W}

  def __call__(self, x, params=None, inverse=False, rng_key=None, **kwargs):
    x_shape = x.shape[1:]
    dim = x_shape[-1]

    if params is None:
      self.W = random.normal(rng_key, shape=(dim, dim))
    else:
      self.W = params["W"]

    A = self.W - self.W.T

    if inverse == False:
      IpA_inv = jnp.linalg.inv(jnp.eye(dim) + A)
      y = jnp.einsum("ij,...j->...i", IpA_inv, x)
      z = y - jnp.einsum("ij,...j->...i", A, y)
    else:
      ImA_inv = jnp.linalg.inv(jnp.eye(dim) - A)
      y = jnp.einsum("ij,...j->...i", ImA_inv, x)
      z = y + jnp.einsum("ij,...j->...i", A, y)

    log_det = jnp.zeros(x.shape[:1])
    return z, log_det

################################################################################################################

tri_solve = jax.scipy.linalg.solve_triangular
L_solve = partial(tri_solve, lower=True, unit_diagonal=True)
U_solve = partial(tri_solve, lower=False, unit_diagonal=True)
U_solve_with_diag = partial(tri_solve, lower=False, unit_diagonal=False)

class PLUMVP(Flow):

  def __init__(self):
    """ Dense layer using the PLU parametrization https://arxiv.org/pdf/1807.03039.pdf
    """
    pass

  def get_params(self):
    return {"A": self.A}

  def __call__(self, x, params=None, inverse=False, rng_key=None, **kwargs):
    x_shape = x.shape[1:]
    dim = x_shape[-1]

    if params is None:
      self.A = random.normal(rng_key, shape=(dim, dim))*0.01
      self.A = self.A.at[jnp.arange(dim),jnp.arange(dim)].set(1.0)
    else:
      self.A = params["A"]

    mask = jnp.ones((dim, dim), dtype=bool)
    upper_mask = jnp.triu(mask)
    lower_mask = jnp.tril(mask, k=-1)

    if inverse == False:
      z = jnp.einsum("ij,...j->...i", self.A*upper_mask, x)
      z = jnp.einsum("ij,...j->...i", self.A*lower_mask, z) + z
    else:
      L_solve_vmap = L_solve
      U_solve_vmap = U_solve_with_diag
      for _ in x.shape[:-1]:
        L_solve_vmap = jax.vmap(L_solve_vmap, in_axes=(None, 0))
        U_solve_vmap = jax.vmap(U_solve_vmap, in_axes=(None, 0))
      z = L_solve_vmap(self.A*lower_mask, x)
      z = U_solve_vmap(self.A*upper_mask, z)

    log_det = jnp.log(jnp.abs(jnp.diag(self.A))).sum()*util.list_prod(x.shape[1:-1])
    log_det = log_det*jnp.ones(x.shape[:1])
    return z, log_det

################################################################################################################

def regular_test():
  from jax.flatten_util import ravel_pytree

  rng_key = random.PRNGKey(0)
  x = random.normal(rng_key, shape=(2, 4))
  x_orig = x

  flow = PLUMVP()
  z, log_det = flow(x, rng_key=rng_key)
  params = flow.get_params()


  reconstr, _ = flow(z, params=params, rng_key=rng_key, inverse=True)


  z2, log_det2 = flow(reconstr, params=params, rng_key=rng_key, inverse=False)
  assert jnp.allclose(x, reconstr)
  assert jnp.allclose(z, z2)

  def jac(x, blah=False):
    flat_x, unflatten = ravel_pytree(x)
    def flat_call(flat_x):
      x = unflatten(flat_x)
      z, _ = flow(x[None], params=params, rng_key=rng_key)
      return z.ravel()
    z = flat_call(flat_x)
    if blah:
      return z
    return jax.jacobian(flat_call)(flat_x)

  jac(x[0], blah=True)
  # import pdb; pdb.set_trace()
  J = jax.vmap(jac)(x)
  true_log_det = jnp.linalg.slogdet(J)[1]
  assert jnp.allclose(log_det, true_log_det)

if __name__ == "__main__":
  from debug import *

  regular_test()
