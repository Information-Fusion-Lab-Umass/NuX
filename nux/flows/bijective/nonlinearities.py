import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
from nux.flows.base import Sequential, Invert
import nux.util as util
import einops
from nux.flows.base import Flow

__all__ = ["Softplus",
           "LeakyReLU",
           "SneakyReLU",
           "SquarePlus",
           "SquareSigmoid",
           "SquareLogit",
           "Sigmoid",
           "Logit",
           "SLog",
           "CartesianToSpherical",
           "SphericalToCartesian",
           "SafeCartesianToSpherical",
           "GaussianCDF",
           "LogisticCDF"]

class Softplus(Flow):
  def __init__(self):
    """
    """
    pass

  def get_params(self):
    return {}

  def __call__(self, x, inverse=False, **kwargs):
    x_shape = x.shape[1:]
    sum_axes = util.last_axes(x_shape)

    if inverse == True:
      x = jnp.where(x < 0.0, 1e-5, x)
      dx = jnp.log1p(-jnp.exp(-x))
      z = x + dx
      log_det = dx.sum(axis=sum_axes)
    else:
      z = jax.nn.softplus(x)
      log_det = jnp.log1p(-jnp.exp(-z)).sum(axis=sum_axes)

    return z, log_det

class LeakyReLU(Flow):

  def __init__(self, alpha: float=0.01):
    """ Leaky relu
    """
    self.alpha = alpha

  def get_params(self):
    return {}

  def __call__(self, x, inverse=False, **kwargs):

    if inverse == False:
      z = jnp.where(x > 0, x, self.alpha*x)
    else:
      z = jnp.where(x > 0, x, x/self.alpha)

    x_shape = x.shape[1:]
    sum_axes = util.last_axes(x_shape)

    log_dx_dz = jnp.where(x > 0, 0, jnp.log(self.alpha))
    log_det = log_dx_dz.sum(axis=sum_axes)

    return z, log_det

class SneakyReLU(Flow):

  def __init__(self, alpha: float=0.1):
    """ Originally from https://invertibleworkshop.github.io/INNF_2019/accepted_papers/pdfs/INNF_2019_paper_26.pdf
    """
    # Sneaky ReLU uses a different convention
    self.alpha = (1.0 - alpha)/(1.0 + alpha)

  def get_params(self):
    return {}

  def __call__(self, x, inverse=False, **kwargs):
    if inverse == False:
      sqrt_1px2 = jnp.sqrt(1 + x**2)
      z = (x + self.alpha*(sqrt_1px2 - 1))/(1 + self.alpha)
      log_det = jnp.log(1 + self.alpha*x/sqrt_1px2) - jnp.log(1 + self.alpha)
    else:
      alpha_sq = self.alpha**2
      b = (1 + self.alpha)*x + self.alpha
      z = (jnp.sqrt(alpha_sq*(1 + b**2 - alpha_sq)) - b)/(alpha_sq - 1)
      sqrt_1px2 = jnp.sqrt(1 + z**2)
      log_det = jnp.log(1 + self.alpha*z/sqrt_1px2) - jnp.log(1 + self.alpha)

    # Always assume that x is batched!!
    x_shape = x.shape[1:]
    sum_axes = util.last_axes(x_shape)

    log_det = log_det.sum(axis=sum_axes)
    return z, log_det

class SquarePlus(Flow):

  def __init__(self, gamma: float=0.5):
    """
    """
    self.gamma = gamma

  def get_params(self):
    return {}

  def __call__(self, x, inverse=False, **kwargs):
    x_shape = x.shape[1:]
    sum_axes = util.last_axes(x_shape)

    if inverse == False:
      sqrt_arg = x**2 + 4*self.gamma
      z = 0.5*(x + jnp.sqrt(sqrt_arg))
      z = jnp.maximum(z, 0.0)
      dzdx = 0.5*(1 + x*jax.lax.rsqrt(sqrt_arg)) # Always positive
      dzdx = jnp.maximum(dzdx, 1e-5)
    else:
      z = x - self.gamma/x
      dzdx = 0.5*(1 + z*jax.lax.rsqrt(z**2 + 4*self.gamma))

    log_det = jnp.log(dzdx).sum(axis=sum_axes)

    return z, log_det

class SquareSigmoid(Flow):

  def __init__(self, gamma: float=0.5):
    """
    """
    self.gamma = gamma

  def get_params(self):
    return {}

  def __call__(self, x, inverse=False, **kwargs):
    x_shape = x.shape[1:]
    sum_axes = util.last_axes(x_shape)

    if inverse == False:
      rsqrt = jax.lax.rsqrt(x**2 + 4*self.gamma)
      z = 0.5*(1 + x*rsqrt)
    else:
      arg = 2*x - 1
      z = 2*jnp.sqrt(self.gamma)*arg*jax.lax.rsqrt(1 - arg**2)
      rsqrt = jax.lax.rsqrt(z**2 + 4*self.gamma)

    dzdx = 2*self.gamma*rsqrt**3
    log_det = jnp.log(dzdx).sum(axis=sum_axes)

    return z, log_det

class SquareLogit(SquareSigmoid):

  def __call__(self, x, inverse=False, **kwargs):
    z, log_det = super().__call__(x, inverse=not inverse, **kwargs)
    return z, -log_det

class Sigmoid(Flow):

  def __init__(self, scale: Optional[float]=None):
    """ Sigmoid function.  Transforms data to [scale/2, 1 - scale/2]
    """
    self.scale = scale

  def get_params(self):
    return {}

  def __call__(self, x, inverse=False, scale=None, **kwargs):
    x_shape = x.shape[1:]
    sum_axes = util.last_axes(x_shape)

    if scale is None:
      scale = self.scale

    if inverse == False:
      z = jax.nn.sigmoid(x)

      if not scale is None:
        z -= scale
        z /= 1.0 - 2*scale

      log_det = -(jax.nn.softplus(x) + jax.nn.softplus(-x))
    else:
      if not scale is None:
        x *= 1.0 - 2*scale
        x += scale

      z = jax.scipy.special.logit(x)
      log_det = -(jax.nn.softplus(z) + jax.nn.softplus(-z))

    if not scale is None:
      log_det -= jnp.log(1.0 - 2*scale)

    log_det = log_det.sum(axis=sum_axes)

    return z, log_det

class Logit(Flow):

  def __init__(self, scale: Optional[float]=0.05, force_values: bool=False):
    """ Logit function.  Transforms from [scale/2, 1 - scale/2] to the reals.
    """
    self.scale = scale
    self.has_scale = scale is not None
    self.force_values = force_values

  def get_params(self):
    return {}

  def __call__(self, x, inverse=False, force_values=False, **kwargs):
    x_shape = x.shape[1:]
    sum_axes = util.last_axes(x_shape)

    if inverse == False:
      if self.has_scale == True:
        x *= (1.0 - 2*self.scale)
        x += self.scale
      z = jax.scipy.special.logit(x)
      log_det = (jax.nn.softplus(z) + jax.nn.softplus(-z))
    else:
      z = jax.nn.sigmoid(x)
      log_det = (jax.nn.softplus(x) + jax.nn.softplus(-x))

      if self.has_scale == True and force_values == False and self.force_values == False:
        z -= self.scale
        z /= (1.0 - 2*self.scale)

    if self.has_scale == True:
      log_det += jnp.log(1.0 - 2*self.scale)

    log_det = log_det.sum(axis=sum_axes)

    return z, log_det

class SLog(Flow):

  def __init__(self, alpha=None):
    """ https://papers.nips.cc/paper/2019/file/b1f62fa99de9f27a048344d55c5ef7a6-Paper.pdf
    """
    self.constant_alpha = alpha

  @property
  def param_dim(self):
    return 1

  def get_params(self):
    if self.constant_alpha:
      return ()
    return dict(alpha=self.alpha)

  def __call__(self, x, params=None, rng_key=None, inverse=False, **kwargs):
    if self.constant_alpha is None:
      if params is None:
        self.alpha = jnp.zeros(x.shape[1:])
      else:
        self.alpha = params["alpha"]
      alpha = util.square_plus(self.alpha) + 1e-4
    else:
      self.alpha = ()
      alpha = self.constant_alpha

    if inverse == False:
      log_det = jnp.log1p(alpha*jnp.abs(x))
      z = jnp.sign(x)/alpha*log_det
    else:
      z = jnp.sign(x)/alpha*(jnp.exp(alpha*jnp.abs(x)) - 1)
      log_det = jnp.log1p(alpha*jnp.abs(z))

    x_shape = x.shape[1:]
    sum_axes = util.last_axes(x_shape)

    log_det = -log_det.sum(axis=sum_axes)
    return z, log_det

class CartesianToSpherical(Flow):
  # This will probably fail if x is close to 0 or 2pi!
  # Use the safe versions instead

  def __init__(self):
    pass

  def get_params(self):
    return {}

  def forward(self, x, eps=1e-5):
    r = jnp.linalg.norm(x, axis=-1, keepdims=True)
    denominators = jnp.sqrt(jnp.cumsum(x[...,::-1]**2, axis=-1)[...,::-1])[...,:-1]
    cos_phi = x[...,:-1]/denominators
    cos_phi = jnp.maximum(-1.0 + eps, cos_phi)
    cos_phi = jnp.minimum(1.0 - eps, cos_phi)
    phi = jnp.arccos(cos_phi)

    last_value = jnp.where(x[...,-1] >= 0, phi[...,-1], 2*jnp.pi - phi[...,-1])
    phi = phi.at[...,-1].set(last_value)

    return jnp.concatenate([r, phi], axis=-1)

  def inverse(self, x):
    r = x[...,:1]
    phi = x[...,1:]
    sin_prod = jnp.cumprod(jnp.sin(phi), axis=-1)
    first_part = jnp.concatenate([jnp.ones(r.shape), sin_prod], axis=-1)
    second_part = jnp.concatenate([jnp.cos(phi), jnp.ones(r.shape)], axis=-1)
    return r*first_part*second_part

  def __call__(self, x, inverse=False, **kwargs):
    if inverse == False:
      z = self.forward(x)
      r, phi = z[...,0], z[...,1:]
    else:
      z = self.inverse(x)
      r, phi = x[...,0], x[...,1:]

    x_shape = x.shape[1:]

    n = x_shape[-1]
    n_range = jnp.arange(n - 2, -1, -1)
    log_abs_sin_phi = jnp.log(jnp.abs(jnp.sin(phi)))
    log_det = -(n - 1)*jnp.log(r) - jnp.sum(n_range*log_abs_sin_phi, axis=-1)
    log_det = log_det.sum(axis=util.last_axes(x.shape[1:-1]))
    return z, log_det

class SphericalToCartesian(CartesianToSpherical):
  def __call__(self, x, inverse=False, **kwargs):
    z, log_det = super().__call__(x, inverse=not inverse, **kwargs)
    return z, -log_det

class SafeCartesianToSpherical(CartesianToSpherical):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # Pass r through the inverse of square plus
    self.r_scale = Invert(SquarePlus())

    # Place phi between 0 and 1 and apply a logit
    from nux.flows.bijective.affine import StaticScale
    self.phi_scale = Sequential([StaticScale(2*jnp.pi), SquareLogit()])

  def __call__(self, x, *args, inverse=False, **kwargs):
    if inverse == False:
      z, log_det = super().__call__(x, *args, inverse=False, **kwargs)
      r, phi = z[...,:1], z[...,1:]
      r, log_det1 = self.r_scale(r, inverse=False)
      phi, log_det2 = self.phi_scale(phi, inverse=False)
      z = jnp.concatenate([r, phi], axis=-1)
    else:
      r, phi = x[...,:1], x[...,1:]
      phi, log_det2 = self.phi_scale(phi, inverse=True)
      r, log_det1 = self.r_scale(r, inverse=True)
      z = jnp.concatenate([r, phi], axis=-1)
      z, log_det = super().__call__(z, *args, inverse=True, **kwargs)

    log_det = log_det + log_det1 + log_det2
    return z, log_det

from nux.priors.gaussian import UnitGaussianPrior
class GaussianCDF(Flow):
  # Gaussian -> Uniform

  def __init__(self):
    pass

  def get_params(self):
    return ()

  def __call__(self, x, params=None, rng_key=None, inverse=False, **kwargs):

    if inverse == False:
      z = jax.scipy.stats.norm.cdf(x)
      _, log_det = UnitGaussianPrior()(x)
    else:
      z = jax.scipy.stats.norm.ppf(x)
      _, log_det = UnitGaussianPrior()(z)

    return z, log_det

from nux.priors.other import LogisticPrior
class LogisticCDF(Flow):
  # Logistic -> Uniform

  def __init__(self):
    pass

  def get_params(self):
    return ()

  def __call__(self, x, params=None, rng_key=None, inverse=False, **kwargs):

    if inverse == False:
      z = jax.scipy.stats.logistic.cdf(x)
      _, log_det = LogisticPrior()(x)
    else:
      z = jax.scipy.stats.logistic.ppf(x)
      _, log_det = LogisticPrior()(z)

    return z, log_det


if __name__ == "__main__":
  from debug import *
  import matplotlib.pyplot as plt
  from jax.flatten_util import ravel_pytree

  H, W, C = 8, 8, 8
  rng_key = random.PRNGKey(0)
  # x = random.normal(rng_key, (1, H, W, C))*2
  x = random.normal(rng_key, (10000, 4))
  # x = x/jnp.linalg.norm(x, axis=-1)[:,None]

  logit = SafeCartesianToSpherical()

  z, log_det = logit(x, inverse=False)
  out, _ = logit(z, inverse=True)

  z2 = random.normal(rng_key, (10000, 4))*1000
  out2, _ = logit(z2, inverse=True)

  import pdb; pdb.set_trace()

  def jac(x):
    x_flat, unflatten = ravel_pytree(x)
    def flat_apply(x_flat):
      return logit(unflatten(x_flat)[None], inverse=False)[0].ravel()
    return jax.jacobian(flat_apply)(x_flat)

  Js = jax.vmap(jac)(x)

  import pdb; pdb.set_trace()

