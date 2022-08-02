import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
from typing import Optional, Mapping, Callable, Sequence
import nux.util as util
from jax.scipy.special import gammaln, logsumexp
__all__ = ["Uniform",
           "UnitGammaPrior",
           "UnitChiSquaredPrior",
           "LogisticPrior",
           "DirichletPrior",
           "PowerSphericalPrior",
           "StudentTPrior",
           "SigmoidUniform"]

class Uniform():

  def __init__(self):
    pass

  def get_params(self):
    return {}

  def __call__(self, x, rng_key=None, inverse=False, reconstruction=False, **kwargs):

    if inverse == True and reconstruction == False:
      x = random.uniform(rng_key, minval=0, maxval=1.0, shape=x.shape)

    return x, jnp.zeros(x.shape[:1])

class UnitGammaPrior():

  def __init__(self):
    pass

  def get_params(self):
    return {}

  def __call__(self, x, rng_key=None, inverse=False, reconstruction=False, **kwargs):

    if inverse and reconstruction == False:
      x = random.gamma(rng_key, a=1.0, shape=x.shape)

    log_pz = jax.scipy.stats.gamma.logpdf(x, a=1.0)

    sum_axes = util.last_axes(x.shape[1:])
    log_pz = jnp.sum(log_pz, axis=sum_axes)
    return x, log_pz

class UnitChiSquaredPrior():

  def __init__(self, df=10):
    self.df = df

  def get_params(self):
    return {}

  def __call__(self, x, rng_key=None, inverse=False, reconstruction=False, **kwargs):

    if inverse and reconstruction == False:
      gaussian_draws = random.normal(rng_key, shape=(*x.shape, self.df))
      x = jnp.sum(gaussian_draws**2, axis=-1)

    log_pz = jax.scipy.stats.chi2.logpdf(x, df=self.df)

    sum_axes = util.last_axes(x.shape[1:])
    log_pz = jnp.sum(log_pz, axis=sum_axes)
    return x, log_pz

class LogisticPrior():

  def __init__(self):
    pass

  def get_params(self):
    return {}

  def __call__(self, x, rng_key=None, inverse=False, reconstruction=False, **kwargs):

    if inverse and reconstruction == False:
      x = random.logistic(rng_key, x.shape)

    log_pz = -2*jnp.logaddexp(0.5*x, -0.5*x)
    log_pz = jnp.sum(log_pz, axis=util.last_axes(x.shape[1:]))
    return x, log_pz

class DirichletPrior():

  def __init__(self):
    pass

  def get_params(self):
    return {}

  def __call__(self, x, rng_key=None, inverse=False, reconstruction=False, **kwargs):

    self.alpha = jnp.ones_like(x)

    if inverse and reconstruction == False:
      x = random.dirichlet(rng_key, self.alpha)

    log_x = jnp.log(x)
    log_pz = jnp.sum((self.alpha - 1)*log_x, axis=-1) + gammaln(self.alpha.sum(axis=-1)) - gammaln(self.alpha).sum(axis=-1)
    return x, log_pz

class PowerSphericalPrior():

  def __init__(self):
    pass

  def get_params(self):
    return {}

  def __call__(self, x, rng_key=None, inverse=False, reconstruction=False, **kwargs):

    self.mu = jnp.ones(x.shape[1:])
    self.mu = self.mu/jnp.linalg.norm(self.mu, axis=-1)
    self.kappa = 1.0

    sum_axes = util.last_axes(x.shape[1:])

    d = x.shape[-1]
    beta = 0.5*(d - 1)
    alpha = beta + self.kappa

    if inverse and reconstruction == False:
      k1, k2 = random.split(rng_key, 2)
      z = random.beta(k1, a=alpha, b=beta, shape=x.shape[:1])
      z = z[:,None]

      v = random.normal(k2, shape=x.shape[:-1] + (x.shape[-1] - 1,))
      v = v/jnp.linalg.norm(v, axis=-1, keepdims=True)

      t = 2*z - 1
      y = jnp.concatenate([t, jnp.sqrt(1 - t**2)*v], axis=-1)

      u_hat = jnp.zeros_like(self.mu).at[...,0].set(1.0) - self.mu
      u = u_hat/jnp.linalg.norm(u_hat, axis=-1)
      u = jnp.broadcast_to(u, y.shape)

      x = y - 2*u*jnp.sum(u*y, axis=sum_axes, keepdims=True)
      x = x/jnp.linalg.norm(x, axis=-1, keepdims=True) # Should already be normalized

    log_pz = -(alpha + beta)*jnp.log(2) - beta*jnp.log(jnp.pi)
    log_pz += jax.scipy.special.gammaln(alpha + beta) - jax.scipy.special.gammaln(alpha)
    log_pz += self.kappa*jnp.log1p(jnp.sum(self.mu*x, axis=sum_axes))

    return x, log_pz

class StudentTPrior():

  def __init__(self, df=50):
    self.df = df

  def get_params(self):
    return {}

  def __call__(self, x, rng_key=None, inverse=False, reconstruction=False, **kwargs):

    if inverse and reconstruction == False:
      x = random.t(rng_key, df=self.df, shape=x.shape)

    log_pz = jax.scipy.stats.t.logpdf(x, df=self.df)
    sum_axes = util.last_axes(x.shape[1:])
    log_pz = log_pz.sum(axis=sum_axes)

    return x, log_pz

from nux.flows.bijective.nonlinearities import SquareSigmoid
class SigmoidUniform():

  def __init__(self):
    self.eps = 1e-5

  def get_params(self):
    return {}

  def __call__(self, x, rng_key=None, inverse=False, reconstruction=False, **kwargs):

    if inverse == False:
      z, log_pz = SquareSigmoid()(x)

    else:
      if reconstruction == False:
        x = random.uniform(rng_key, minval=self.eps, maxval=1.0 - self.eps, shape=x.shape)
      z, log_pz = SquareSigmoid()(x, inverse=True)

    return z, log_pz

if __name__ == "__main__":
  from debug import *
  import scipy.stats

  key = random.PRNGKey(0)
  x = random.normal(key, shape=(1000, 4))*100

  prior = SigmoidUniform()
  z, log_pz = prior(x, rng_key=key)

  samples, _ = prior(x, rng_key=key, inverse=True)
  import pdb; pdb.set_trace()
