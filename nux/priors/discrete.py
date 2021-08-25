import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
from typing import Optional, Mapping, Callable, Sequence
import nux.util as util

__all__ = ["DiscreteUnitGaussianPrior",
           "DiscreteUnitLogisticPrior",
           "DiscreteLogisticPrior",
           "BetaBinomialPrior"]

class DiscreteUnitGaussianPrior():

  def __init__(self):
    pass

  def get_params(self):
    return {}

  def __call__(self, x, rng_key=None, inverse=False, reconstruction=False, **kwargs):

    if inverse and reconstruction == False:
      x = random.normal(rng_key, x.shape)
      x = jnp.round(x)

    pmf = 0.5*(jax.scipy.special.erf(x + 1) - jax.scipy.special.erf(x))
    log_pmf = jnp.sum(log_pmf, axis=util.last_axes(x.shape[1:]))
    return x, log_pmf

class DiscreteUnitLogisticPrior():

  def __init__(self):
    pass

  def get_params(self):
    return {}

  def __call__(self, x, rng_key=None, inverse=False, reconstruction=False, **kwargs):

    if inverse and reconstruction == False:
      x = random.logistic(rng_key, x.shape)
      x = jnp.round(x)

    sum_axes = util.last_axes(x.shape[1:])

    # Numerically stable version of jnp.log(jax.nn.sigmoid(x + 0.5) - jax.nn.sigmoid(x - 0.5))
    a = 0.5 + x
    b = 0.5 - x
    one = jnp.ones_like(x)
    zero = jnp.zeros_like(x)
    terms = jnp.concatenate([a[None], b[None], one[None], zero[None]])
    log_pmf = jnp.log(jnp.e - 1.0) - jax.scipy.special.logsumexp(terms, axis=0)
    log_pmf = jnp.sum(log_pmf, axis=sum_axes)
    return x, log_pmf

class DiscreteLogisticPrior():

  def __init__(self):
    pass

  def get_params(self):
    return dict(mu=self.mu, s=self.s)

  def __call__(self, x, params=None, rng_key=None, inverse=False, reconstruction=False, **kwargs):

    if params is None:
      self.mu, self.s = jnp.zeros((2, *x.shape[1:]))
    else:
      self.mu, self.s = params["mu"], params["s"]

    s = util.square_plus(self.s, gamma=1.0) + 1e-4

    if inverse and reconstruction == False:
      x = random.logistic(rng_key, x.shape)
      x = jnp.round(self.mu + x*s)

    sum_axes = util.last_axes(x.shape[1:])

    # true_log_pmf = jnp.log(jax.nn.sigmoid((x + 0.5 - self.mu)/s) - jax.nn.sigmoid((x - 0.5 - self.mu)/s))

    # Numerically stable version of the log pmf
    a = (-0.5 + x - self.mu)/s
    b = (-0.5 - x + self.mu)/s
    neg_one_over_s = -jnp.ones_like(x)/s
    zero = jnp.zeros_like(x)
    terms = jnp.concatenate([a[None], b[None], neg_one_over_s[None], zero[None]])
    log_pmf = jnp.log1p(-jnp.e**(-1/s)) - jax.scipy.special.logsumexp(terms, axis=0)
    log_pmf = jnp.sum(log_pmf, axis=sum_axes)
    return x, log_pmf

class BetaBinomialPrior():

  def __init__(self, alpha, beta, n):
    self.alpha = alpha
    self.beta = beta
    self.n = n

  def get_params(self):
    return {}

  def __call__(self, x, params=None, rng_key=None, inverse=False, reconstruction=False, **kwargs):

    if inverse and reconstruction == False:
      # Sample from the cdf
      cdf = jax.scipy.stats.betabinom.pmf(jnp.arange(0,self.n), self.n, self.alpha, self.beta).cumsum()
      u = random.uniform(rng_key, x.shape)
      x = jnp.sum(cdf < u[...,None], axis=-1)

    log_pmf = jax.scipy.stats.betabinom.logpmf(x, self.n, self.alpha, self.beta)

    sum_axes = util.last_axes(x.shape[1:])
    log_pmf = jnp.sum(log_pmf, axis=sum_axes)

    return x, log_pmf


if __name__ == "__main__":
  from debug import *
  import matplotlib.pyplot as plt
  import scipy.stats
  from .mixture import Mixture

  rng_key = random.PRNGKey(0)
  x = random.randint(rng_key, minval=-10, maxval=10, shape=(1000,))

  # x = jnp.arange(-100, 100)*1.0
  # x = x[:,None]

  x = random.randint(rng_key, minval=0, maxval=6, shape=(100000, 1))

  # prior = Mixture(5, DiscreteLogisticPrior())
  prior = BetaBinomialPrior(0.5, 1.5, n=4)
  _, log_pmf = prior(x, rng_key=rng_key)
  pmf = jnp.exp(log_pmf)
  params = prior.get_params()

  samples, samples_log_pmf = prior(x, params=params, rng_key=rng_key, inverse=True)
  bins = jnp.bincount(samples.ravel())
  normalized_bins = bins/bins.sum()

  compare = jnp.arange(0, 10)
  _, log_compare_pmf = prior(compare, rng_key=rng_key)
  samples_compare_pmf = jnp.exp(log_compare_pmf)

  import pdb; pdb.set_trace()

