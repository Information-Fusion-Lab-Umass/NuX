import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence
from nux.flows.base import *
import nux.util as util
from jax.scipy.special import logsumexp
import nux.networks as net
import nux

__all__ = ["GaussianMixtureCDF",
           "LogisticMixtureCDF",
           "CoupingGaussianMixtureCDF",
           "CoupingLogisticMixtureCDF",
           "LogitsticMixtureLogit",
           "CouplingLogitsticMixtureLogit"]

################################################################################################################

def bisection_body(f, carry, inputs):
  x, current_x, current_z, lower, upper = carry

  gt = current_x > x
  lt = 1.0 - gt

  new_z = gt*0.5*(current_z + lower) + lt*0.5*(current_z + upper)
  lower = gt*lower                   + lt*current_z
  upper = gt*current_z               + lt*upper

  current_z = new_z
  current_x = f(current_z)
  dx = current_x - x

  return (x, current_x, current_z, lower, upper), dx

def bisection(f, lower, upper, x, n_iters=10000):
  # Compute f^{-1}(x)
  z = jnp.zeros_like(x)

  carry = (x, f(z), z, lower, upper)
  carry, diffs = jax.lax.scan(partial(bisection_body, f), carry, jnp.arange(n_iters))
  x, current_x, current_z, lower, upper = carry
  return current_z

################################################################################################################

def mixture_forward(eval_fun, log_det_fun, x, theta):
  # Split the parameters
  n_components = theta.shape[-1]//3
  weight_logits, means, log_scales = jnp.split(theta, jnp.array([n_components, 2*n_components]), axis=-1)
  weight_logits = jax.nn.log_softmax(weight_logits)

  # We're going to have a set of parameters for each element of x
  assert means.shape[:-1] == x.shape

  # We are going to vmap over each pixel
  f = eval_fun
  log_det = log_det_fun
  for i in range(len(x.shape)):
    f = vmap(f)
    log_det = vmap(log_det)

  # Apply the mixture
  return f(weight_logits, means, log_scales, x), log_det(weight_logits, means, log_scales, x).sum()

def mixture_forward2(f_and_log_det_fun, x, theta):
  # Split the parameters
  n_components = theta.shape[-1]//3
  weight_logits, means, log_scales = jnp.split(theta, jnp.array([n_components, 2*n_components]), axis=-1)
  weight_logits = jax.nn.log_softmax(weight_logits)

  # We're going to have a set of parameters for each element of x
  assert means.shape[:-1] == x.shape

  # We are going to vmap over each pixel
  f_and_log_det = f_and_log_det_fun
  for i in range(len(x.shape)):
    f_and_log_det = vmap(f_and_log_det)

  # Apply the mixture
  z, log_det = f_and_log_det(weight_logits, means, log_scales, x)
  return z, log_det.sum()

def mixture_inverse(eval_fun, log_det_fun, x, theta):
  # Split the parameters
  n_components = theta.shape[-1]//3
  weight_logits, means, log_scales = jnp.split(theta, jnp.array([n_components, 2*n_components]), axis=-1)
  weight_logits = jax.nn.log_softmax(weight_logits)

  def bisection_no_vmap(weight_logits, means, log_scales, x):
    # Write a wrapper around the inverse function
    assert weight_logits.ndim == 1
    assert x.ndim == 0

    # Define the starting search range
    # lower = jnp.min(means - 200*jnp.exp(log_scales))
    # upper = jnp.max(means + 200*jnp.exp(log_scales))

    # If we're outside of this range, then there's a bigger problem in the rest of the network.
    lower = jnp.zeros_like(x) - 1000
    upper = jnp.zeros_like(x) + 1000

    filled_f = partial(eval_fun, weight_logits, means, log_scales)
    return bisection(filled_f, lower, upper, x)

  # We are going to vmap over each pixel
  f_inv = bisection_no_vmap
  log_det = log_det_fun
  for i in range(len(x.shape)):
    f_inv = vmap(f_inv)
    log_det = vmap(log_det)

  # Apply the mixture inverse
  z = f_inv(weight_logits, means, log_scales, x)
  return z, log_det(weight_logits, means, log_scales, z).sum()

################################################################################################################

class MixtureCDF(Layer):

  def __init__(self, n_components: int=4, name: str="mixture_cdf", **kwargs):
    super().__init__(name=name, **kwargs)
    self.n_components = n_components

    self.forward = partial(mixture_forward, self.f, self.log_det)
    self.inverse = partial(mixture_inverse, self.f, self.log_det)

  def f(self, weight_logits, means, log_scales, x):
    assert 0

  def log_det(self, weight_logits, means, log_scales, x):
    assert 0

  def f_and_log_det(self, weight_logits, means, log_scales, x):
    return self.f(weight_logits, means, log_scales, x), self.log_det(weight_logits, means, log_scales, x)

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    outputs = {}

    x_shape = self.get_unbatched_shapes(sample)["x"]
    theta = hk.get_parameter("theta", shape=x_shape + (3*self.n_components,), dtype=x.dtype, init=hk.initializers.RandomNormal())
    if sample == False:
      outputs["x"], outputs["log_det"] = self.auto_batch(self.forward, in_axes=(0, None))(x, theta)
    else:
      outputs["x"], outputs["log_det"] = self.auto_batch(self.inverse, in_axes=(0, None))(x, theta)

    return outputs

################################################################################################################

from nux.flows.bijective.coupling_base import CouplingBase

class CouplingMixtureCDF(CouplingBase):

  def __init__(self,
               n_components: int=8,
               create_network: Callable=None,
               name: str="coupling_mixture_cdf",
               network_kwargs: Optional=None,
               reverse: Optional[bool]=False,
               use_condition: bool=False,
               **kwargs):
    super().__init__(name=name)
    self.n_components   = n_components
    self.create_network = create_network
    self.network_kwargs = network_kwargs
    self.reverse        = reverse
    self.use_condition  = use_condition

    self.forward = partial(mixture_forward2, self.f_and_log_det)
    # self.forward = partial(mixture_forward, self.f, self.log_det)
    self.inverse = partial(mixture_inverse, self.f, self.log_det)

  def f(self, weight_logits, means, log_scales, x):
    assert 0

  def log_det(self, weight_logits, means, log_scales, x):
    assert 0

  def f_and_log_det(self, weight_logits, means, log_scales, x):
    return self.f(weight_logits, means, log_scales, x), self.log_det(weight_logits, means, log_scales, x)

  def get_out_shape(self, x):
    x_shape = x.shape[len(self.batch_shape):]
    out_dim = x_shape[-1]*3*self.n_components
    return x_shape[:-1] + (out_dim,)

  def transform(self, x, params=None, sample=False):
    if params is None:
      x_shape = x.shape[len(self.batch_shape):]
      theta = hk.get_parameter("theta", shape=x_shape + (3*self.n_components,), dtype=x.dtype, init=hk.initializers.RandomNormal())
      in_axes = (0, None)
    else:
      theta = params.reshape(x.shape + (3*self.n_components,))
      in_axes = (0, 0)

    if sample == self.reverse:
      z, log_det = self.auto_batch(self.forward, in_axes=in_axes)(x, theta)
    else:
      z, log_det = self.auto_batch(self.inverse, in_axes=in_axes)(x, theta)

    if self.reverse:
      log_det *= -1

    return z, log_det

################################################################################################################

class _GaussianMixtureMixin():

  def __init__(self, n_components: int=4, name: str="gaussian_mixture_cdf", **kwargs):
    super().__init__(n_components=n_components, name=name, **kwargs)

  def f(self, weight_logits, means, log_scales, x):
    dx = x - means
    cdf = jax.scipy.special.ndtr(dx*jnp.exp(-0.5*log_scales))
    z = jnp.sum(jnp.exp(weight_logits)*cdf)
    return z

  def log_det(self, weight_logits, means, log_scales, x):
    # log_det is log_pdf(x)
    dx = x - means
    log_pdf = -0.5*(dx**2)*jnp.exp(-log_scales) - 0.5*log_scales - 0.5*jnp.log(2*jnp.pi)
    return logsumexp(weight_logits + log_pdf, axis=-1).sum()


class _LogitsticMixtureMixin():

  def __init__(self, n_components: int=4, name: str="logistic_mixture_cdf", **kwargs):
    super().__init__(n_components=n_components, name=name, **kwargs)

  def f(self, weight_logits, means, log_scales, x):
    z_scores = (x - means)*jnp.exp(-log_scales)
    log_cdf = jax.nn.log_sigmoid(z_scores)
    z = jax.scipy.special.logsumexp(weight_logits + log_cdf, axis=-1).sum()
    return jnp.exp(z)

  def log_det(self, weight_logits, means, log_scales, x):
    # log_det is log_pdf(x)
    z_scores = (x - means)*jnp.exp(-log_scales)
    log_pdf = -log_scales + jax.nn.log_sigmoid(z_scores) + jax.nn.log_sigmoid(-z_scores)
    return logsumexp(weight_logits + log_pdf, axis=-1).sum()


class _LogitsticMixtureLogitMixin():
  """ Combined logistic mixture -> logit """
  def __init__(self, n_components: int=4, name: str="logistic_mixture_cdf_logit", restrict_scales: bool=False, **kwargs):
    super().__init__(n_components=n_components, name=name, **kwargs)
    self.restrict_scales = restrict_scales

  def f(self, weight_logits, means, log_scales, x):
    log_scales = jnp.logaddexp(log_scales, -12)
    if self.restrict_scales:
      log_scales = 1.5*jnp.tanh(log_scales)

    z_scores = (x - means)*jnp.exp(-log_scales)

    t1 = -jax.nn.softplus(-z_scores)
    t2 = t1 - z_scores

    log_z = logsumexp(weight_logits + t1)
    log_1mz = logsumexp(weight_logits + t2)
    return log_z - log_1mz

  def log_det(self, weight_logits, means, log_scales, x):
    log_scales = jnp.logaddexp(log_scales, -12)
    if self.restrict_scales:
      log_scales = 1.5*jnp.tanh(log_scales)

    z_scores = (x - means)*jnp.exp(-log_scales)

    t1 = -jax.nn.softplus(-z_scores)
    t2 = t1 - z_scores
    # t2 = -jax.nn.softplus(z_scores)

    a = weight_logits + t1
    b = weight_logits + t2

    log_pdf = -log_scales + t1 + t2
    mixture_log_pdf = logsumexp(weight_logits + log_pdf)

    logit_log_det = logsumexp(jnp.hstack([a, b])) - logsumexp(a) - logsumexp(b)

    return mixture_log_pdf + logit_log_det

  def f_and_log_det(self, weight_logits, means, log_scales, x):
    log_scales = jnp.logaddexp(log_scales, -12)
    if self.restrict_scales:
      log_scales = 1.5*jnp.tanh(log_scales)

    z_scores = (x - means)*jnp.exp(-log_scales)

    t1 = -jax.nn.softplus(-z_scores)
    t2 = t1 - z_scores

    a = weight_logits + t1
    b = weight_logits + t2

    log_z = logsumexp(a)
    log_1mz = logsumexp(b)
    z = log_z - log_1mz

    log_pdf = -log_scales + t1 + t2
    mixture_log_pdf = logsumexp(weight_logits + log_pdf)

    logit_log_det = logsumexp(jnp.hstack([a, b])) - log_z - log_1mz

    log_det = mixture_log_pdf + logit_log_det
    return z, log_det

################################################################################################################

class GaussianMixtureCDF(_GaussianMixtureMixin, MixtureCDF):
  pass

class LogisticMixtureCDF(_LogitsticMixtureMixin, MixtureCDF):
  pass

class CoupingGaussianMixtureCDF(_GaussianMixtureMixin, CouplingMixtureCDF):
  pass

class CoupingLogisticMixtureCDF(_LogitsticMixtureMixin, CouplingMixtureCDF):
  pass

class LogitsticMixtureLogit(_LogitsticMixtureLogitMixin, MixtureCDF):
  pass

class CouplingLogitsticMixtureLogit(_LogitsticMixtureLogitMixin, CouplingMixtureCDF):
  pass
