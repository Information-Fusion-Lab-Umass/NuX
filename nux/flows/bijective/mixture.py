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
from nux.networks.nonlinearities import logistic_cdf_mixture_logit
import nux

__all__ = ["GaussianMixtureCDF",
           "LogisticMixtureCDF",
           "CoupingGaussianMixtureCDF",
           "CoupingLogisticMixtureCDF",
           "LogitsticMixtureLogit",
           "CouplingLogitsticMixtureLogit"]

################################################################################################################

def bisection_body(f, val):
  x, current_x, current_z, lower, upper, dx, i = val

  gt = current_x > x
  lt = 1.0 - gt

  new_z = gt*0.5*(current_z + lower) + lt*0.5*(current_z + upper)
  lower = gt*lower                   + lt*current_z
  upper = gt*current_z               + lt*upper

  current_z = new_z
  current_x = f(current_z)
  dx = current_x - x

  return x, current_x, current_z, lower, upper, dx, i + 1

def bisection(f, lower, upper, x, atol=1e-8, max_iters=10000):
  # Compute f^{-1}(x) using the bisection method.  f must be monotonic.
  z = jnp.zeros_like(x)

  def cond_fun(val):
    x, current_x, current_z, lower, upper, dx, i = val

    max_iters_reached = jnp.where(i > max_iters, True, False)
    tolerance_achieved = jnp.allclose(dx, 0.0, atol=atol)

    return ~(max_iters_reached | tolerance_achieved)

  val = (x, f(z), z, lower, upper, 10.0, 0.0)
  val = jax.lax.while_loop(cond_fun, partial(bisection_body, f), val)
  x, current_x, current_z, lower, upper, dx, i = val
  return current_z

################################################################################################################

def mixture_forward(f_and_log_det_fun, x, theta):
  # Split the parameters
  n_components = theta.shape[-1]//3
  weight_logits, means, log_scales = jnp.split(theta, jnp.array([n_components, 2*n_components]), axis=-1)

  # We are going to vmap over each pixel
  f_and_log_det = f_and_log_det_fun
  for i in range(len(x.shape)):
    in_axes = [0, 0, 0, 0]
    in_axes[0] = None if weight_logits.ndim - 1 <= i else 0
    in_axes[1] = None if means.ndim - 1 <= i else 0
    in_axes[2] = None if log_scales.ndim - 1 <= i else 0
    f_and_log_det = vmap(f_and_log_det, in_axes=in_axes)

  # Apply the mixture
  z, log_det = f_and_log_det(weight_logits, means, log_scales, x)
  return z, log_det.sum()

def mixture_inverse(eval_fun, log_det_fun, x, theta):
  # Split the parameters
  n_components = theta.shape[-1]//3
  weight_logits, means, log_scales = jnp.split(theta, jnp.array([n_components, 2*n_components]), axis=-1)

  def bisection_no_vmap(weight_logits, means, log_scales, x):
    # Write a wrapper around the inverse function
    assert weight_logits.ndim == 1
    assert x.ndim == 0

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

  def __init__(self,
               n_components: int=4,
               name: str="mixture_cdf"
  ):
    """ Base class for a mixture cdf with no coupling
    Args:
      n_components: Number of mixture components to use
      name        : Optional name for this module.
    """
    super().__init__(name=name)
    self.n_components = n_components

    self.forward = partial(mixture_forward, self.f_and_log_det)
    self.inverse = partial(mixture_inverse, self.f, self.log_det)

  def f(self, weight_logits, means, log_scales, x):
    assert 0

  def log_det(self, weight_logits, means, log_scales, x):
    assert 0

  def f_and_log_det(self, weight_logits, means, log_scales, x):
    return self.f(weight_logits, means, log_scales, x), self.log_det(weight_logits, means, log_scales, x)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    outputs = {}

    x_shape = self.get_unbatched_shapes(sample)["x"]
    theta = hk.get_parameter("theta", shape=(3*self.n_components,), dtype=x.dtype, init=hk.initializers.RandomNormal(0.1))
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
               reverse: Optional[bool]=False,
               create_network: Callable=None,
               network_kwargs: Optional=None,
               use_condition: bool=False,
               name: str="coupling_mixture_cdf"
  ):
    """ Base class for a mixture cdf with coupling
    Args:
      n_components  : Number of mixture components to use
      reverse       : Reverse the flow.  We might want this class for only sampling,
                      so it wouldn't make sense to always invert with the iterative method.
      create_network: Function to create the conditioner network.  Should accept a tuple
                      specifying the output shape.  See coupling_base.py
      use_condition : Should we use inputs["condition"] to form t([xb, condition]), s([xb, condition])?
      network_kwargs: Dictionary with settings for the default network (see get_default_network in util.py)
      name          : Optional name for this module.
    """
    super().__init__(name=name)
    self.n_components   = n_components
    self.create_network = create_network
    self.network_kwargs = network_kwargs
    self.reverse        = reverse
    self.use_condition  = use_condition

    self.forward = partial(mixture_forward, self.f_and_log_det)
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
      theta = hk.get_parameter("theta", shape=x_shape + (3*self.n_components,), dtype=x.dtype, init=hk.initializers.RandomNormal(0.1))
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

  def __init__(self,
               n_components: int=4,
               name: str="gaussian_mixture_cdf"
  ):
    """ Mix in class for Gaussian mixture cdf models
    Args:
      n_components  : Number of mixture components to use
      name          : Optional name for this module.
    """
    super().__init__(n_components=n_components, name=name)

  def f(self, weight_logits, means, log_scales, x):
    weight_logits = jax.nn.log_softmax(weight_logits)
    dx = x - means
    cdf = jax.scipy.special.ndtr(dx*jnp.exp(-0.5*log_scales))
    z = jnp.sum(jnp.exp(weight_logits)*cdf)
    return z

  def log_det(self, weight_logits, means, log_scales, x):
    weight_logits = jax.nn.log_softmax(weight_logits)
    # log_det is log_pdf(x)
    dx = x - means
    log_pdf = -0.5*(dx**2)*jnp.exp(-log_scales) - 0.5*log_scales - 0.5*jnp.log(2*jnp.pi)
    return logsumexp(weight_logits + log_pdf, axis=-1).sum()


class _LogitsticMixtureMixin():

  def __init__(self,
               n_components: int=4,
               name: str="logistic_mixture_cdf"
  ):
    """ Mix in class for logistic mixture cdf models
    Args:
      n_components  : Number of mixture components to use
      name          : Optional name for this module.
    """
    super().__init__(n_components=n_components, name=name)

  def f(self, weight_logits, means, log_scales, x):
    weight_logits = jax.nn.log_softmax(weight_logits)
    z_scores = (x - means)*jnp.exp(-log_scales)
    log_cdf = jax.nn.log_sigmoid(z_scores)
    z = jax.scipy.special.logsumexp(weight_logits + log_cdf, axis=-1).sum()
    return jnp.exp(z)

  def log_det(self, weight_logits, means, log_scales, x):
    weight_logits = jax.nn.log_softmax(weight_logits)
    # log_det is log_pdf(x)
    z_scores = (x - means)*jnp.exp(-log_scales)
    log_pdf = -log_scales + jax.nn.log_sigmoid(z_scores) + jax.nn.log_sigmoid(-z_scores)
    return logsumexp(weight_logits + log_pdf, axis=-1).sum()


class _LogitsticMixtureLogitMixin():

  def __init__(self,
               n_components: int=4,
               name: str="logistic_mixture_cdf_logit",
               restrict_scales: bool=True
  ):
    """ Mix in class for logistic mixture cdf followed by logit models.
        This works pretty well in practice.  See nux/networks/nonlinearities.py
    Args:
      n_components   : Number of mixture components to use
      restrict_scales: Whether or not to bound the scales.  If log_scales is
                       unbounded, we can get model more complex distributions
                       at the risk of numerical instability.
      name           : Optional name for this module.
    """
    super().__init__(n_components=n_components, name=name)
    self.restrict_scales = restrict_scales

  def f(self, weight_logits, means, log_scales, x):
    if self.restrict_scales:
      log_scales = 1.5*jnp.tanh(log_scales)

    return logistic_cdf_mixture_logit(weight_logits, means, log_scales, x)

  def log_det(self, weight_logits, means, log_scales, x):
    if self.restrict_scales:
      log_scales = 1.5*jnp.tanh(log_scales)

    dzdx = jax.grad(logistic_cdf_mixture_logit, argnums=3)(weight_logits, means, log_scales, x)
    return jnp.log(dzdx)

  def f_and_log_det(self, weight_logits, means, log_scales, x):
    if self.restrict_scales:
      log_scales = 1.5*jnp.tanh(log_scales)

    z, dzdx = jax.jit(jax.value_and_grad(logistic_cdf_mixture_logit, argnums=3))(weight_logits, means, log_scales, x)
    log_det = jnp.log(dzdx)

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
