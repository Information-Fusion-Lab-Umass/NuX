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
           "CoupingGaussianMixtureCDFWithLogitLinear",
           "CoupingLogisticMixtureCDFWithLogitLinear",
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

class MixtureCDF(AutoBatchedLayer):

  def __init__(self, n_components: int=4, name: str="mixture_cdf", **kwargs):
    super().__init__(name=name, **kwargs)
    self.n_components = n_components
    self.weight_init  = hk.initializers.RandomNormal()
    self.means_init   = hk.initializers.RandomNormal()
    self.vars_init    = hk.initializers.RandomNormal()

  def f(self, weight_logits, means, log_scales, x):
    assert 0

  def log_det(self, weight_logits, means, log_scales, x):
    assert 0

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    outputs = {}

    theta = hk.get_parameter("theta", shape=x.shape + (3*self.n_components,), dtype=x.dtype, init=hk.initializers.RandomNormal())
    if sample == False:
      outputs["x"], outputs["log_det"] = mixture_forward(self.f, self.log_det, x, theta)
    else:
      outputs["x"], outputs["log_det"] = mixture_inverse(self.f, self.log_det, x, theta)

    return outputs

################################################################################################################

class CouplingMixtureCDF(AutoBatchedLayer):

  def __init__(self,
               n_components: int=8,
               create_network: Callable=None,
               name: str="coupling_mixture_cdf",
               **kwargs):
    super().__init__(name=name)
    self.n_components   = n_components
    self.create_network = create_network
    self.network_kwargs = kwargs["res_net_kwargs"]

  def f(self, weight_logits, means, log_scales, x):
    assert 0

  def log_det(self, weight_logits, means, log_scales, x):
    assert 0

  def get_network(self, x1, x2, n_components):
    if(x1.ndim == 3):
      image_in = True
    else:
      image_in = False

    output_dim = x2.shape[-1]

    if self.create_network is not None:
      return self.create_network(output_dim)

    if(image_in):
      return net.ResNet(out_channel=output_dim*3*n_components,
                        **self.network_kwargs)

    return net.MLP(out_dim=output_dim*3*n_components,
                   layer_sizes=self.layer_sizes,
                   parameter_norm=self.parameter_norm,
                   nonlinearity="relu")

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]

    x1, x2 = jnp.split(x, jnp.array([x.shape[-1]//2]), axis=-1)

    # We will transform the first half of the pixels with fixed parametes
    theta1 = hk.get_parameter("theta1", shape=x1.shape + (3*self.n_components,), dtype=x1.dtype, init=hk.initializers.RandomNormal())

    # Get the conditioner network
    network = self.get_network(x1, x2, self.n_components)

    if sample == False:
      # Run the first part of the mixture
      z1, log_det1 = mixture_forward(self.f, self.log_det, x1, theta1)

      # Run the second part
      theta2 = network(x1).reshape(x2.shape + (3*self.n_components,))
      z2, log_det2 = mixture_forward(self.f, self.log_det, x2, theta2)

    else:
      # Run the first part of the mixture
      z1, log_det1 = mixture_inverse(self.f, self.log_det, x1, theta1)

      # Run the second part
      theta2 = network(z1).reshape(x2.shape + (3*self.n_components,))
      z2, log_det2 = mixture_inverse(self.f, self.log_det, x2, theta2)

    z = jnp.concatenate([z1, z2], axis=-1)
    log_det = log_det1 + log_det2

    return {"x": z, "log_det": log_det}

################################################################################################################

class CouplingMixtureCDFWithLogitLinear(AutoBatchedLayer):

  def __init__(self,
               n_components: int=8,
               create_network: Callable=None,
               name: str="coupling_mixture_cdf_logit_linear",
               **kwargs):
    super().__init__(name=name)
    self.n_components   = n_components
    self.create_network = create_network
    self.network_kwargs = kwargs["res_net_kwargs"]

  def f(self, weight_logits, means, log_scales, x):
    assert 0

  def log_det(self, weight_logits, means, log_scales, x):
    assert 0

  def get_network(self, x1, x2, n_components):
    if(x1.ndim == 3):
      image_in = True
    else:
      image_in = False

    output_dim = x2.shape[-1]

    if self.create_network is not None:
      return self.create_network(output_dim)

    if(image_in):
      return net.ResNet(out_channel=output_dim*(3*n_components + 2),
                        **self.network_kwargs)

    return net.MLP(out_dim=output_dim*(3*n_components + 2),
                   layer_sizes=self.layer_sizes,
                   parameter_norm=self.parameter_norm,
                   nonlinearity="relu")

  def separate_outputs(self, theta_and_linear):
    theta, linear = jnp.split(theta_and_linear, jnp.array([3*self.n_components]), axis=-1)
    log_scale, bias = jnp.split(linear, 2, axis=-1)
    log_scale, bias = log_scale.squeeze(axis=-1), bias.squeeze(axis=-1)
    return theta, (log_scale, bias)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]

    x1, x2 = jnp.split(x, jnp.array([x.shape[-1]//2]), axis=-1)

    # We will transform the first half of the pixels with fixed parametes
    theta1_and_linear = hk.get_parameter("theta1", shape=x1.shape + (3*self.n_components + 2,), dtype=x1.dtype, init=hk.initializers.RandomNormal(stddev=0.5))
    theta1, (log_scale1, bias1) = self.separate_outputs(theta1_and_linear)
    log_scale1 = jnp.tanh(log_scale1)

    # Get the conditioner network
    network = self.get_network(x1, x2, self.n_components)

    # Define the logit flow to go from (0,1) -> R
    logit = nux.Logit(scale=0.001)

    ################################################################################

    # Transform the first part of the input
    if sample == False:

      # Apply the mixture
      z1, log_det1 = mixture_forward(self.f, self.log_det, x1, theta1)
      z1_1 = z1

      # Apply the logit
      logit_outputs = logit({"x": z1}, sample=False, no_batching=True)
      z1 = logit_outputs["x"]
      log_det1 += logit_outputs["log_det"]
      z1_2 = z1

      # Apply the shift and scale
      z1 = (z1 - bias1)*jnp.exp(-log_scale1)
      log_det1 += -log_scale1.sum()

      z1_3 = z1

    else:

      # Undo the shift and scale
      z1 = jnp.exp(log_scale1)*x1 + bias1
      z1_1 = z1
      log_det1 = -log_scale1.sum()

      # Undo the logit
      logit_outputs = logit({"x": z1}, sample=True, no_batching=True)
      z1 = logit_outputs["x"]
      z1_2 = z1
      log_det1 += logit_outputs["log_det"]

      # Undo the mixture
      z1, log_det_mix1 = mixture_inverse(self.f, self.log_det, z1, theta1)
      z1_3 = z1
      log_det1 += log_det_mix1

    ################################################################################

    # Transform the second part of the input
    if sample == False:

      # Apply the mixture
      theta2_and_linear = network(x1).reshape(x2.shape + (3*self.n_components + 2,))
      theta2, (log_scale2, bias2) = self.separate_outputs(theta2_and_linear)
      log_scale2 = jnp.tanh(log_scale2)

      z2, log_det2 = mixture_forward(self.f, self.log_det, x2, theta2)
      z2_1 = z2

      # Apply the logit
      logit_outputs = logit({"x": z2}, sample=False, no_batching=True)
      z2 = logit_outputs["x"]
      log_det2 += logit_outputs["log_det"]
      z2_2 = z2

      # Apply the shift and scale
      z2 = (z2 - bias2)*jnp.exp(-log_scale2)
      log_det2 += -log_scale2.sum()

      z2_3 = z2

    else:
      # Get the parameters of for the inverse
      theta2_and_linear = network(z1).reshape(x2.shape + (3*self.n_components + 2,))
      theta2, (log_scale2, bias2) = self.separate_outputs(theta2_and_linear)
      log_scale2 = jnp.tanh(log_scale2)

      # Undo the shift and scale
      z2 = jnp.exp(log_scale2)*x2 + bias2
      z2_1 = z2
      log_det2 = -log_scale2.sum()

      # Undo the logit
      logit_outputs = logit({"x": z2}, sample=True, no_batching=True)
      z2 = logit_outputs["x"]
      z2_2 = z2
      log_det2 += logit_outputs["log_det"]

      # Run the second part
      z2, log_det_mix2 = mixture_inverse(self.f, self.log_det, z2, theta2)
      log_det2 += log_det_mix2
      z2_3 = z2


    z = jnp.concatenate([z1, z2], axis=-1)
    log_det = log_det1 + log_det2

    return {"x": z, "log_det": log_det}

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
  def __init__(self, n_components: int=4, name: str="logistic_mixture_cdf", **kwargs):
    super().__init__(n_components=n_components, name=name, **kwargs)

  def f(self, weight_logits, means, log_scales, x):
    # log_scales = jnp.tanh(log_scales)
    z_scores = (x - means)*jnp.exp(-log_scales)
    log_z = logsumexp(weight_logits - jax.nn.softplus(-z_scores))
    log_1mz = logsumexp(weight_logits - jax.nn.softplus(z_scores))
    return log_z - log_1mz

  def log_det(self, weight_logits, means, log_scales, x):
    # log_scales = jnp.tanh(log_scales)
    z_scores = (x - means)*jnp.exp(-log_scales)
    t1 = -jax.nn.softplus(-z_scores)
    t2 = -jax.nn.softplus(z_scores)

    a = weight_logits + t1
    b = weight_logits + t2

    log_pdf = -log_scales + t1 + t2
    mixture_log_pdf = logsumexp(weight_logits - log_scales + t1 + t2)

    logit_log_det = logsumexp(jnp.hstack([a, b])) - logsumexp(a) - logsumexp(b)

    return mixture_log_pdf + logit_log_det

################################################################################################################

class GaussianMixtureCDF(_GaussianMixtureMixin, MixtureCDF):
  pass

class LogisticMixtureCDF(_LogitsticMixtureMixin, MixtureCDF):
  pass

class CoupingGaussianMixtureCDF(_GaussianMixtureMixin, CouplingMixtureCDF):
  pass

class CoupingLogisticMixtureCDF(_LogitsticMixtureMixin, CouplingMixtureCDF):
  pass

class CoupingGaussianMixtureCDFWithLogitLinear(_GaussianMixtureMixin, CouplingMixtureCDFWithLogitLinear):
  pass

class CoupingLogisticMixtureCDFWithLogitLinear(_LogitsticMixtureMixin, CouplingMixtureCDFWithLogitLinear):
  pass

class LogitsticMixtureLogit(_LogitsticMixtureLogitMixin, MixtureCDF):
  pass

class CouplingLogitsticMixtureLogit(_LogitsticMixtureLogitMixin, CouplingMixtureCDF):
  pass