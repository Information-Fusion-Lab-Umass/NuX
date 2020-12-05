import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence, Any
import nux.util.weight_initializers as init
from jax.scipy.special import logsumexp

__all__ = ["logistic_cdf_mixture_logit",
           "LogisticLogit"]

@jax.custom_jvp
def logistic_cdf_mixture_logit(weight_logits, means, log_scales, x):
  # weight_logits doesn't have to be normalized with log_softmax!  This normalization
  # happens automatically when we compute z!

  scales = jnp.exp(-log_scales)
  shifted_x = x[...,None] - means
  x_hat = shifted_x*scales

  t1 = -jax.nn.softplus(-x_hat)
  t2 = t1 - x_hat

  t = weight_logits + jnp.concatenate([t1[None], t2[None]], axis=0)
  lse_t = logsumexp(t, axis=-1)
  log_z, log_1mz = lse_t
  z = log_z - log_1mz
  return z

@logistic_cdf_mixture_logit.defjvp
def jvp(primals, tangents):
  # We get the gradients (almost) for free when we evaluate the function
  weight_logits, means, log_scales, x = primals

  scales = jnp.exp(-log_scales)
  shifted_x = x[...,None] - means
  x_hat = shifted_x*scales

  t1 = -jax.nn.softplus(-x_hat)
  t2 = t1 - x_hat

  t12 = jnp.concatenate([t1[None], t2[None]], axis=0)
  t = weight_logits + t12
  lse_t = logsumexp(t, axis=-1)
  log_z, log_1mz = lse_t
  z = log_z - log_1mz

  # dz/dz_score
  softmax_t = jnp.exp(t - lse_t[...,None])
  softmax_t1, softmax_t2 = softmax_t
  sigma, sigma_bar = jnp.exp(t12)
  dx_hat = softmax_t1*sigma_bar + softmax_t2*sigma

  # Final gradients
  dmeans         = -dx_hat*scales
  dx             = -dmeans.sum(axis=-1)
  dscales        = dx_hat*shifted_x
  dlog_scales    = -scales*dscales
  dweight_logits = softmax_t1 - softmax_t2

  tangent_out = jnp.sum(dweight_logits*tangents[0], axis=-1)
  tangent_out += jnp.sum(dmeans*tangents[1], axis=-1)
  tangent_out += jnp.sum(dlog_scales*tangents[2], axis=-1)
  tangent_out += dx*tangents[3]

  return z, tangent_out

class LogisticLogit(hk.Module):

  def __init__(self,
               n_components: int=4,
               name: str="logistic_logit"
  ):
    """ Logistic mixtue cdf followed by logit.

        Let x_hat = (x - mean)/scale
        sigma     = sigmoid(x_hat)
        sigma_bar = sigmoid(-x_hat)
        z = log(<pi,sigma>/<pi,sigma_bar>)

        The implementation is written so that everything is numerically stable
        and has an efficient gradient.
    Args:
      n_components: Number of mixture components
      name        : Optional name for this module.
    """
    super().__init__(name=name)
    self.n_components = n_components

  def __call__(self, x, **kwargs):
    init = hk.initializers.RandomNormal()
    weight_logits = hk.get_parameter("weight_logits", (self.n_components,), init=jnp.zeros)
    means         = hk.get_parameter("means", (self.n_components,), init=init)
    log_scales    = hk.get_parameter("log_scales", (self.n_components,), init=jnp.zeros)

    log_scales = 1.5*jnp.tanh(log_scales)
    z = logistic_cdf_mixture_logit(weight_logits, means, log_scales, x)
    return z