import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
from jax.scipy.special import logsumexp
from jax.flatten_util import ravel_pytree
from nux.flows.base import Flow

__all__ = ["LogisticCDFMixtureLogit"]

@jax.custom_jvp
def logistic_cdf_mixture_logit(weight_logits, means, scales, x):
  # weight_logits doesn't have to be normalized with log_softmax.
  # This normalization happens automatically when we compute z.

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
  weight_logits, means, scales, x = primals

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
  dweight_logits = softmax_t1 - softmax_t2

  tangent_out = jnp.sum(dweight_logits*tangents[0], axis=-1)
  tangent_out += jnp.sum(dmeans*tangents[1], axis=-1)
  tangent_out += jnp.sum(dscales*tangents[2], axis=-1)
  tangent_out += dx*tangents[3]

  return z, tangent_out

################################################################################################################

class LogisticCDFMixtureLogit(Flow):

  def __init__(self, K: int=4, newton_inverse=False, **kwargs):
    """ Used in Flow++ https://arxiv.org/pdf/1902.00275.pdf
    """
    self.K = K
    self.newton_inverse = newton_inverse

  @property
  def param_multiplier(self):
    return 3*self.K

  def get_param_dim(self, dim):
    return self.param_multiplier*dim

  def get_params(self):
    return {"theta": self.theta}

  @property
  def coupling_param_keys(self):
    return ("theta",)

  def extract_coupling_params(self, theta):
    return (theta,)

  def __call__(self, x, params=None, inverse=False, rng_key=None, **kwargs):

    if params is None:
      x_shape = x.shape[1:]
      self.theta = random.normal(rng_key, shape=x_shape[:-1] + (x_shape[-1]*self.param_multiplier,))
    else:
      self.theta = params["theta"]

    if self.theta.ndim == x.ndim - 1:
      # Not using coupling
      in_axes = (None, None, None, 0)
      theta = self.theta.reshape(x.shape[1:] + (self.param_multiplier,))
    else:
      in_axes = 0
      theta = self.theta.reshape(x.shape + (self.param_multiplier,))

    # Split the parameters
    weight_logits, means, scales = theta[...,:self.K], theta[...,self.K:2*self.K], theta[...,2*self.K:]
    scales = util.square_plus(scales, gamma=1.0) + 1e-4

    # Create the jvp function that we'll need
    @partial(jax.vmap, in_axes=in_axes)
    def _f_and_df(weight_logits, means, scales, x):
      primals = weight_logits, means, scales, x
      tangents = jax.tree_util.tree_map(jnp.zeros_like, primals[:-1]) + (jnp.ones_like(x),)
      return jax.jvp(logistic_cdf_mixture_logit, primals, tangents)

    # Fill with the parameters
    f_and_df = partial(_f_and_df, weight_logits, means, scales)

    if inverse == False:
      # Only need a single pass
      z, dzdx = f_and_df(x)
    else:
      if self.newton_inverse:
        # Invert with newtons method.  Might be unstable.
        z = util.newtons_with_grad(f_and_df, x)
      else:
        # Invert with bisection method.
        f = lambda x: f_and_df(x)[0]
        lower, upper = -1000.0, 1000.0
        lower, upper = jnp.broadcast_to(lower, x.shape), jnp.broadcast_to(upper, x.shape)
        z = util.bisection(f, lower, upper, x)
      reconstr, dzdx = f_and_df(z)
    ew_log_det = jnp.log(dzdx)

    sum_axes = util.last_axes(x.shape[1:])
    log_det = ew_log_det.sum(sum_axes)

    return z, log_det

################################################################################################################

def regular_test():

  rng_key = random.PRNGKey(0)
  x = random.normal(rng_key, shape=(2, 4, 4, 2))
  flow = LogisticCDFMixtureLogit(4)
  z, log_det = flow(x, rng_key=rng_key)
  params = flow.get_params()

  reconstr, _ = flow(z, params, inverse=True)
  z2, _ = flow(reconstr, params, inverse=False)
  assert jnp.allclose(x, reconstr)

  flat_x, unflatten = ravel_pytree(x)
  def flat_call(flat_x):
    x = unflatten(flat_x)
    z, _ = flow(x, params=params)
    return z.ravel()

  J = jax.jacobian(flat_call)(flat_x)
  true_log_det = jnp.linalg.slogdet(J)[1]
  assert jnp.allclose(log_det.sum(), true_log_det)

def coupling_test():

  rng_key = random.PRNGKey(0)
  x = random.normal(rng_key, shape=(2, 4, 4, 2))

  K = 4
  theta = random.normal(rng_key, shape=x.shape[:-1] + (3*K*x.shape[-1],))
  params = dict(theta=theta)

  flow = LogisticCDFMixtureLogit(4)
  z, log_det = flow(x, params=params, rng_key=rng_key)
  reconstr, _ = flow(z, params, inverse=True)
  assert jnp.allclose(x, reconstr)

  flat_x, unflatten = ravel_pytree(x)
  def flat_call(flat_x):
    x = unflatten(flat_x)
    z, _ = flow(x, params=params)
    return z.ravel()

  J = jax.jacobian(flat_call)(flat_x)
  true_log_det = jnp.linalg.slogdet(J)[1]
  assert jnp.allclose(log_det.sum(), true_log_det)

if __name__ == "__main__":
  from debug import *

  regular_test()
  coupling_test()
