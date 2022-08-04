import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import nux.util as util
from jax.flatten_util import ravel_pytree
from nux.flows.base import Flow

__all__ = ["RationalQuadraticSpline"]

################################################################################################################

def forward_spline(x,
                   mask,
                   knot_x_k,
                   knot_y_k,
                   delta_k,
                   knot_x_kp1,
                   knot_y_kp1,
                   delta_kp1):

  delta_y = (knot_y_kp1 - knot_y_k)
  delta_x = (knot_x_kp1 - knot_x_k)
  delta_x = jnp.where(mask, delta_x, 1.0)
  s_k = delta_y/delta_x

  zeta = (x - knot_x_k)/delta_x
  onemz = (1 - zeta)
  z1mz = zeta*onemz

  # Return the output
  alpha = delta_y*(s_k*zeta**2 + delta_k*z1mz)
  beta = s_k + (delta_kp1 + delta_k - 2*s_k)*z1mz
  gamma = alpha/beta
  z = knot_y_k + gamma

  z = jnp.where(mask, z, x)

  dzdx = (s_k/beta)**2 * (delta_kp1*zeta**2 + 2*s_k*z1mz + delta_k*onemz**2)
  dzdx = jnp.where(mask, dzdx, 1.0)
  return z, dzdx

def inverse_spline(x,
                   mask,
                   knot_x_k,
                   knot_y_k,
                   delta_k,
                   knot_x_kp1,
                   knot_y_kp1,
                   delta_kp1):

  delta_y = (knot_y_kp1 - knot_y_k)
  delta_x = (knot_x_kp1 - knot_x_k)
  delta_x = jnp.where(mask, delta_x, 1.0)
  s_k = delta_y/delta_x

  knot_y_diff = x - knot_y_k
  term = knot_y_diff*(delta_kp1 + delta_k - 2*s_k)

  # Solve the quadratic
  b = delta_y*delta_k - term
  a = delta_y*s_k - b
  c = -s_k*knot_y_diff
  argument = b**2 - 4*a*c
  argument = jnp.where(mask, argument, 1.0) # Avoid nans
  d = -b - jnp.sqrt(argument)
  zeta = 2*c/d
  z1mz = zeta*(1 - zeta)

  # Solve for x
  z = zeta*delta_x + knot_x_k

  z = jnp.where(mask, z, x)

  beta = s_k + (delta_kp1 + delta_k - 2*s_k)*z1mz
  dzdx = (s_k/beta)**2 * (delta_kp1*zeta**2 + 2*s_k*z1mz + delta_k*(1 - zeta)**2)
  dzdx = jnp.where(mask, dzdx, 1.0)
  return z, dzdx

################################################################################################################

def find_knots(x, knot_x, knot_y, knot_derivs, inverse):

  # Need to find the knots for each dimension of x
  searchsorted = partial(jnp.searchsorted, side="right")
  take = jnp.take
  for i in range(len(x.shape)):
    searchsorted = jax.vmap(searchsorted)
    take = jax.vmap(take)

  if inverse == False:
    indices = searchsorted(knot_x, x) - 1
  else:
    indices = searchsorted(knot_y, x) - 1

  # Find the corresponding knots and derivatives
  knot_x_k = take(knot_x, indices)
  knot_y_k = take(knot_y, indices)
  delta_k = take(knot_derivs, indices)

  # We need the next indices too
  knot_x_kp1 = take(knot_x, indices + 1)
  knot_y_kp1 = take(knot_y, indices + 1)
  delta_kp1 = take(knot_derivs, indices + 1)
  args = knot_x_k, knot_y_k, delta_k, knot_x_kp1, knot_y_kp1, delta_kp1

  return args

################################################################################################################

def get_knot_params(settings, theta):
  K, min_width, min_height, min_derivative, bounds = settings

  # Get the individual parameters
  tw, th, td = theta[...,:K], theta[...,K:2*K], theta[...,2*K:]

  # Make the parameters fit the discription of knots
  tw, th = jax.nn.softmax(tw, axis=-1), jax.nn.softmax(th, axis=-1)
  tw = min_width + (1.0 - min_width*K)*tw
  th = min_height + (1.0 - min_height*K)*th
  td = min_derivative + util.square_plus(td)
  knot_x, knot_y = jnp.cumsum(tw, axis=-1), jnp.cumsum(th, axis=-1)

  # Pad the knots so that the first element is 0
  pad = [(0, 0)]*(len(td.shape) - 1) + [(1, 0)]
  knot_x = jnp.pad(knot_x, pad)
  knot_y = jnp.pad(knot_y, pad)

  # Scale by the bounds
  knot_x = (bounds[0][1] - bounds[0][0])*knot_x + bounds[0][0]
  knot_y = (bounds[1][1] - bounds[1][0])*knot_y + bounds[1][0]

  # This fails because there is not a transpose rule for lax.scatter
  # # Set the max and min values exactly
  # knot_x = knot_x.at[...,0].set(bounds[0][0])
  # knot_x = knot_x.at[...,-1].set(bounds[0][1])
  # knot_y = knot_y.at[...,0].set(bounds[1][0])
  # knot_y = knot_y.at[...,-1].set(bounds[1][1])

  # Pad the derivatives so that the first and last elts are 1
  pad = [(0, 0)]*(len(td.shape) - 1) + [(1, 1)]
  knot_derivs = jnp.pad(td, pad, constant_values=1)

  return knot_x, knot_y, knot_derivs

################################################################################################################

class RationalQuadraticSpline(Flow):

  def __init__(self,
               K: int=4,
               min_width: Optional[float]=1e-3,
               min_height: Optional[float]=1e-3,
               min_derivative: Optional[float]=1e-3,
               bounds: Sequence[float]=((-10.0, 10.0), (-10.0, 10.0)),
               **kwargs
  ):
    """
    """
    self.K              = K
    self.min_width      = min_width
    self.min_height     = min_height
    self.min_derivative = min_derivative
    self.bounds         = bounds

  @property
  def param_multiplier(self):
    return 3*self.K - 1

  def get_param_dim(self, dim):
    return self.param_multiplier*dim

  def get_params(self):
    return {"theta": self.theta}

  @property
  def coupling_param_keys(self):
    return ("theta",)

  def extract_coupling_params(self, theta):
    return (theta,)

  def __call__(self, x, params=None, inverse=False, rng_key=None, no_llc=False, **kwargs):

    if params is None:
      x_shape = x.shape[1:]
      self.theta = random.normal(rng_key, shape=x_shape[:-1] + (x_shape[-1]*self.param_multiplier,))
    else:
      self.theta = params["theta"]

    if self.theta.ndim == x.ndim - 1:
      # No coupling
      theta = self.theta.reshape(x.shape[1:] + (self.param_multiplier,))
      theta = jnp.broadcast_to(theta, x.shape[:1] + theta.shape)
    else:
      theta = self.theta.reshape(x.shape + (self.param_multiplier,))

    # Get the parameters
    settings = self.K, self.min_width, self.min_height, self.min_derivative, self.bounds
    knot_x, knot_y, knot_derivs = get_knot_params(settings, theta)

    # The relevant knot depends on if we are inverting or not
    if inverse == False:
      mask = (x > self.bounds[0][0] + 1e-5)&(x < self.bounds[0][1] - 1e-5)
      apply_fun = forward_spline
    else:
      mask = (x > self.bounds[1][0] + 1e-5)&(x < self.bounds[1][1] - 1e-5)
      apply_fun = inverse_spline

    args = find_knots(x, knot_x, knot_y, knot_derivs, inverse)

    z, dzdx = apply_fun(x, mask, *args)
    if no_llc == False:
      elementwise_log_det = jnp.log(dzdx)
    else:
      elementwise_log_det = jnp.zeros_like(dzdx)

    sum_axes = util.last_axes(x.shape[len(x.shape[:1]):])
    log_det = elementwise_log_det.sum(sum_axes)

    return z, log_det

################################################################################################################

def regular_test():

  K = 2

  rng_key = random.PRNGKey(0)
  x_shape = (4, 4, 2)
  x = random.normal(rng_key, shape=(2,) + x_shape)
  flow = RationalQuadraticSpline(K)
  z, log_det = flow(x, rng_key=rng_key)
  params = flow.get_params()

  reconstr, _ = flow(z, params, inverse=True)
  z2, _ = flow(reconstr, params, inverse=False)
  assert jnp.allclose(x, reconstr, atol=1e-5)

  def jac(x):
    flat_x, unflatten = ravel_pytree(x)
    def flat_call(flat_x):
      x = unflatten(flat_x)
      z, _ = flow(x[None], params=params)
      return z.ravel()

    return jax.jacobian(flat_call)(flat_x)

  J = jax.vmap(jac)(x)
  true_log_det = jnp.linalg.slogdet(J)[1]

  assert jnp.allclose(log_det, true_log_det)

def coupling_test():

  rng_key = random.PRNGKey(0)
  x = random.normal(rng_key, shape=(2, 4, 4, 2))

  K = 4
  theta = random.normal(rng_key, shape=x.shape[:-1] + ((3*K - 1)*x.shape[-1],))
  params = dict(theta=theta)

  flow = RationalQuadraticSpline(K)
  z, log_det = flow(x, params=params, rng_key=rng_key)
  reconstr, _ = flow(z, params, inverse=True)
  assert jnp.allclose(x, reconstr, atol=1e-5)

  def jac(x, params):
    flat_x, unflatten = ravel_pytree(x)
    def flat_call(flat_x):
      x = unflatten(flat_x)
      z, _ = flow(x[None], params=params)
      return z.ravel()

    return jax.jacobian(flat_call)(flat_x)

  J = jax.vmap(jac)(x, params)
  true_log_det = jnp.linalg.slogdet(J)[1]

  assert jnp.allclose(log_det, true_log_det)

if __name__ == "__main__":
  from debug import *

  regular_test()
  coupling_test()

  import pdb; pdb.set_trace()

  rng_key = random.PRNGKey(0)
  x = random.normal(rng_key, shape=(2, 3))
  bounds = ((-1.0, 1.0), (-1.0, 1.0))

  K = 3
  flow = RationalQuadraticSpline(K, bounds=bounds)
  z, log_det = flow(x, rng_key=rng_key)
  theta = flow.get_params()["theta"]

  import pdb; pdb.set_trace()