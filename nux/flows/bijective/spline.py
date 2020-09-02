import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap, jit
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence
from nux.flows.base import *
import nux.util as util

@partial(jit, static_argnums=(0, 2))
def get_knot_parameters(network, x, K, min_width=1e-3, min_height=1e-3, min_derivative=1e-3):
  # Create the entire set of parameters
  theta = network(x)
  theta = theta.reshape((-1, 3*K - 1))

  # Get the individual parameters
  tw, th, td = jnp.split(theta, jnp.array([K, 2*K]), axis=-1)

  # Make the parameters fit the discription of knots
  tw, th = jax.nn.softmax(tw), jax.nn.softmax(th)
  tw = min_width + (1.0 - min_width*K)*tw
  th = min_height + (1.0 - min_height*K)*th
  td = min_derivative + jax.nn.softplus(td)
  knot_x, knot_y = jnp.cumsum(tw, axis=-1), jnp.cumsum(th, axis=-1)

  # Pad the knots so that the first element is 0
  pad = [(0, 0)]*(len(td.shape) - 1) + [(1, 0)]
  knot_x = jnp.pad(knot_x, pad)
  knot_y = jnp.pad(knot_y, pad)

  # Pad the derivatives so that the first and last elts are 1
  pad = [(0, 0)]*(len(td.shape) - 1) + [(1, 1)]
  knot_derivs = jnp.pad(td, pad, constant_values=1)

  return knot_x, knot_y, knot_derivs

@jit
def spline_forward(knot_x, knot_y, knot_derivs, inputs):
  eps = 1e-5
  mask = (inputs > eps)&(inputs < 1.0 - eps)

  # Find the knot index for each data point
  vmapper = lambda f: vmap(f)
  searchsorted = vmapper(partial(jnp.searchsorted, side='right'))
  take = vmap(jnp.take)
  if inputs.ndim == 2:
    searchsorted = vmapper(searchsorted)
    take = vmap(take)

  indices = searchsorted(knot_x, inputs) - 1

  # Find the corresponding knots and derivatives
  x_k = take(knot_x, indices)
  y_k = take(knot_y, indices)
  delta_k = take(knot_derivs, indices)

  # We need the next indices too
  x_kp1 = take(knot_x, indices + 1)
  y_kp1 = take(knot_y, indices + 1)
  delta_kp1 = take(knot_derivs, indices + 1)

  # Some more values we need
  dy = (y_kp1 - y_k)
  dx = (x_kp1 - x_k)
  dx = jnp.where(mask, dx, 1.0) # Need this otherwise we can get nans in gradients
  s_k = dy/dx
  zeta = (inputs - x_k)/dx
  z1mz = zeta*(1 - zeta)

  # Return the output
  numerator = dy*(s_k*zeta**2 + delta_k*z1mz)
  denominator = s_k + (delta_kp1 + delta_k - 2*s_k)*z1mz
  outputs = y_k + numerator/denominator

  # Calculate the log Jacobian determinant
  deriv_numerator = s_k**2*(delta_kp1*zeta**2 + 2*s_k*z1mz + delta_k*(1 - zeta)**2)
  deriv_denominator = (s_k + (delta_kp1 + delta_k - 2*s_k)*z1mz)**2
  deriv = deriv_numerator/deriv_denominator

  derivs_for_logdet = jnp.where(mask, deriv, 1.0)
  outputs = jnp.where(mask, outputs, inputs)

  log_det = jnp.log(jnp.abs(derivs_for_logdet)).sum(axis=-1)

  return outputs, log_det

@jit
def spline_inverse(knot_x, knot_y, knot_derivs, inputs):
  eps = 1e-5
  mask = (inputs > eps)&(inputs < 1.0 - eps)

  # Find the knot index for each data point
  vmapper = lambda f: vmap(f)
  searchsorted = vmapper(partial(jnp.searchsorted, side='right'))
  take = vmap(jnp.take)
  if inputs.ndim == 2:
    searchsorted = vmapper(searchsorted)
    take = vmap(take)

  indices = searchsorted(knot_y, inputs) - 1

  # Find the corresponding knots and derivatives
  x_k = take(knot_x, indices)
  y_k = take(knot_y, indices)
  delta_k = take(knot_derivs, indices)

  # We need the next indices too
  x_kp1 = take(knot_x, indices + 1)
  y_kp1 = take(knot_y, indices + 1)
  delta_kp1 = take(knot_derivs, indices + 1)

  # Some more values we need
  dy = (y_kp1 - y_k)
  dx = (x_kp1 - x_k)
  dx = jnp.where(mask, dx, 1.0) # Need this otherwise we can get nans in gradients
  s_k = dy/dx
  y_diff = inputs - y_k
  term = y_diff*(delta_kp1 + delta_k - 2*s_k)

  # Solve the quadratic
  a = dy*(s_k - delta_k) + term
  b = dy*delta_k - term
  c = -s_k*y_diff
  zeta = 2*c/(-b - jnp.sqrt(b**2 - 4*a*c))
  z1mz = zeta*(1 - zeta)
  # Solve for x
  outputs = zeta*dx + x_k

  # Calculate the log Jacobian determinant
  deriv_numerator = s_k**2*(delta_kp1*zeta**2 + 2*s_k*z1mz + delta_k*(1 - zeta)**2)
  deriv_denominator = (s_k + (delta_kp1 + delta_k - 2*s_k)*z1mz)**2
  deriv = deriv_numerator/deriv_denominator

  derivs_for_logdet = jnp.where(mask, deriv, 1.0)
  outputs = jnp.where(mask, outputs, inputs)

  log_det = jnp.log(jnp.abs(derivs_for_logdet)).sum(axis=-1)

  return outputs, log_det

################################################################################################################

class NeuralSpline(AutoBatchedLayer):

  def __init__(self,
               K: int,
               create_network: Optional[Callable]=None,
               hidden_layer_sizes: Optional[Sequence[int]]=[1024]*4,
               name: str="rq_spline",
               **kwargs
  ):
    super().__init__(name=name, **kwargs)
    self.K                  = K
    self.hidden_layer_sizes = hidden_layer_sizes
    self.create_network     = None

  def get_network(self, shape):
    if self.create_network is not None:
      return self.create_network(shape)
    if len(shape) == 1:
      return util.SimpleMLP(shape, self.hidden_layer_sizes, is_additive=True)
    else:
      assert 0, 'Currently only implemented for 1d inputs'

  def call(self, inputs: Mapping[str, jnp.ndarray], sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    x1, x2 = jnp.split(x, jnp.array([x.shape[-1]//2]), axis=-1)
    network_out_shape = (x2.shape[-1]*(3*self.K - 1),)
    network = self.get_network(network_out_shape)

    knot_x, knot_y, knot_derivs = get_knot_parameters(network, x1, self.K)
    if sample == False:
      z2, log_det = spline_forward(knot_x, knot_y, knot_derivs, x2)
    else:
      z2, log_det = spline_inverse(knot_x, knot_y, knot_derivs, x2)

    z = jnp.concatenate([x1, z2], axis=-1)
    outputs = {"x": z, "log_det": log_det}
    return outputs

################################################################################################################

__all__ = ['NeuralSpline']
