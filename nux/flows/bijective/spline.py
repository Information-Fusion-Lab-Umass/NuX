import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap, jit
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence
from nux.internal.layer import InvertibleLayer
import nux.util as util
from nux.flows.bijective.coupling_base import Elementwise
import nux.networks as net

__all__ = ["RationalQuadraticSpline"]

def get_knot_params(theta: jnp.ndarray,
                    K: int,
                    min_width: Optional[float]=1e-3,
                    min_height: Optional[float]=1e-3,
                    min_derivative: Optional[float]=1e-3,
                    bounds: Sequence[float]=((-3.0, 3.0), (-3.0, 3.0))
  ):
  # Get the individual parameters
  tw, th, td = jnp.split(theta, jnp.array([K, 2*K]), axis=-1)

  # Make the parameters fit the discription of knots
  tw, th = jax.nn.softmax(tw), jax.nn.softmax(th)
  tw = min_width + (1.0 - min_width*K)*tw
  th = min_height + (1.0 - min_height*K)*th
  td = min_derivative + util.proximal_relu(td)
  # td = min_derivative + jax.nn.softplus(td)
  knot_x, knot_y = jnp.cumsum(tw, axis=-1), jnp.cumsum(th, axis=-1)

  # Pad the knots so that the first element is 0
  pad = [(0, 0)]*(len(td.shape) - 1) + [(1, 0)]
  knot_x = jnp.pad(knot_x, pad)
  knot_y = jnp.pad(knot_y, pad)

  # Scale by the bounds
  knot_x = (bounds[0][1] - bounds[0][0])*knot_x + bounds[0][0]
  knot_y = (bounds[1][1] - bounds[1][0])*knot_y + bounds[1][0]

  # Pad the derivatives so that the first and last elts are 1
  pad = [(0, 0)]*(len(td.shape) - 1) + [(1, 1)]
  knot_derivs = jnp.pad(td, pad, constant_values=1)

  return knot_x, knot_y, knot_derivs

def spline(theta: jnp.ndarray,
           inputs,
           K: int,
           sample: bool,
           min_width: Optional[float]=1e-3,
           min_height: Optional[float]=1e-3,
           min_derivative: Optional[float]=1e-3,
           bounds: Sequence[float]=((-3.0, 3.0), (-3.0, 3.0))
  ):

  knot_x, knot_y, knot_derivs = get_knot_params(theta,
                                                K,
                                                min_width=min_width,
                                                min_height=min_height,
                                                min_derivative=min_derivative,
                                                bounds=bounds)
  eps = 1e-5

  if sample == False:
    mask = (inputs > bounds[0][0] + eps)&(inputs < bounds[0][1] - eps)
  else:
    mask = (inputs > bounds[1][0] + eps)&(inputs < bounds[1][1] - eps)

  # Find the knot index for each data point
  searchsorted = vmap(partial(jnp.searchsorted, side="right"))
  take = vmap(jnp.take)

  if sample == False:
    indices = searchsorted(knot_x, inputs) - 1
  else:
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

  if sample == False:
    zeta = (inputs - x_k)/dx
    z1mz = zeta*(1 - zeta)

    # Return the output
    numerator = dy*(s_k*zeta**2 + delta_k*z1mz)
    denominator = s_k + (delta_kp1 + delta_k - 2*s_k)*z1mz
    outputs = y_k + numerator/denominator
  else:
    y_diff = inputs - y_k
    term = y_diff*(delta_kp1 + delta_k - 2*s_k)

    # Solve the quadratic
    a = dy*(s_k - delta_k) + term
    b = dy*delta_k - term
    c = -s_k*y_diff
    zeta = 2*c/(-b - jnp.sqrt(b**2 - 4*a*c))
    z1mz = zeta*(1 - zeta)

    denominator = s_k + (delta_kp1 + delta_k - 2*s_k)*z1mz

    # Solve for x
    outputs = zeta*dx + x_k

  # Calculate the log Jacobian determinant
  deriv_numerator = s_k**2*(delta_kp1*zeta**2 + 2*s_k*z1mz + delta_k*(1 - zeta)**2)
  deriv_denominator = denominator**2
  deriv = deriv_numerator/deriv_denominator

  derivs_for_logdet = jnp.where(mask, deriv, 1.0)
  outputs = jnp.where(mask, outputs, inputs)

  elementwise_log_det = jnp.log(jnp.abs(derivs_for_logdet))

  return outputs, elementwise_log_det

################################################################################################################

class RationalQuadraticSpline(Elementwise):

  def __init__(self,
               K: int=4,
               bounds: Sequence[float]=((-10.0, 10.0), (-10.0, 10.0)),
               create_network: Optional[Callable]=None,
               axis: Optional[int]=-1,
               coupling: bool=True,
               split_kind: str="channel",
               masked: bool=False,
               use_condition: bool=False,
               condition_method: str="nin",
               apply_to_both_halves: Optional[bool]=True,
               network_kwargs: Optional[Mapping]=None,
               name: str="rq_neural_spline",
               **kwargs
  ):
    """ Neural spline flow https://arxiv.org/pdf/1906.04032.pdf
    Args:
      K                : Number of bins to use
      bounds           : The interval to apply the spline to
      create_network   : Function to create the conditioner network.  Should accept a tuple
                         specifying the output shape.  See coupling_base.py
      network_kwargs   : Dictionary with settings for the default network (see get_default_network in util.py)
      name             : Optional name for this module.
    """
    super().__init__(create_network=create_network,
                     axis=-1,
                     coupling=coupling,
                     split_kind=split_kind,
                     masked=masked,
                     use_condition=use_condition,
                     condition_method=condition_method,
                     name=name,
                     apply_to_both_halves=apply_to_both_halves,
                     network_kwargs=network_kwargs,
                     **kwargs)
    self.K              = K
    self.bounds         = bounds
    self.forward_spline = partial(spline, K=K, sample=False, bounds=bounds)
    self.inverse_spline = partial(spline, K=K, sample=True, bounds=bounds)

  def get_out_shape(self, x):
    x_shape = x.shape[len(self.batch_shape):]
    out_dim = x_shape[-1]*(3*self.K - 1)
    return x_shape[:-1] + (out_dim,)

  def transform(self, x, params=None, sample=False, rng=None, **kwargs):
    x_flat = x.reshape(self.batch_shape + (-1,))
    param_dim = (3*self.K - 1)
    if params is None:
      x_shape = x_flat.shape[len(self.batch_shape):]
      theta = hk.get_parameter("theta", shape=(x_flat.shape[-1],) + (param_dim,), dtype=x_flat.dtype, init=hk.initializers.RandomNormal())
      in_axes = (None, 0)
    else:
      theta = params.reshape(self.batch_shape + (x_flat.shape[-1],) + (param_dim,))
      in_axes = (0, 0)

    if sample == False:
      z, elementwise_log_det = self.auto_batch(self.forward_spline, in_axes=in_axes)(theta, x_flat)
    else:
      z, elementwise_log_det = self.auto_batch(self.inverse_spline, in_axes=in_axes)(theta, x_flat)

    z = z.reshape(x.shape)
    elementwise_log_det = elementwise_log_det.reshape(x.shape)

    return z, elementwise_log_det
