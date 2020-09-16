import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap, jit
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence
from nux.flows.base import *
import nux.util as util
import nux.networks as net

__all__ = ["NeuralSpline"]

@partial(jit, static_argnums=(1,))
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
  td = min_derivative + jax.nn.softplus(td)
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

  if(sample == False):
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

  log_det = jnp.log(jnp.abs(derivs_for_logdet)).sum(axis=-1)

  return outputs, log_det

################################################################################################################

class NeuralSpline(AutoBatchedLayer):

  def __init__(self,
               K: int,
               create_network: Optional[Callable]=None,
               layer_sizes: Optional[Sequence[int]]=[1024]*4,
               parameter_norm: Optional[str]=None,
               bounds: Sequence[float]=((-4.0, 4.0), (-4.0, 4.0)),
               name: str="rq_spline",
               **kwargs
  ):
    super().__init__(name=name, **kwargs)
    self.K              = K
    self.layer_sizes    = layer_sizes
    self.bounds         = bounds
    self.create_network = create_network
    self.parameter_norm = parameter_norm

    self.forward_spline = jit(partial(spline, K=K, sample=False, bounds=bounds))
    self.inverse_spline = jit(partial(spline, K=K, sample=True, bounds=bounds))

  def get_network(self, shape):
    if self.create_network is not None:
      return self.create_network(shape)
    if len(shape) == 1:
      out_dim = shape[-1]
      return net.MLP(out_dim=out_dim,
                     layer_sizes=self.layer_sizes,
                     parameter_norm=self.parameter_norm,
                     nonlinearity="relu")
    else:
      assert 0, "Currently only implemented for 1d inputs"

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    x1, x2 = jnp.split(x, jnp.array([x.shape[-1]//2]), axis=-1)
    param_dim = 3*self.K - 1

    # Define the network
    network_out_shape = (x2.shape[-1]*param_dim,)
    network = self.get_network(network_out_shape)

    # Define the extra parameters for the other half of the input
    theta1 = hk.get_parameter("theta1", shape=(x1.shape[-1], param_dim), dtype=x.dtype, init=hk.initializers.RandomNormal())

    if sample == False:
      # Run the first part of the spline
      z1, log_det1 = self.forward_spline(theta1, x1)

      # Run the second part
      theta2 = network(x1).reshape((-1, param_dim))
      z2, log_det2 = self.forward_spline(theta2, x2)
    else:
      # Run the first part of the spline
      z1, log_det1 = self.inverse_spline(theta1, x1)

      # Run the second part and condition on the result of the first part
      theta2 = network(z1).reshape((-1, param_dim))
      z2, log_det2 = self.inverse_spline(theta2, x2)

    z = jnp.concatenate([z1, z2], axis=-1)

    outputs = {"x": z, "log_det": log_det1 + log_det2}

    return outputs

