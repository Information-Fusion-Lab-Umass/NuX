import jax
from jax import vmap, jit, random
import jax.numpy as jnp
from functools import partial
import haiku as hk
import nux.util as util
import nux.flows
import nux.flows.base as base

@partial(jit, static_argnums=(1,))
def get_knot_params(theta, K, min_width=1e-3, min_height=1e-3, min_derivative=1e-3, bounds=((-3.0, 3.0), (-3.0, 3.0))):

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

def spline(theta, inputs, K, reverse, min_width=1e-3, min_height=1e-3, min_derivative=1e-3, bounds=((-3.0, 3.0), (-3.0, 3.0))):

    knot_x, knot_y, knot_derivs = get_knot_params(theta,
                                                  K,
                                                  min_width=min_width,
                                                  min_height=min_height,
                                                  min_derivative=min_derivative,
                                                  bounds=bounds)

    eps = 1e-5

    if(reverse == False):
        mask = (inputs > bounds[0][0] + eps)&(inputs < bounds[0][1] - eps)
    else:
        mask = (inputs > bounds[1][0] + eps)&(inputs < bounds[1][1] - eps)

    # Find the knot index for each data point
    searchsorted = vmap(partial(jnp.searchsorted, side='right'))
    take = vmap(jnp.take)

    if(reverse == False):
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

    if(reverse == False):
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

@base.auto_batch
def NeuralSpline(K, network=None, hidden_layer_sizes=[1024]*4, bounds=((-10.0, 10.0), (-10.0, 10.0)), name='unnamed'):
    x1_dim = None

    forward_spline = jit(partial(spline, K=K, reverse=False, bounds=bounds))
    inverse_spline = jit(partial(spline, K=K, reverse=True, bounds=bounds))

    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        x = inputs['x']

        x1, x2 = jnp.split(x, jnp.array([x1_dim]), axis=-1)

        if(reverse == False):
            theta1 = params['theta']
            z1, log_det1 = forward_spline(theta1, x1)

            theta2 = network.apply(params['hk_params'], None, x1).reshape((-1, 3*K - 1))
            z2, log_det2 = forward_spline(theta2, x2)
        else:
            theta1 = params['theta']
            z1, log_det1 = inverse_spline(theta1, x1)

            theta2 = network.apply(params['hk_params'], None, z1).reshape((-1, 3*K - 1))
            z2, log_det2 = inverse_spline(theta2, x2)

        z = jnp.concatenate([z1, z2], axis=-1)

        outputs = {'x': z, 'log_det': log_det1 + log_det2}
        return outputs, state

    def create_params_and_state(key, input_shapes):
        x_shape = input_shapes['x']
        nonlocal x1_dim
        x1_dim = x_shape[-1]//2
        x2_dim = x_shape[-1] - x1_dim
        network_out_shape = (x2_dim*(3*K - 1),)

        theta = random.normal(key, (x1_dim, 3*K - 1))

        nonlocal network
        if(network is None):
            network = hk.transform(lambda x, **kwargs: util.SimpleMLP(network_out_shape, hidden_layer_sizes, is_additive=True)(x, **kwargs), apply_rng=True)

        params = {'hk_params': network.init(key, jnp.zeros((x1_dim,))), 'theta': theta}
        state = {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

__all__ = ['NeuralSpline']
