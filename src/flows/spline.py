import jax
from jax import vmap, jit
import jax.numpy as jnp
from functools import partial
import haiku as hk
import src.util as util

@partial(jit, static_argnums=(0, 3))
def get_knot_parameters(apply_fun, params, x, K, min_width=1e-3, min_height=1e-3, min_derivative=1e-3):
    # Create the entire set of parameters
    theta = apply_fun(params, x)
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
    if(inputs.ndim == 2):
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
    if(inputs.ndim == 2):
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

def NeuralSpline(K, network=None, hidden_layer_sizes=[1024]*4, name='unnamed'):
    x1_dim = None

    def init_fun(key, input_shape):
        nonlocal x1_dim
        x1_dim = input_shape[-1]//2
        x2_dim = input_shape[-1] - x1_dim
        network_out_shape = (x2_dim*(3*K - 1),)

        nonlocal network
        if(network is None):
            network = hk.transform(lambda x, **kwargs: util.SimpleMLP(network_out_shape, hidden_layer_sizes, 'affine')(x, **kwargs))

        params = network.init(key, jnp.zeros((x1_dim,)))

        return name, input_shape, params, ()

    def forward(params, state, x, **kwargs):
        x1, x2 = jnp.split(x, jnp.array([x1_dim]), axis=-1)
        knot_x, knot_y, knot_derivs = get_knot_parameters(network.apply, params, x1, K)
        z2, log_det = spline_forward(knot_x, knot_y, knot_derivs, x2)

        z = jnp.concatenate([x1, z2], axis=-1)
        return log_det, z, state

    def inverse(params, state, z, **kwargs):
        z1, z2 = jnp.split(z, jnp.array([x1_dim]), axis=-1)
        knot_x, knot_y, knot_derivs = get_knot_parameters(network.apply, params, z1, K)
        x2, log_det = spline_inverse(knot_x, knot_y, knot_derivs, z2)

        x = jnp.concatenate([z1, x2], axis=-1)
        return log_det, x, state

    return init_fun, forward, inverse

################################################################################################################

__all__ = ['NeuralSpline']
