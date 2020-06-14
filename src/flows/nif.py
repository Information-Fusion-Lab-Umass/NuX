import jax.nn.initializers as jaxinit
import jax
from jax import random, jit, vmap
import jax.nn
import jax.numpy as jnp
from functools import partial, reduce
from jax.tree_util import tree_map
from jax.scipy.special import logsumexp
import src.util as util

################################################################################################################

def importance_sample_prior(prior_forward, prior_params, prior_state, z, sigma_ATA_chol, n_training_importance_samples, **kwargs):
    # Sample from N(z|\mu(x),\Sigma(x))
    key = kwargs.pop('key', None)
    if(key is not None):
        # Re-fill key for the rest of the flow
        k1, k2 = random.split(key, 2)
        kwargs['key'] = k2

        # See how many samples we should pull
        n_importance_samples = kwargs.get('n_importance_samples', n_training_importance_samples)

        # Sample from the posterior
        noise = random.normal(k1, (n_importance_samples,) + z.shape)
        z_samples = z[None,...] + jnp.dot(noise, sigma_ATA_chol.T)
    else:
        # We're only using the mean, but put it on an axis so that we can use vmap
        z_samples = z[None]

    # Compute the rest of the flow with the samples of z
    vmapped_forward = vmap(partial(prior_forward, prior_params, prior_state, **kwargs))
    log_pxs, zs, updated_prior_states = vmapped_forward(z_samples)
    return log_pxs, zs, updated_prior_states

################################################################################################################

def TallAffineDiagCov(flow, out_dim, A_init=jaxinit.glorot_normal(), b_init=jaxinit.normal(), name='unnamed'):
    """ Affine function to go up a dimension

        Args:
    """
    prior_init, prior_forward, prior_inverse = flow

    def init_fun(key, input_shape):
        # Assume that we have a flat input
        x_shape = input_shape
        output_shape = x_shape[:-1] + (out_dim,)
        z_dim, x_dim = out_dim, x_shape[-1]
        keys = random.split(key, 3)

        # Create the parameters for the NIF
        A = A_init(keys[0], (x_shape[-1], out_dim))
        b = b_init(keys[1], (x_shape[-1],))
        log_diag_cov = jnp.ones(input_shape[-1])*0.0

        # Create the flow parameters
        prior_name, prior_output_shape, prior_params, prior_state = prior_init(keys[2], output_shape)

        # Return everything
        params = ((A, b, log_diag_cov), prior_params)
        state = ((), prior_state)
        return (name, prior_name), prior_output_shape, params, state

    def forward(params, state, x, **kwargs):
        ((A, b, log_diag_cov), prior_params) = params
        _, prior_state = state

        # Get the terms to compute and sample from the posterior
        sigma = kwargs.get('sigma', 1.0)
        z, log_hx, sigma_ATA_chol = util.tall_affine_posterior_diag_cov(x, b, A, log_diag_cov, sigma)

        # Importance sample from N(z|\mu(x),\Sigma(x)) and compile the results
        log_pxs, zs, updated_prior_states = importance_sample_prior(prior_forward, prior_params, prior_state, z, sigma_ATA_chol, n_training_importance_samples, **kwargs)

        # Compile the results
        log_px = logsumexp(log_pxs, axis=0) - jnp.log(log_pxs.shape[0])
        z = jnp.mean(zs, axis=0) # Just average the state
        updated_prior_states = tree_map(lambda x:x[0], updated_prior_states)
        log_pz, z, updated_prior_states = log_px, z, updated_prior_states

        # Compute the final estimate of the integral
        log_px = log_pz + log_hx

        return log_px, z, ((), updated_prior_states)

    def inverse(params, state, z, **kwargs):
        ((A, b, log_diag_cov), prior_params) = params
        _, prior_state = state

        log_pz, z, updated_state = prior_inverse(prior_params, prior_state, z, **kwargs)

        # Compute Az + b
        # Don't need to sample because we already sampled from p(z)!!!!
        x = jnp.dot(z, A.T) + b
        key = kwargs.pop('key', None)
        if(key is not None):

            sigma = kwargs.get('sigma', 1.0)
            noise = random.normal(key, x.shape)*sigma

            x += noise*jnp.exp(0.5*log_diag_cov)

        # Compute N(x|Az + b, \Sigma).  This is just the log partition function.
        log_px = - 0.5*jnp.sum(log_diag_cov) - 0.5*x.shape[-1]*jnp.log(2*jnp.pi)
        return log_pz + log_px, x, ((), updated_state)

    return init_fun, forward, inverse

################################################################################################################

# NEED TO CHANGE THE STANDARD LAYERS TO JUST USE A SINGLE IMPORTANCE SAMPLE!!!
def IWA_grad(flow):
    init_fun, flow_forward, flow_inverse = flow

    @jax.custom_vjp
    def forward(params, state, x, **kwargs):
        return flow_forward(params, state, x, **kwargs)


    def forward_fwd(params, state, x, **kwargs):
        pass

# ################################################################################################################

# def importance_sample_prior(prior_forward, prior_params, prior_state, z, sigma_ATA_chol, n_training_importance_samples, **kwargs):
#     # Sample from N(z|\mu(x),\Sigma(x))
#     key = kwargs.pop('key', None)
#     if(key is not None):
#         # Re-fill key for the rest of the flow
#         k1, k2 = random.split(key, 2)
#         kwargs['key'] = k2

#         # See how many samples we should pull
#         n_importance_samples = kwargs.get('n_importance_samples', n_training_importance_samples)

#         # Sample from the posterior
#         noise = random.normal(k1, (n_importance_samples,) + z.shape)
#         z_samples = z[None,...] + jnp.dot(noise, sigma_ATA_chol.T)
#     else:
#         # We're only using the mean, but put it on an axis so that we can use vmap
#         z_samples = z[None]

#     # Compute the rest of the flow with the samples of z
#     vmapped_forward = vmap(partial(prior_forward, prior_params, prior_state, **kwargs))
#     log_pxs, zs, updated_prior_states = vmapped_forward(z_samples)

#     # Compile the results
#     log_px = logsumexp(log_pxs, axis=0) - jnp.log(log_pxs.shape[0])
#     z = jnp.mean(zs, axis=0) # Just average the state
#     updated_prior_states = tree_map(lambda x:x[0], updated_prior_states)
#     return log_px, z, updated_prior_states

# ################################################################################################################

# def TallAffineDiagCov(flow, out_dim, n_training_importance_samples=32, A_init=jaxinit.glorot_normal(), b_init=jaxinit.normal(), name='unnamed'):
#     """ Affine function to go up a dimension

#         Args:
#     """
#     _init_fun, _forward, _inverse = flow

#     def init_fun(key, input_shape):
#         x_shape = input_shape
#         output_shape = x_shape[:-1] + (out_dim,)
#         keys = random.split(key, 3)

#         x_dim = x_shape[-1]
#         z_dim = out_dim
#         A = A_init(keys[0], (x_shape[-1], out_dim))
#         b = b_init(keys[1], (x_shape[-1],))
#         flow_name, flow_output_shape, flow_params, flow_state = _init_fun(keys[2], output_shape, condition_shape)
#         log_diag_cov = jnp.ones(input_shape[-1])*0.0
#         params = ((A, b, log_diag_cov), flow_params)
#         state = ((), flow_state)
#         return (name, flow_name), flow_output_shape, params, state

#     def forward(params, state, x, **kwargs):
#         ((A, b, log_diag_cov), flow_params) = params
#         _, flow_state = state

#         # Get the terms to compute and sample from the posterior
#         sigma = kwargs.get('sigma', 1.0)
#         z, log_hx, sigma_ATA_chol = tall_affine_posterior_diag_cov(x, b, A, log_diag_cov, sigma)

#         # Importance sample from N(z|\mu(x),\Sigma(x)) and compile the results
#         log_pz, z, updated_flow_states = importance_sample_prior(_forward, flow_params, flow_state, z, sigma_ATA_chol, n_training_importance_samples, **kwargs)

#         # Compute the final estimate of the integral
#         log_px = log_pz + log_hx

#         return log_px, z, ((), updated_flow_states)

#     def inverse(params, state, z, **kwargs):
#         ((A, b, log_diag_cov), flow_params) = params
#         _, flow_state = state

#         log_pz, z, updated_state = _inverse(flow_params, flow_state, z, **kwargs)

#         # Compute Az + b
#         # Don't need to sample because we already sampled from p(z)!!!!
#         x = jnp.dot(z, A.T) + b
#         key = kwargs.pop('key', None)
#         if(key is not None):

#             sigma = kwargs.get('sigma', 1.0)
#             noise = random.normal(key, x.shape)*sigma

#             x += noise*jnp.exp(0.5*log_diag_cov)

#         # Compute N(x|Az + b, \Sigma).  This is just the log partition function.
#         log_px = - 0.5*jnp.sum(log_diag_cov) - 0.5*x.shape[-1]*jnp.log(2*jnp.pi)
#         return log_pz + log_px, x, ((), updated_state)

#     return init_fun, forward, inverse

################################################################################################################

def UpSample(repeats, name='unnamed'):
    # language=rst
    """
    Up sample by just repeating consecutive values over specified axes

    :param repeats - The number of times to repeat.  Pass in (2, 1, 2), for example, to repeat twice over
                     the 0th axis, no repeats over the 1st axis, and twice over the 2nd axis
    """
    full_repeats = None
    def init_fun(key, input_shape):
        nonlocal full_repeats
        full_repeats = [repeats[i] if i < len(repeats) else 1 for i in range(len(input_shape))]
        output_shape = []
        for s, r in zip(input_shape, full_repeats):
            assert s%r == 0
            output_shape.append(s//r)
        output_shape = tuple(output_shape)
        log_diag_cov = jnp.ones(input_shape)*5
        b = jnp.zeros(input_shape)

        params, state = (log_diag_cov, b), ()
        return name, output_shape, params, state

    def forward(params, state, x, **kwargs):
        if(x.ndim == 4):
            return vmap(partial(forward, params, state, **kwargs), in_axes=(0, 0, None))(log_px, x, condition)

        log_diag_cov, b = params
        log_diag_cov = -jax.nn.softplus(log_diag_cov) - 3.0 # Limit the amount of noise

        # Compute the posterior and the manifold penalty
        z_mean, log_hx, rm_diag = upsample_posterior(x, b, log_diag_cov, full_repeats)

        # Sample z
        key = kwargs.pop('key', None)
        if(key is not None):
            noise = random.normal(key, z_mean.shape)
            z = z_mean + noise/jnp.sqrt(rm_diag)
        else:
            z = z_mean

        return log_px + log_hx, z, state

    def inverse(params, state, log_pz, z, **kwargs):
        if(z.ndim == 4):
            return vmap(partial(inverse, params, state, **kwargs), in_axes=(0, 0, None))(log_pz, z, condition)

        log_diag_cov, b = params
        log_diag_cov = -jax.nn.softplus(log_diag_cov) - 3.0 # Limit the amount of noise

        # x ~ N(x|Az + b, Sigma)

        x_mean = upsample(full_repeats, z) + b

        key = kwargs.pop('key', None)
        if(key is not None):
            noise = jnp.exp(0.5*log_diag_cov)*random.normal(key, x_mean.shape)
            x = x_mean + noise
        else:
            noise = jnp.zeros_like(x_mean)
            x = x_mean

        log_px = -0.5*jnp.sum(noise*jnp.exp(-0.5*log_diag_cov)*noise) - 0.5*jnp.sum(log_diag_cov) - 0.5*x.shape[-1]*jnp.log(2*jnp.pi)

        return log_pz + log_px, x, state

    return init_fun, forward, inverse

################################################################################################################
