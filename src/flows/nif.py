import jax.nn.initializers as jaxinit
import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from functools import partial, reduce
from jax.tree_util import tree_map
from jax.scipy.special import logsumexp
import src.util as util
import src.flows.base as base

################################################################################################################

@base.auto_batch
def TallAffineDiagCov(out_dim, A_init=jaxinit.glorot_normal(), b_init=jaxinit.normal(), name='tall_affine_diag_cov'):
    """ Affine function to go up a dimension

        Args:
    """
    def forward(params, state, inputs, **kwargs):
        x = inputs['x']
        A, b, log_diag_cov = params['A'], params['b'], params['log_diag_cov']
        sigma = state['sigma']
        key = kwargs.pop('key', None)

        # Sample from the stochastic inverse q(z|x)
        z, log_hx, sigma_ATA_chol = util.tall_affine_posterior_diag_cov(x, b, A, log_diag_cov, sigma)

        if(key is not None):
            n_importance_samples = kwargs.get('n_importance_samples', None)
            if(n_importance_samples is None):
                noise = random.normal(key, z.shape)
                z += sigma_ATA_chol@noise
                log_det = log_hx
            else:
                noise = random.normal(key, (n_importance_samples,) + z.shape)
                z += jnp.einsum('ij,bj->bi', sigma_ATA_chol, noise)
                log_det = jnp.broadcast_to(log_hx, (n_importance_samples,))
        else:
            # Treat this as an injective flow and use the log determinant
            log_det = -0.5*jnp.linalg.slogdet(A.T@A)[1]

        # Compute the reconstruction error and stochastic inverse likelihood
        if(key is not None):
            if(n_importance_samples is not None):
                mean = jnp.einsum('ij,bj->bi', A, z) + b
                log_pxgz = vmap(util.gaussian_diag_cov_logpdf, in_axes=(None, 0, None))(x, mean, log_diag_cov)
                log_qzgx = vmap(util.gaussian_chol_cov_logpdf, in_axes=(0, 0, None))(noise, jnp.zeros_like(noise), sigma_ATA_chol)
            else:
                mean = A@z + b
                log_pxgz = util.gaussian_diag_cov_logpdf(x, mean, log_diag_cov)
                log_qzgx = util.gaussian_chol_cov_logpdf(noise, jnp.zeros_like(noise), sigma_ATA_chol)
        else:
            log_pxgz = 0.0
            log_qzgx = 0.0

        outputs = {'x': z, 'log_det': log_det, 'log_pxgz': log_pxgz, 'log_qzgx': log_qzgx}
        return outputs, state

    def inverse(params, state, inputs, **kwargs):
        z = inputs['x']
        A, b, log_diag_cov = params['A'], params['b'], params['log_diag_cov']
        sigma = state['sigma']
        key = kwargs.pop('key', None)

        # Transform onto the manifold
        x = A@z + b

        # This is used for testing!
        injected_x = kwargs.get('injected_x', None)
        if(injected_x is not None):
            key = None

        # Sample from N(x|Az + b, Sigma)
        if(key is not None):
            noise = random.normal(key, x.shape)
            x += noise*jnp.exp(0.5*log_diag_cov)*sigma

            # Compute the likelihood p(x|z)
            log_det = util.gaussian_diag_cov_logpdf(noise, jnp.zeros_like(noise), log_diag_cov)
        else:
            # Treat this as an injective flow and use the log determinant
            log_det = -0.5*jnp.linalg.slogdet(A.T@A)[1]

        outputs = {'x': x, 'log_det': log_det}

        # Evaluate the log pdf of a given x.  This is used for testing.
        if(injected_x is not None):
            assert key is None, 'Do not pass a key!'
            log_px = util.gaussian_diag_cov_logpdf(injected_x, x, log_diag_cov)
            outputs['log_det'] = log_px

        return outputs, state

    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        if(reverse == False):
            return forward(params, state, inputs, **kwargs)
        return inverse(params, state, inputs, **kwargs)

    def create_params_and_state(key, input_shapes):
        x_shape = input_shapes['x']
        output_shape = x_shape[:-1] + (out_dim,)
        z_dim, x_dim = out_dim, x_shape[-1]
        keys = random.split(key, 3)

        # Create the parameters for the NIF
        A = A_init(keys[0], (x_shape[-1], out_dim))
        b = b_init(keys[1], (x_shape[-1],))*0.0
        log_diag_cov = jnp.zeros(x_shape[-1])

        # Create the flow parameters
        params = {'A': A, 'b': b, 'log_diag_cov': log_diag_cov}
        state = {'sigma': 1.0}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

# def UpSample(repeats, name='upsample'):
#     # language=rst
#     """
#     Up sample by just repeating consecutive values over specified axes

#     :param repeats - The number of times to repeat.  Pass in (2, 1, 2), for example, to repeat twice over
#                      the 0th axis, no repeats over the 1st axis, and twice over the 2nd axis
#     """
#     full_repeats = None
#     def init_fun(key, input_shape):
#         nonlocal full_repeats
#         full_repeats = [repeats[i] if i < len(repeats) else 1 for i in range(len(input_shape))]
#         output_shape = []
#         for s, r in zip(input_shape, full_repeats):
#             assert s%r == 0
#             output_shape.append(s//r)
#         output_shape = tuple(output_shape)
#         log_diag_cov = jnp.ones(input_shape)*5
#         b = jnp.zeros(input_shape)

#         params, state = (log_diag_cov, b), ()
#         return name, output_shape, params, state

#     def forward(params, state, x, **kwargs):
#         if(x.ndim == 4):
#             return vmap(partial(forward, params, state, **kwargs), in_axes=(0, 0, None))(log_px, x, condition)

#         log_diag_cov, b = params
#         log_diag_cov = -jax.nn.softplus(log_diag_cov) - 3.0 # Limit the amount of noise

#         # Compute the posterior and the manifold penalty
#         z_mean, log_hx, rm_diag = upsample_posterior(x, b, log_diag_cov, full_repeats)

#         # Sample z
#         key = kwargs.pop('key', None)
#         if(key is not None):
#             noise = random.normal(key, z_mean.shape)
#             z = z_mean + noise/jnp.sqrt(rm_diag)
#         else:
#             z = z_mean

#         return log_px + log_hx, z, state

#     def inverse(params, state, log_pz, z, **kwargs):
#         if(z.ndim == 4):
#             return vmap(partial(inverse, params, state, **kwargs), in_axes=(0, 0, None))(log_pz, z, condition)

#         log_diag_cov, b = params
#         log_diag_cov = -jax.nn.softplus(log_diag_cov) - 3.0 # Limit the amount of noise

#         # x ~ N(x|Az + b, Sigma)

#         x_mean = upsample(full_repeats, z) + b

#         key = kwargs.pop('key', None)
#         if(key is not None):
#             noise = jnp.exp(0.5*log_diag_cov)*random.normal(key, x_mean.shape)
#             x = x_mean + noise
#         else:
#             noise = jnp.zeros_like(x_mean)
#             x = x_mean

#         log_px = -0.5*jnp.sum(noise*jnp.exp(-0.5*log_diag_cov)*noise) - 0.5*jnp.sum(log_diag_cov) - 0.5*x.shape[-1]*jnp.log(2*jnp.pi)

#         return log_pz + log_px, x, state

#     return init_fun, forward, inverse

################################################################################################################

__all__ = ['TallAffineDiagCov']

# __all__ = ['TallAffineDiagCov',
#            'UpSample']