import jax.nn.initializers as jaxinit
import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from functools import partial, reduce
from jax.tree_util import tree_map
from jax.scipy.special import logsumexp
import nux.util as util
import nux.flows.base as base
import haiku as hk

################################################################################################################

def every_other_element(x):
    # language=rst
    """
    Re-arrange a vector so that the even elements come first, then the odd elements
    """
    assert x.ndim == 1
    dim_x = x.shape[0]
    y = jnp.pad(x, (0, 1)) if dim_x%2 == 1 else x

    dim_y = y.shape[0]
    y = y.reshape((-1, 2)).T.reshape(dim_y)

    return y[:-1] if dim_x%2 == 1 else y

def every_other_index(N):
    return every_other_element(jnp.arange(N))

def remap_indices(indices, N):
    return jnp.array([list(indices).index(i) for i in range(N)])

################################################################################################################

@jit
def stochastic_inverse(x, b, A, log_diag_cov, s):
    """
    Compute N(z|mu(x), Sigma(x)) and h(x).
    """
    # In case we want to change the noise model.  This equation corresponds
    # to how we are changing noise in the inverse section
    log_diag_cov = log_diag_cov + 2*jnp.log(s)

    diag_cov = jnp.exp(log_diag_cov)
    xminusb = x - b

    # Find the pseudo inverse and the projection
    ATSA = A.T/diag_cov@A
    ATSA_inv = jnp.linalg.inv(ATSA)
    z = jnp.dot(xminusb, (ATSA_inv@A.T/diag_cov).T)
    x_proj = jnp.dot(z, A.T)/diag_cov

    # Get the terms that don't depend on z
    dim_x, dim_z = A.shape
    log_hx = -0.5*jnp.sum(xminusb*(xminusb/diag_cov - x_proj), axis=-1)
    log_hx -= 0.5*jnp.linalg.slogdet(ATSA)[1]
    log_hx -= 0.5*log_diag_cov.sum()
    log_hx -= 0.5*(dim_x - dim_z)*jnp.log(2*jnp.pi)

    # For sampling
    sigma_ATA_chol = jnp.linalg.cholesky(ATSA_inv)

    return z, log_hx, sigma_ATA_chol

################################################################################################################

@jit
def sample_stochastic_inverse(x, A, b, log_diag_cov, s, key):
    # language=rst
    """
    Sample from the stochastic inverse of N(x|Az + b, diag(exp(log_diag_cov)))
    """

    # Sample from the stochastic inverse q(z|x)
    z, log_hx, sigma_ATA_chol = stochastic_inverse(x, b, A, log_diag_cov, s)

    # Sample z and compute the manifold term
    if(key is not None):
        noise = sigma_ATA_chol@random.normal(key, z.shape)
        z += noise
        log_det = log_hx
    else:
        # Treat this as an injective flow and use the log determinant
        log_det = -0.5*jnp.linalg.slogdet(A.T@A)[1]

    # Compute the reconstruction error and stochastic inverse likelihood
    if(key is not None):
        mean = A@z + b
        log_pxgz = util.gaussian_diag_cov_logpdf(x, mean, log_diag_cov)
        log_qzgx = util.gaussian_chol_cov_logpdf(noise, jnp.zeros_like(noise), sigma_ATA_chol)
    else:
        log_pxgz = 0.0
        log_qzgx = 0.0

    outputs = {'x': z, 'log_det': log_det, 'log_pxgz': log_pxgz, 'log_qzgx': log_qzgx}
    return outputs

@jit
def generate(z, A, b, log_diag_cov, s, key):
    # language=rst
    """
    Sample from N(x|Az + b, diag(exp(log_diag_cov)))
    """

    # Transform onto the manifold
    x = A@z + b

    # Sample from N(x|Az + b, Sigma)
    if(key is not None):
        noise = random.normal(key, x.shape)*jnp.exp(0.5*log_diag_cov)*s
        x += noise

        # Compute the likelihood p(x|z)
        log_det = util.gaussian_diag_cov_logpdf(noise, jnp.zeros_like(noise), log_diag_cov)
    else:
        # Treat this as an injective flow and use the log determinant
        log_det = -0.5*jnp.linalg.slogdet(A.T@A)[1]

    outputs = {'x': x, 'log_det': log_det}
    return outputs

def forward_mc(z, target_x, A, b, log_diag_cov, s):
    # language=rst
    """
    Compute N(target_x|Az + b, diag(exp(log_diag_cov)))
    """
    # Transform onto the manifold
    x = A@z + b

    # Evaluate the log pdf of a given x.  This is used for testing.
    log_pxgz = util.gaussian_diag_cov_logpdf(target_x, x, log_diag_cov)

    outputs = {'x': x, 'log_pxgz': log_pxgz}
    return outputs

################################################################################################################

@base.auto_batch
def TallAffineDiagCov(out_dim, A_init=jaxinit.glorot_normal(), b_init=jaxinit.normal(), name='tall_affine_diag_cov'):
    """ Affine function to go up a dimension

        Args:
    """
    def apply_fun(params, state, inputs, reverse=False, forward_monte_carlo=False, key=None, **kwargs):
        A, b, log_diag_cov = params['A'], params['b'], params['log_diag_cov']
        s = state['s']

        if(reverse == False):
            outputs = sample_stochastic_inverse(inputs['x'], A, b, log_diag_cov, s, key)
        else:
            if(forward_monte_carlo):
                outputs = forward_mc(inputs['x'], inputs['target_x'], A, b, log_diag_cov, s)
            else:
                outputs = generate(inputs['x'], A, b, log_diag_cov, s, key)

        return outputs, state

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
        state = {'s': 1.0}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

@base.auto_batch
def CouplingTallAffineDiagCov(out_dim,
                              kind='every_other',
                              haiku_network1=None,
                              haiku_network2=None,
                              hidden_layer_sizes=[1024]*4,
                              A_init=jaxinit.glorot_normal(),
                              name='coupling_tall_affine_diag_cov'):
    ### p(x1, x2) = \int \int p(z1, z2)N(x1|A1@z1+b(x2),\Sigma(x2))N(x2|A2@z2+b(z1),\Sigma(z1))dz1 dz2
    """ General change of dimension.

        Args:
    """
    network1, network2 = None, None

    dims = {}
    split_indices = {'x': {}, 'z': {}}

    def apply_fun(params, state, inputs, reverse=False, forward_monte_carlo=False, key=None, **kwargs):
        A1, A2 = params['A1'], params['A2']
        network_params1, network_params2 = params['hk_params1'], params['hk_params2']
        s = state['s']
        k1, k2 = random.split(key, 2) if key is not None else (None,)*2

        if(reverse == False):
            x = inputs['x']
            assert x.ndim == 1

            # Split x
            if(kind == 'every_other'):
                x1, x2 = jnp.split(x[...,split_indices['x']['every_other']], jnp.array([dims['x1_dim']]), axis=-1)
            else:
                x1, x2 = jnp.split(x, jnp.array([dims['x1_dim']]), axis=-1)

            # Compute the bias and covariance conditioned on x2
            b1, log_diag_cov1 = network1.apply(network_params1, x2, **kwargs)
            outputs1 = sample_stochastic_inverse(x1, A1, b1, log_diag_cov1, s, k1)
            z1 = outputs1.pop('x')

            # Compute the bias and covariance conditioned on z1
            b2, log_diag_cov2 = network2.apply(network_params2, z1, **kwargs)
            outputs2 = sample_stochastic_inverse(x2, A2, b2, log_diag_cov2, s, k2)
            z2 = outputs2.pop('x')

            # Combine the results
            if(kind == 'every_other'):
                z = jnp.concatenate([z1, z2], axis=-1)[...,split_indices['z']['regular']]
            else:
                z = jnp.concatenate([z1, z2], axis=-1)

            outputs = jax.tree_util.tree_multimap(lambda x, y: x + y, outputs1, outputs2)
            outputs['x'] = z

        else:

            # If we're doing forward monte carlo, need to split the target output
            if(forward_monte_carlo):
                target_x = inputs['target_x']
                if(kind == 'every_other'):
                    target_x1, target_x2 = jnp.split(target_x[...,split_indices['x']['every_other']], jnp.array([dims['x1_dim']]), axis=-1)
                else:
                    target_x1, target_x2 = jnp.split(target_x, jnp.array([dims['x1_dim']]), axis=-1)

            # Split z
            z = inputs['x']
            assert z.ndim == 1
            if(kind == 'every_other'):
                z1, z2 = jnp.split(z[...,split_indices['z']['every_other']], jnp.array([dims['z1_dim']]), axis=-1)
            else:
                z1, z2 = jnp.split(z, jnp.array([dims['z1_dim']]), axis=-1)

            # Compute the bias and covariance conditioned on z1
            b2, log_diag_cov2 = network2.apply(network_params2, z1, **kwargs)

            if(forward_monte_carlo):
                outputs2 = forward_mc(z2, target_x2, A2, b2, log_diag_cov2, s)
            else:
                outputs2 = generate(z2, A2, b2, log_diag_cov2, s, k2)
            x2 = outputs2.pop('x')

            # Compute the bias and covariance conditioned on x2
            b1, log_diag_cov1 = network1.apply(network_params1, x2, **kwargs)
            if(forward_monte_carlo):
                outputs1 = forward_mc(z1, target_x1, A1, b1, log_diag_cov1, s)
            else:
                outputs1 = generate(z1, A1, b1, log_diag_cov1, s, k1)
            x1 = outputs1.pop('x')

            # Combine the results
            if(kind == 'every_other'):
                x = jnp.concatenate([x1, x2], axis=-1)[...,split_indices['x']['regular']]
            else:
                x = jnp.concatenate([x1, x2], axis=-1)

            outputs = jax.tree_util.tree_multimap(lambda x, y: x + y, outputs1, outputs2)
            outputs['x'] = x

        return outputs, state

    def create_params_and_state(key, input_shapes):
        x_shape = input_shapes['x']
        assert len(x_shape) == 1, 'Only working with vectors for the moment!!!'
        assert out_dim > 1, 'Can\'t end up with single dimension!  Need at least 2.'

        keys = random.split(key, 4)

        nonlocal dims
        dims['x_dim'] = x_shape[-1]
        dims['z_dim'] = out_dim
        assert dims['x_dim'] >= dims['z_dim']

        # We will split each vector into two almost equal size pieces
        dims['x1_dim'] = dims['x_dim']//2
        dims['x2_dim'] = dims['x_dim'] - dims['x1_dim']

        dims['z1_dim'] = dims['z_dim']//2
        dims['z2_dim'] = dims['z_dim'] - dims['z1_dim']

        # If we're splitting using every other index, generate the indexers needed
        nonlocal split_indices
        if(kind == 'every_other'):
            split_indices['x']['every_other'] = every_other_index(dims['x_dim'])
            split_indices['x']['regular'] = remap_indices(split_indices['x']['every_other'], dims['x_dim'])

            split_indices['z']['every_other'] = every_other_index(dims['z_dim'])
            split_indices['z']['regular'] = remap_indices(split_indices['z']['every_other'], dims['z_dim'])

        A1 = A_init(keys[0], (dims['x1_dim'], dims['z1_dim']))
        A2 = A_init(keys[1], (dims['x2_dim'], dims['z2_dim']))

        A1 = util.whiten(A1)
        A2 = util.whiten(A2)

        # Initialize each of the conditioner networks
        nonlocal network1
        input_shape1 = (dims['x2_dim'],)
        output_shape1 = (dims['x1_dim'],)
        if(haiku_network1 is None):
            network1 = hk.transform(lambda x, **kwargs: util.SimpleMLP(output_shape1, hidden_layer_sizes, is_additive=False)(x, **kwargs))
        else:
            network1 = hk.transform(lambda x, **kwargs: haiku_network(output_shape1)(x, **kwargs))
        hk_params1 = network1.init(keys[2], jnp.zeros(input_shape1))

        nonlocal network2
        input_shape2 = (dims['z1_dim'],)
        output_shape2 = (dims['x2_dim'],)
        if(haiku_network2 is None):
            network2 = hk.transform(lambda x, **kwargs: util.SimpleMLP(output_shape2, hidden_layer_sizes, is_additive=False)(x, **kwargs))
        else:
            network2 = hk.transform(lambda x, **kwargs: haiku_network(output_shape2)(x, **kwargs))
        hk_params2 = network2.init(keys[2], jnp.zeros(input_shape2))

        # Compile the parameters and state
        params = {'A1': A1, 'A2': A2, 'hk_params1': hk_params1, 'hk_params2': hk_params2}
        state = {'s': 1.0}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

__all__ = ['TallAffineDiagCov',
           'CouplingTallAffineDiagCov']