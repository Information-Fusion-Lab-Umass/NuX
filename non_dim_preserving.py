import numpy as onp
import jax
from jax import random, jit, vmap, jacobian, grad, value_and_grad
import jax.nn
import jax.numpy as np
from functools import partial, reduce
from jax.experimental import stax
from jax.nn.initializers import glorot_normal, normal, ones, zeros
from jax.ops import index, index_add, index_update
import staxplusplus as spp
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_multimap
from jax.tree_util import tree_flatten
from jax.scipy.special import logsumexp
from util import is_testing, TRAIN, TEST, householder_prod, householder_prod_transpose
import util
import normalizing_flows as nf

def AffineGaussianPriorFullCov(out_dim, A_init=glorot_normal(), Sigma_chol_init=normal(), name='unnamed'):
    """ Analytic solution to int N(z|0,I)N(x|Az,Sigma)dz.
        Allows normalizing flow to start in different dimension.

        Args:
    """
    triangular_indices = None
    def init_fun(key, input_shape, condition_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(key, 2)

        # Initialize the affine matrix
        A = A_init(k1, (input_shape[-1], out_dim))

        # Initialize the cholesky decomposition of the covariance matrix
        nonlocal triangular_indices
        dim = input_shape[-1]
        triangular_indices = util.upper_triangular_indices(dim)
        flat_dim = util.n_elts_upper_triangular(dim)
        Sigma_chol_flat = Sigma_chol_init(k2, (flat_dim,))

        params = (A, Sigma_chol_flat)
        state = ()
        return name, output_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        A, Sigma_chol_flat = params
        x_dim, z_dim = A.shape

        # Need to make the diagonal positive
        Sigma_chol = Sigma_chol_flat[triangular_indices]

        diag = np.diag(Sigma_chol)
        Sigma_chol = index_update(Sigma_chol, np.diag_indices(Sigma_chol.shape[0]), np.exp(diag))

        Sigma_inv_A = jax.scipy.linalg.cho_solve((Sigma_chol, True), A)
        ATSA = np.eye(z_dim) + A.T@Sigma_inv_A
        ATSA_inv = np.linalg.inv(ATSA)

        if(x.ndim == 1):
            z = np.einsum('ij,j->i', ATSA_inv@Sigma_inv_A.T, x)
            x_proj = np.einsum('ij,j->i', Sigma_inv_A, z)
            a = util.upper_cho_solve(Sigma_chol, x)
        elif(x.ndim == 2):
            z = np.einsum('ij,bj->bi', ATSA_inv@Sigma_inv_A.T, x)
            x_proj = np.einsum('ij,bj->bi', Sigma_inv_A, z)
            a = vmap(partial(util.upper_cho_solve, Sigma_chol))(x)
        else:
            assert 0, 'Got an invalid shape.  x.shape: %s'%(str(x.shape))

        log_hx = -0.5*np.sum(x*(a - x_proj), axis=-1)
        log_hx -= 0.5*np.linalg.slogdet(ATSA)[1]
        log_hx -= diag.sum()
        log_hx -= 0.5*x_dim*np.log(2*np.pi)
        return log_px + log_hx, z, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        # Passing back through the network, we just need to sample from N(x|Az,Sigma).
        # Assume we have already sampled z ~ N(0,I)
        A, Sigma_chol_flat = params

        # Compute Az
        if(z.ndim == 1):
            x = np.einsum('ij,j->i', A, z)
        elif(z.ndim == 2):
            x = np.einsum('ij,bj->bi', A, z)
        else:
            assert 0, 'Got an invalid shape.  z.shape: %s'%(str(z.shape))

        Sigma_chol = Sigma_chol_flat[triangular_indices]
        diag = np.diag(Sigma_chol)
        Sigma_chol = index_update(Sigma_chol, np.diag_indices(Sigma_chol.shape[0]), np.exp(diag))

        key = kwargs.pop('key', None)
        if(key is not None):
            noise = random.normal(key, x.shape)
            x += np.dot(noise, Sigma_chol.T)
        else:
            noise = np.zeros_like(x)

        # Compute N(x|Az+b, Sigma)
        # log_px = util.gaussian_diag_cov_logpdf(noise, np.zeros_like(noise), log_diag_cov)
        return log_pz, x, state

    return init_fun, forward, inverse

################################################################################################################

def AffineGaussianPriorDiagCov(out_dim, A_init=glorot_normal(), name='unnamed'):
    """ Analytic solution to int N(z|0,I)N(x|Az,Sigma)dz.
        Allows normalizing flow to start in different dimension.

        Args:
    """
    def init_fun(key, input_shape, condition_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        A = A_init(key, (input_shape[-1], out_dim))
        log_diag_cov = np.zeros(input_shape[-1])
        params = (A, log_diag_cov)
        state = ()
        return name, output_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        A, log_diag_cov = params
        diag_cov = np.exp(log_diag_cov)

        x_dim, z_dim = A.shape
        ATSA = np.eye(z_dim) + (A.T/diag_cov)@A
        ATSA_inv = np.linalg.inv(ATSA)

        if(x.ndim == 1):
            z = np.einsum('ij,j->i', ATSA_inv@A.T/diag_cov, x)
            x_proj = A@z/diag_cov
        elif(x.ndim == 2):
            z = np.einsum('ij,bj->bi', ATSA_inv@A.T/diag_cov, x)
            x_proj = np.einsum('ij,bj->bi', A, z)/diag_cov
        else:
            assert 0, 'Got an invalid shape.  x.shape: %s'%(str(x.shape))

        log_hx = -0.5*np.sum(x*(x/diag_cov - x_proj), axis=-1)
        log_hx -= 0.5*np.linalg.slogdet(ATSA)[1]
        log_hx -= 0.5*log_diag_cov.sum()
        log_hx -= 0.5*x_dim*np.log(2*np.pi)
        return log_px + log_hx, z, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        # Passing back through the network, we just need to sample from N(x|Az,Sigma).
        # Assume we have already sampled z ~ N(0,I)
        A, log_diag_cov = params

        log_diag_cov = np.zeros_like(log_diag_cov)

        # Compute Az
        if(z.ndim == 1):
            x = np.einsum('ij,j->i', A, z)
        elif(z.ndim == 2):
            x = np.einsum('ij,bj->bi', A, z)
        else:
            assert 0, 'Got an invalid shape.  z.shape: %s'%(str(z.shape))

        key = kwargs.pop('key', None)
        if(key is not None):
            noise = random.normal(key, x.shape)*np.exp(0.5*log_diag_cov)
            x += noise
        else:
            noise = x*0.0

        # Compute N(x|Az+b, Sigma)
        log_px = util.gaussian_diag_cov_logpdf(noise, np.zeros_like(noise), log_diag_cov)

        return log_pz + log_px, x, state

    return init_fun, forward, inverse

################################################################################################################

@jit
def tall_affine_posterior_diag_cov(x, b, A, log_diag_cov):
    """ Sample from N(z|mu(x), Sigma(x)) and compute h(x).
        Args:
    """
    diag_cov = np.exp(log_diag_cov)
    xminusb = x - b

    # Find the pseudo inverse and the projection
    ATSA = A.T/diag_cov@A
    ATSA_inv = np.linalg.inv(ATSA)
    z = np.dot(xminusb, (ATSA_inv@A.T/diag_cov).T)
    x_proj = np.dot(z, A.T)/diag_cov

    # Get the terms that don't depend on z
    dim_x, dim_z = A.shape
    log_hx = -0.5*np.sum(xminusb*(xminusb/diag_cov - x_proj), axis=-1)
    log_hx -= 0.5*np.linalg.slogdet(ATSA)[1]
    log_hx -= 0.5*log_diag_cov.sum()
    log_hx -= 0.5*(dim_x - dim_z)*np.log(2*np.pi)

    # For sampling
    sigma_ATA_chol = np.linalg.cholesky(ATSA_inv)

    return z, log_hx, sigma_ATA_chol


# @partial(jit, static_argnums=(0, 6))
def importance_sample_prior(prior_forward, prior_params, prior_state, z, condition, sigma_ATA_chol, n_training_importance_samples, **kwargs):
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
        z_samples = z[None,...] + np.dot(noise, sigma_ATA_chol.T)
    else:
        # We're only using the mean, but put it on an axis so that we can use vmap
        z_samples = z[None]

    # Compute the rest of the flow with the samples of z
    vmapped_forward = vmap(partial(prior_forward, **kwargs), in_axes=(None, None, None, 0, None))
    log_pxs, zs, updated_prior_states = vmapped_forward(prior_params, prior_state, np.zeros(z_samples.shape[0]), z_samples, condition)

    # Compile the results
    log_px = logsumexp(log_pxs, axis=0) - np.log(log_pxs.shape[0])
    z = np.mean(zs, axis=0) # Just average the state
    updated_prior_states = updated_prior_states # For some reason, this isn't changed with vmap?
    return log_px, z, updated_prior_states

################################################################################################################

def TallAffineDiagCov(flow, out_dim, n_training_importance_samples=1, A_init=glorot_normal(), b_init=normal(), name='unnamed'):
    """ Affine function to go up a dimension

        Args:
    """
    _init_fun, _forward, _inverse = flow

    def init_fun(key, input_shape, condition_shape):
        x_shape = input_shape
        output_shape = x_shape[:-1] + (out_dim,)
        keys = random.split(key, 3)

        x_dim = x_shape[-1]
        z_dim = out_dim
        A = A_init(keys[0], (x_shape[-1], out_dim))
        b = b_init(keys[1], (x_shape[-1],))
        flow_name, flow_output_shape, flow_params, flow_state = _init_fun(keys[2], output_shape, condition_shape)
        log_diag_cov = np.ones(input_shape[-1])*0.0
        params = ((A, b, log_diag_cov), flow_params)
        state = ((), flow_state)
        return (name, flow_name), flow_output_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        ((A, b, log_diag_cov), flow_params) = params
        _, flow_state = state

        # Get the terms to compute and sample from the posterior
        z, log_hx, sigma_ATA_chol = tall_affine_posterior_diag_cov(x, b, A, log_diag_cov)

        # Importance sample from N(z|\mu(x),\Sigma(x)) and compile the results
        log_pz, z, updated_flow_states = importance_sample_prior(_forward, flow_params, flow_state, z, condition, sigma_ATA_chol, n_training_importance_samples, **kwargs)

        # Compute the final estimate of the integral
        log_px = log_px + log_pz + log_hx

        return log_px, z, ((), updated_flow_states)

    def inverse(params, state, log_pz, z, condition, **kwargs):
        ((A, b, log_diag_cov), flow_params) = params
        _, flow_state = state

        log_pz, z, updated_state = _inverse(flow_params, flow_state, log_pz, z, condition, **kwargs)

        # Compute Az + b
        # Don't need to sample because we already sampled from p(z)!!!!
        x = np.dot(z, A.T) + b

        # Compute N(x|Az + b, \Sigma).  This is just the log partition function.
        log_px = - 0.5*np.sum(log_diag_cov) - 0.5*x.shape[-1]*np.log(2*np.pi)
        return log_pz + log_px, x, ((), updated_state)

    return init_fun, forward, inverse

################################################################################################################

def every_other(x):
    assert x.ndim == 1
    dim_x = x.shape[0]
    y = np.pad(x, (0, 1)) if dim_x%2 == 1 else x

    dim_y = y.shape[0]
    y = y.reshape((-1, 2)).T.reshape(dim_y)

    return y[:-1] if dim_x%2 == 1 else y

def CoupledDimChange(transform_fun,
                     prior_flow,
                     out_dim,
                     kind='every_other',
                     A_init=glorot_normal(),
                     name='unnamed'):
    ### p(x1, x2) = \int \int p(z1, z2)N(x1|A1@z1+b(x2),\Sigma(x2))N(x2|A2@z2+b(z1),\Sigma(z1))dz1 dz2
    """ General change of dimension.

        Args:
    """
    apply_fun = None
    prior_init_fun, prior_forward, prior_inverse = prior_flow
    x1_dim, x2_dim, z1_dim, z2_dim = None, None, None, None
    x_every_other_idx, x_regular_idx, z_every_other_idx, z_regular_idx = None, None, None, None

    def init_fun(key, input_shape, condition_shape):
        x_shape = input_shape
        assert len(x_shape) == 1, 'Only working with vectors for the moment!!!'
        assert out_dim > 1, 'Can\'t end up with single dimension!  Need at least 2.'

        output_shape = (out_dim,)
        keys = random.split(key, 5)

        x_dim = x_shape[-1]
        z_dim = out_dim
        assert x_dim >= z_dim

        # Figure out how to split x and how that impacts the shapes of z_1 and z_2
        nonlocal x1_dim, x2_dim, z1_dim, z2_dim
        x1_dim = x_dim//2
        x2_dim = x_dim - x1_dim

        z1_dim = out_dim//2
        z2_dim = out_dim - z1_dim

        # If we're splitting using every other index, generate the indexers needed
        if(kind == 'every_other'):
            nonlocal x_every_other_idx, x_regular_idx, z_every_other_idx, z_regular_idx
            x_every_other_idx = every_other(np.arange(x_dim))
            x_regular_idx = np.array([list(x_every_other_idx).index(i) for i in range(x_dim)])

            z_every_other_idx = every_other(np.arange(z_dim))
            z_regular_idx = np.array([list(z_every_other_idx).index(i) for i in range(z_dim)])

        # We're not going to learn A for the moment
        A1 = A_init(keys[0], (x1_dim, z1_dim))
        A2 = A_init(keys[1], (x2_dim, z2_dim))

        # Initialize the flow
        prior_name, prior_output_shape, prior_params, prior_state = prior_init_fun(keys[2], output_shape, condition_shape)

        # Initialize each of the flows.  apply_fun can be shared
        nonlocal apply_fun
        init_fun1, apply_fun = transform_fun(out_shape=(x1_dim,))
        init_fun2, _ = transform_fun(out_shape=(x2_dim,))

        # Initialize the transform function.
        # Should output bias and log diagonal covariance
        t_name1, (log_diag_cov_shape1, b_shape1), t_params1, t_state1 = init_fun1(keys[3], (x2_dim,))
        t_name2, (log_diag_cov_shape2, b_shape2), t_params2, t_state2 = init_fun2(keys[4], (z1_dim,))

        names = (name, prior_name, t_name1, t_name2)
        params = ((A1, A2), prior_params, t_params1, t_params2)
        state = ((), prior_state, t_state1, t_state2)
        return names, prior_output_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        (A1, A2), prior_params, t_params1, t_params2 = params
        _, prior_state, t_state1, t_state2 = state

        # Get multiple keys if we're sampling
        key = kwargs.pop('key', None)
        if(key is not None):
            # Re-fill key for the rest of the flow
            k1, k2, k3, k4, k5 = random.split(key, 5)
        else:
            k1, k2, k3, k4, k5 = (None,)*5

        # Determine if we are batching or not
        is_batched = x.ndim == 2
        posterior_fun = vmap(tall_affine_posterior_diag_cov, in_axes=(0, 0, None, 0)) if is_batched else tall_affine_posterior_diag_cov

        # Split x
        if(kind == 'every_other'):
            x1, x2 = np.split(x[...,x_every_other_idx], np.array([x1_dim]), axis=-1)
        else:
            x1, x2 = np.split(x, np.array([x1_dim]), axis=-1)

        # Compute the bias and covariance conditioned on x2
        (log_diag_cov1, b1), updated_t_state1 = apply_fun(t_params1, t_state1, x2, key=k1, **kwargs)

        # Get the terms to compute and sample from the posterior
        z1, log_hx1, sigma_ATA_chol1 = posterior_fun(x1, b1, A1, log_diag_cov1)

        # Sample z1
        if(key is not None):
            noise = random.normal(k2, z1.shape)
            if(is_batched):
                z1 += np.einsum('bij,bj->bi', sigma_ATA_chol1, noise)
            else:
                z1 += np.einsum('ij,j->i', sigma_ATA_chol1, noise)

        # Compute the bias and covariance conditioned on z1
        (log_diag_cov2, b2), updated_t_state2 = apply_fun(t_params2, t_state2, z1, key=k3, **kwargs)

        # Get the terms to compute and sample from the posterior
        z2, log_hx2, sigma_ATA_chol2 = posterior_fun(x2, b2, A2, log_diag_cov2)

        # Sample z2
        if(key is not None):
            noise = random.normal(k4, z2.shape)
            if(is_batched):
                z2 += np.einsum('bij,bj->bi', sigma_ATA_chol2, noise)
            else:
                z2 += np.einsum('ij,j->i', sigma_ATA_chol2, noise)

        # Combine z
        if(kind == 'every_other'):
            z = np.concatenate([z1, z2], axis=-1)[...,z_regular_idx]
        else:
            z = np.concatenate([z1, z2], axis=-1)

        # Compute the prior
        log_pz, z, updated_prior_state = prior_forward(prior_params, prior_state, log_px, z, condition, key=k5, **kwargs)

        # Return the full estimate of the integral and the updated
        updated_states = ((), updated_prior_state, updated_t_state1, updated_t_state2)
        return log_pz + log_hx1 + log_hx2, z, updated_states

    def inverse(params, state, log_pz, z, condition, **kwargs):
        (A1, A2), prior_params, t_params1, t_params2 = params
        _, prior_state, t_state1, t_state2 = state

        # Get multiple keys if we're sampling
        key = kwargs.pop('key', None)
        if(key is not None):
            # Re-fill key for the rest of the flow
            k1, k2, k3 = random.split(key, 3)
        else:
            k1, k2, k3 = (None,)*3

        # Run the input through the prior
        log_pz, z, updated_prior_state = prior_inverse(prior_params, prior_state, log_pz, z, condition, key=k1, **kwargs)

        # Split z
        if(kind == 'every_other'):
            z1, z2 = np.split(z[...,z_every_other_idx], np.array([z1_dim]), axis=-1)
        else:
            z1, z2 = np.split(z, np.array([z1_dim]), axis=-1)

        # Compute the bias and covariance conditioned on z1
        (log_diag_cov2, b2), updated_t_state2 = apply_fun(t_params2, t_state2, z1, key=k2, **kwargs)

        # Compute x2
        x2 = np.dot(z2, A2.T) + b2

        # Compute the bias and covariance conditioned on x2
        (log_diag_cov1, b1), updated_t_state1 = apply_fun(t_params1, t_state1, x2, key=k3, **kwargs)

        # Compute x1
        x1 = np.dot(z1, A1.T) + b1

        # Combine x
        if(kind == 'every_other'):
            x = np.concatenate([x1, x2], axis=-1)[...,x_regular_idx]
        else:
            x = np.concatenate([x1, x2], axis=-1)

        # Compute N(x|Az + b, \Sigma).  This is just the log partition function.
        log_px1 = - 0.5*np.sum(log_diag_cov1) - 0.5*x.shape[-1]*np.log(2*np.pi)
        log_px2 = - 0.5*np.sum(log_diag_cov2) - 0.5*x.shape[-1]*np.log(2*np.pi)

        updated_states = ((), updated_prior_state, updated_t_state1, updated_t_state2)
        return log_pz + log_px1 + log_px2, x, updated_states

    return init_fun, forward, inverse

################################################################################################################

def ConditionedTallAffineDiagCov(transform_fun,
                                 flow,
                                 out_dim,
                                 n_training_importance_samples=1,
                                 A_init=glorot_normal(),
                                 name='unnamed'):
    """ Affine function to go up a dimension

        Args:
    """
    apply_fun = None
    _init_fun, _forward, _inverse = flow

    def init_fun(key, input_shape, condition_shape):
        x_shape = input_shape
        output_shape = x_shape[:-1] + (out_dim,)
        keys = random.split(key, 3)

        x_dim = x_shape[-1]
        z_dim = out_dim

        # We're not going to learn A for the moment
        A = A_init(keys[0], (x_shape[-1], out_dim))

        # Initialize the flow
        flow_name, flow_output_shape, flow_params, flow_state = _init_fun(keys[1], output_shape, condition_shape)

        # Initialize the transform function.
        # Should output bias and log diagonal covariance
        nonlocal apply_fun
        initfun, apply_fun = transform_fun(out_shape=x_shape)
        transform_name, (log_diag_cov_shape, b_shape), transform_params, transform_state = initfun(keys[2], condition_shape)

        names = (name, flow_name, transform_name)
        params = (A, flow_params, transform_params)
        state = ((), flow_state, transform_state)
        return names, flow_output_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        A, flow_params, transform_params = params
        _, flow_state, transform_state = state

        # Get the bias and log diagonal covariance
        (log_diag_cov, b), updated_transform_state = apply_fun(transform_params, transform_state, condition, **kwargs)

        # Be careful when things are batched!!! Handle batching with vmap at a higher level!
        if(b.ndim == 2):
            assert b.shape[0] == 1

        # Get the terms to compute and sample from the posterior
        z, log_hx, sigma_ATA_chol = tall_affine_posterior_diag_cov(x, b, A, log_diag_cov)

        # Importance sample from N(z|\mu(x),\Sigma(x)) and compile the results
        log_pz, z, updated_flow_states = importance_sample_prior(_forward, flow_params, flow_state, z, condition, sigma_ATA_chol, n_training_importance_samples, **kwargs)

        # Compute the final estimate of the integral
        log_px = log_px + log_pz + log_hx

        return log_px, z, ((), updated_flow_states)

    def inverse(params, state, log_pz, z, condition, **kwargs):
        A, flow_params, transform_params = params
        _, flow_state, transform_state = state

        # Pass the input through the prior flow p(z)
        log_pz, z, updated_flow_state = _inverse(flow_params, flow_state, log_pz, z, condition, **kwargs)

        # Get the bias and log diagonal covariance
        (log_diag_cov, b), updated_transform_state = apply_fun(transform_params, transform_state, condition, **kwargs)

        # Compute Az + b
        # Don't need to sample because we already sampled from p(z)!!!!
        x = np.dot(z, A.T) + b

        # Compute N(x|Az + b, \Sigma).  This is just the log partition function.
        log_px = - 0.5*np.sum(log_diag_cov) - 0.5*x.shape[-1]*np.log(2*np.pi)
        return log_pz + log_px, x, ((), updated_state)

    return init_fun, forward, inverse

# ################################################################################################################

def UpSample(repeats, name='unnamed'):
    # language=rst
    """
    Up sample by just repeating consecutive values over specified axes

    :param repeats - The number of times to repeat.  Pass in (2, 1, 2), for example, to repeat twice over
                     the 0th axis, no repeats over the 1st axis, and twice over the 2nd axis
    """
    full_repeats = None
    def init_fun(key, input_shape, condition_shape):
        nonlocal full_repeats
        full_repeats = [repeats[i] if i < len(repeats) else 1 for i in range(len(input_shape))]
        output_shape = []
        for s, r in zip(input_shape, full_repeats):
            assert s%r == 0
            output_shape.append(s//r)
        output_shape = tuple(output_shape)
        log_diag_cov = np.ones(input_shape)*5
        b = np.zeros(input_shape)

        params, state = (log_diag_cov, b), ()
        return name, output_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
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
            z = z_mean + noise/np.sqrt(rm_diag)
        else:
            z = z_mean

        return log_px + log_hx, z, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        if(z.ndim == 4):
            return vmap(partial(inverse, params, state, **kwargs), in_axes=(0, 0, None))(log_pz, z, condition)

        log_diag_cov, b = params
        log_diag_cov = -jax.nn.softplus(log_diag_cov) - 3.0 # Limit the amount of noise

        # x ~ N(x|Az + b, Sigma)

        x_mean = upsample(full_repeats, z) + b

        key = kwargs.pop('key', None)
        if(key is not None):
            noise = np.exp(0.5*log_diag_cov)*random.normal(key, x_mean.shape)
            x = x_mean + noise
        else:
            noise = np.zeros_like(x_mean)
            x = x_mean

        log_px = -0.5*np.sum(noise*np.exp(-0.5*log_diag_cov)*noise) - 0.5*np.sum(log_diag_cov) - 0.5*x.shape[-1]*np.log(2*np.pi)

        return log_pz + log_px, x, state

    return init_fun, forward, inverse

################################################################################################################

def indexer_and_shape_from_mask(mask):
    # language=rst
    """
    Given a 2d mask array, create an array that can index into a vector with the same number of elements
    as nonzero elements in mask and result in an array of the same size as mask, but with the elements
    specified from the vector. Also return the shape of the resulting array when mask is applied.

    :param mask: 2d boolean mask array
    """
    index = onp.zeros_like(mask, dtype=int)
    non_zero_indices = np.nonzero(mask)
    index[non_zero_indices] = np.arange(len(non_zero_indices[0])) + 1

    nonzero_x, nonzero_y = non_zero_indices
    n_rows = onp.unique(nonzero_x).size
    assert nonzero_x.size%n_rows == 0
    n_cols = nonzero_x.size // n_rows
    shape = (n_rows, n_cols)
    return index, shape

def check_mask(mask):
    # language=rst
    """
    Check if the 2d boolean mask is valid

    :param mask: 2d boolean mask array
    """
    if(np.any(mask) == False):
        assert 0, 'Empty mask!  Reduce num'
    if(np.sum(mask)%2 == 1):
        assert 0, 'Need masks with an even number!  Choose a different num'

def checkerboard_masks(num, shape):
    # language=rst
    """
    Finds masks to factor an array with a given shape so that each pixel will be
    present in the resulting masks.  Also return indices that will help reverse
    the factorization.

    :param masks: A list of 2d boolean mask array whose union is equal to np.ones(shape, dtype=bool)
    :param indices: A list of index matrices that undo the application of image[mask]
    """
    masks = []
    indices = []
    shapes = []

    for i in range(2*num):
        start = 2**i
        step = 2**(i + 1)

        # Staggered indices
        mask = onp.zeros(shape, dtype=bool)
        mask[start::step,::step] = True
        mask[::step,start::step] = True
        check_mask(mask)
        masks.append(mask)
        index, new_shape = indexer_and_shape_from_mask(mask)
        indices.append(index)
        shapes.append(new_shape)

        if(len(masks) + 1 == num):
            break

        # Not staggered indices
        mask = onp.zeros(shape, dtype=bool)
        mask[start::step,start::step] = True
        mask[start::step,start::step] = True
        check_mask(mask)
        masks.append(mask)
        index, new_shape = indexer_and_shape_from_mask(mask)
        indices.append(index)
        shapes.append(new_shape)

        if(len(masks) + 1 == num):
            break

    used = sum(masks).astype(bool)
    mask = ~used
    masks.append(mask)
    index, new_shape = indexer_and_shape_from_mask(mask)
    indices.append(index)
    shapes.append(new_shape)

    return masks, indices, shapes

def upsample(repeats, z):
    x = z
    is_batched = int(x.ndim == 2 or x.ndim == 4)
    for i, r in enumerate(repeats):
        x = np.repeat(x, r, axis=i + is_batched)
    return x

def downsample(repeats, x):
    return x[[slice(0, None, r) for r in repeats]]

def upsample_pseudo_inverse(x, repeats):
    # language=rst
    """
    Compute the pseudo inverse of an upsample
    """
    hr, wr, cr = repeats
    assert cr == 1
    Hx, Wx, C = x.shape
    assert Hx%hr == 0 and Wx%wr == 0
    H, W = Hx//hr, Wx//wr

    return x.reshape((H, hr, W, wr, C)).transpose((0, 2, 4, 1, 3)).reshape((H, W, C, hr*wr)).mean(axis=-1)

def upsample_idx(repeats, idx):
    repeats = (repeats[0], repeats[1])
    for i, r in enumerate(repeats):
        idx = np.repeat(idx, r, axis=i)

    idx = onp.array(idx)

    k = 1
    for i in range(idx.shape[0]):
        for j in range(idx.shape[1]):
            if(idx[i,j] >= 1):
                idx[i,j] = k
                k += 1
    return idx

def upsample_posterior(x, b, log_diag_cov, repeats):
    """ Posterior of N(x|Az + b, Sigma) where A is an upsample matrix"""
    assert x.shape == b.shape
    assert x.shape == log_diag_cov.shape
    assert x.ndim == 3
    xmb = x - b
    one_over_diag_cov = np.exp(-log_diag_cov)

    # Compute the diagonal of the riemannian metric.  This is the diagonal of A^T Sigma^{-1} A
    hr, wr, cr = repeats; assert cr == 1 # Haven't tested cr != 1
    Hx, Wx, C = x.shape
    H, W = Hx//hr, Wx//wr
    rm_diag = one_over_diag_cov.reshape((H, hr, W, wr, C)).transpose((0, 2, 4, 1, 3)).reshape((H, W, C, hr*wr)).sum(axis=-1)

    # Compute the mean of z
    z_mean = upsample_pseudo_inverse(xmb*one_over_diag_cov, (2, 2, 1))/rm_diag*(hr*wr)
    x_proj = upsample(repeats, z_mean)*one_over_diag_cov
    dim_x = np.prod(x.shape)
    dim_z = np.prod(z_mean.shape)

    # Compute the manifold error term
    log_hx = -0.5*np.sum(xmb*(xmb*one_over_diag_cov - x_proj))
    log_hx -= 0.5*np.sum(np.log(rm_diag))
    log_hx -= 0.5*log_diag_cov.sum()
    log_hx -= 0.5*(dim_x - dim_z)*np.log(2*np.pi)

    # return z_mean, log_hx, rm_diag, x_proj
    return z_mean, log_hx, rm_diag

def split_x(x, idx):
    H, W, C = x.shape
    # W will be cut in half
    return x[idx > 0].reshape((H, W//2, C))

def recombine(z, index):
    # language=rst
    """
    Use a structured set of indices to create a matrix from a vector

    :param z: Flat input that contains the elements of the output matrix
    :param indices: An array of indices that correspond to values in z
    """
    return np.pad(z.ravel(), (1, 0))[index]

# Applies the upsampled z indices
recombine_vmapped = vmap(recombine, in_axes=(2, None), out_axes=2)

def FilledCoupledUpsample(n_channels=512, name='unnamed'):
    def Coupling2D(out_shape, n_channels=n_channels):
        return spp.DoubledLowDimInputConvBlock(n_channels=n_channels)
    return CoupledUpSample(Coupling2D, (2, 2))

def CoupledUpSample(transform_fun, repeats, name='unnamed'):
    # language=rst
    """
    Up sample by just repeating consecutive values over specified axes

    :param repeats - The number of times to repeat.  Pass in (2, 1, 2), for example, to repeat twice over
                     the 0th axis, no repeats over the 1st axis, and twice over the 2nd axis
    """
    full_repeats = None
    apply_fun = None
    z_masks, z_shapes = None, None
    z_indices, upsampled_z_indices = None, None

    def init_fun(key, input_shape, condition_shape):
        keys = random.split(key, 3)
        x_shape = input_shape
        nonlocal full_repeats
        full_repeats = [repeats[i] if i < len(repeats) else 1 for i in range(len(x_shape))]
        z_shape = []
        for s, r in zip(x_shape, full_repeats):
            assert s%r == 0
            z_shape.append(s//r)
        z_shape = tuple(z_shape)
        Hz, Wz, Cz = z_shape
        Hx, Wx, Cx = x_shape

        # Generate the masks needed to go from z to x
        nonlocal z_masks, z_indices, z_shapes
        z_masks, z_indices, z_shapes = checkerboard_masks(2, z_shape[:-1])
        z_shapes = [(h, w, z_shape[-1]) for (h, w) in z_shapes]

        # Used to go from x1 to z1
        slices = [slice(0, None, r) for r in full_repeats]

        # Lets us go from an upsampled, split z, to a sparse x
        nonlocal upsampled_z_indices
        upsampled_z_indices = [upsample_idx(full_repeats, idx) for idx in z_indices]

        # Figure out how to split x and z
        z1_shape = (Hz, Wz//2, Cz)
        x2_shape = (Hx, Wx//2, Cx)

        # Initialize each of the flows.  apply_fun can be shared
        nonlocal apply_fun
        init_fun1, apply_fun = transform_fun(out_shape=x2_shape)
        init_fun2, _ = transform_fun(out_shape=x2_shape)

        # Initialize the transform function.
        # Should output bias and log diagonal covariance
        t_name1, (log_diag_cov_shape1, b_shape1), t_params1, t_state1 = init_fun1(keys[1], x2_shape)
        t_name2, (log_diag_cov_shape2, b_shape2), t_params2, t_state2 = init_fun2(keys[2], x2_shape) # For simplicity, will be passing in an upsampled z1 to this

        names = (name, t_name1, t_name2)
        params = ((), t_params1, t_params2)
        state = ((), t_state1, t_state2)
        return names, z_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        if(x.ndim == 4):
            return vmap(partial(forward, params, state, **kwargs), in_axes=(0, 0, None))(log_px, x, condition)

        _, t_params1, t_params2 = params
        _, t_state1, t_state2 = state

        # Get multiple keys if we're sampling
        key = kwargs.pop('key', None)
        if(key is not None):
            # Re-fill key for the rest of the flow
            k1, k2, k3, k4, k5 = random.split(key, 5)
        else:
            k1, k2, k3, k4, k5 = (None,)*5

        # Determine if we are batching or not
        is_batched = x.ndim == 4
        posterior_fun = vmap(upsample_posterior, in_axes=(0, 0, 0, None)) if is_batched else upsample_posterior
        combine = vmap(recombine_vmapped, in_axes=(0, None)) if is_batched else recombine_vmapped

        # Split x
        x1, x2 = split_x(x, upsampled_z_indices[0]), split_x(x, upsampled_z_indices[1])

        """ Posterior of N(x_1|Az_1 + b(x_2), Sigma(x_2)) """

        # Compute the bias and covariance conditioned on x2.  \sigma(x2), b(x2)
        (log_diag_cov1, b1), updated_t_state1 = apply_fun(t_params1, t_state1, x2, key=k1, **kwargs)
        log_diag_cov1 = -jax.nn.softplus(log_diag_cov1)

        # Compute the posterior and the manifold penalty
        z1_mean, log_hx1, rm1_diag = posterior_fun(x1, b1, log_diag_cov1, full_repeats)

        # Sample z1
        if(key is not None):
            noise = random.normal(k2, z1_mean.shape)
            z1 = z1_mean + noise/np.sqrt(rm1_diag)
        else:
            z1 = z1_mean

        """ Posterior of N(x_2|Az_2 + b(z_1), Sigma(z_1)) """

        # Compute the bias and covariance conditioned on z1.  \sigma(z1), b(z1)
        (log_diag_cov2, b2), updated_t_state2 = apply_fun(t_params2, t_state2, upsample(full_repeats, z1), key=k3, **kwargs)
        log_diag_cov2 = -jax.nn.softplus(log_diag_cov2)

        # Compute the posterior and the manifold penalty
        z2_mean, log_hx2, rm2_diag = posterior_fun(x2, b2, log_diag_cov2, full_repeats)

        # Sample z2
        if(key is not None):
            noise = random.normal(k4, z2_mean.shape)
            z2 = z2_mean + noise/np.sqrt(rm2_diag)
        else:
            z2 = z2_mean

        """ Combine z """
        z = combine(z1, z_indices[0]) + combine(z2, z_indices[1])

        # Return the full estimate of the integral and the updated
        updated_states = ((), updated_t_state1, updated_t_state2)
        return log_px + log_hx1 + log_hx2, z, updated_states

    def inverse(params, state, log_pz, z, condition, **kwargs):
        if(z.ndim == 4):
            return vmap(partial(inverse, params, state, **kwargs), in_axes=(0, 0, None))(log_pz, z, condition)

        _, t_params1, t_params2 = params
        _, t_state1, t_state2 = state

        # Get multiple keys if we're sampling
        key = kwargs.pop('key', None)
        if(key is not None):
            # Re-fill key for the rest of the flow
            k1, k2, k3, k4, k5 = random.split(key, 5)
        else:
            k1, k2, k3, k4, k5 = (None,)*5

        # Split z
        z1, z2 = z[z_masks[0]].reshape(z_shapes[0]), z[z_masks[1]].reshape(z_shapes[1])

        """ N(x_2|Az_2 + b(z_1), Sigma(z_1)) """

        # Compute the bias and covariance conditioned on z1.  \sigma(z1), b(z1)
        (log_diag_cov2, b2), updated_t_state2 = apply_fun(t_params2, t_state2, upsample(full_repeats, z1), key=k2, **kwargs)
        log_diag_cov2 = -jax.nn.softplus(log_diag_cov2)

        # Compute the mean of x2
        x2_mean = upsample(full_repeats, z2) + b2

        # Sample x2
        if(key is not None):
            noise2 = np.exp(0.5*log_diag_cov2)*random.normal(k3, x2_mean.shape)
            x2 = x2_mean + noise2
        else:
            noise2 = np.zeros_like(x2_mean)
            x2 = x2_mean

        """ N(x_1|Az_1 + b(x_2), Sigma(x_2)) """

        # Compute the bias and covariance conditioned on x2.  \sigma(x2), b(x2)
        (log_diag_cov1, b1), updated_t_state1 = apply_fun(t_params1, t_state1, x2, key=k4, **kwargs)
        log_diag_cov1 = -jax.nn.softplus(log_diag_cov1)

        # Compute the mean of x2
        x1_mean = upsample(full_repeats, z1) + b1

        # Sample x2
        if(key is not None):
            noise1 = np.exp(0.5*log_diag_cov1)*random.normal(k5, x1_mean.shape)
            x1 = x1_mean + noise1
        else:
            noise1 = np.zeros_like(x1_mean)
            x1 = x1_mean

        # Combine x
        is_batched = z.ndim == 4
        combine = recombine_vmapped if is_batched else recombine
        x = recombine_vmapped(x1, upsampled_z_indices[0]) + recombine_vmapped(x2, upsampled_z_indices[1])

        # Compute N(x1|Az1 + b(x2), Sigma(x2))N(x2|Az2 + b(z1), Sigma(z1))
        log_px1 = -0.5*np.sum(noise1*np.exp(-0.5*log_diag_cov1)*noise1) - 0.5*np.sum(log_diag_cov1) - 0.5*x1.shape[-1]*np.log(2*np.pi)
        log_px2 = -0.5*np.sum(noise2*np.exp(-0.5*log_diag_cov2)*noise2) - 0.5*np.sum(log_diag_cov2) - 0.5*x2.shape[-1]*np.log(2*np.pi)

        updated_states = ((), updated_t_state1, updated_t_state2)
        return log_pz + log_px1 + log_px2, x, updated_states

    return init_fun, forward, inverse

def EasyUpsample(n_channels=256, ratio=2, name='unnamed'):
    def Coupling2D(out_shape, n_channels=n_channels):
        return spp.sequential(spp.LowDimInputConvBlock(n_channels=n_channels), spp.SqueezeExcitation(ratio=ratio))

    return CoupledUpSample(Coupling2D, (2, 2), name=name)

# ################################################################################################################

def marginal_test(flow, x, key, n_keys=100, n_z=100, **kwargs):
    # x should have shape (1, x_dim)
    # Assumes that we just have one component of the flow.
    # Will verify that E_{p(z)}[N(x|Az+b, \Sigma)] = h(x)E_{N(z|\mu(x),\Sigma(x))}[p(z)]
    init_fun, forward, inverse = flow
    _, out_shape, params, state = init_fun(key, x.shape[1:], ())

    _, z, _ = forward(params, state, np.zeros(x.shape[0]), x, (), key=key, **kwargs)
    z_shape = z.shape[1:]

    def forward_w_key(key):
        log_px, z, _ = forward(params, state, np.zeros(x.shape[0]), x, (), n_importance_samples=n_z, key=key, **kwargs)
        return log_px

    keys = np.array(random.split(key, n_keys))
    expectation1 = vmap(forward_w_key)(keys)

    def inverse_w_z_and_key(z_samples, key):
        # log_pfz, _, _ = inverse(params, state, np.zeros(z_samples.shape[0]), z_samples, (), key=key, **kwargs)
        _, fz, _ = inverse(params, state, np.zeros(z_samples.shape[0]), z_samples, (), key=key, **kwargs)
        ((A, b, log_diag_cov), flow_params) = params
        log_px = vmap(util.gaussian_diag_cov_logpdf, in_axes=(None, 0, None))(x, fz, log_diag_cov)
        return logsumexp(log_px, axis=0) - np.log(log_px.shape[0])

    # Sample the zs
    z_samples = random.normal(random.split(key, 2)[1], (n_z,) + z_shape)
    expectation2 = vmap(inverse_w_z_and_key, in_axes=(None, 0))(z_samples, keys)
    return expectation1, expectation2