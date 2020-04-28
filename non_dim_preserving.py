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

        # Compute Az
        if(z.ndim == 1):
            x = np.einsum('ij,j->i', A, z)
        elif(z.ndim == 2):
            x = np.einsum('ij,bj->bi', A, z)
        else:
            assert 0, 'Got an invalid shape.  z.shape: %s'%(str(z.shape))

        key = kwargs.pop('key', None)
        if(key is not None):
            noise = random.normal(key, x.shape)
            x += noise*np.exp(-0.5*log_diag_cov)
        else:
            noise = x*0.0

        # Compute N(x|Az+b, Sigma)
        log_px = util.gaussian_diag_cov_logpdf(noise, np.zeros_like(noise), log_diag_cov)
        return log_pz + log_px, x, updated_state

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

################################################################################################################

# def TallSVDAffineScalarCov(flow,
#                            out_dim,
#                            n_householders,
#                            U_init=glorot_normal(),
#                            log_s_init=normal(),
#                            VT_init=glorot_normal(),
#                            b_init=normal(),
#                            name='unnamed'):
#     """ Affine function to go up a dimension using an SVD parametrization and scalar variance.

#         Args:
#     """
#     _init_fun, _forward, _inverse = flow

#     def init_fun(key, input_shape, condition_shape):
#         x_shape = input_shape
#         output_shape = x_shape[:-1] + (out_dim,)
#         keys = random.split(key, 6)

#         # Initialize the parameters of A
#         U = U_init(keys[0], (n_householders, input_shape[-1]))
#         log_s = log_s_init(keys[1], (out_dim,))
#         VT = VT_init(keys[2], (n_householders, out_dim))

#         # Initialize the rest of the gaussian parameters
#         b = b_init(keys[3], (x_shape[-1],))
#         log_sigma = 0.0

#         # Initialize the flow parameters
#         flow_name, flow_output_shape, flow_params, flow_state = _init_fun(keys[5], output_shape, condition_shape)

#         params = (((U, log_s, VT), (b, log_sigma)), flow_params)
#         state = ((), flow_state)
#         return (name, flow_name), flow_output_shape, params, state

#     def forward_single_x(U, log_s, VT, b, log_sigma, x, key):
#         dim_x = U.shape[-1]
#         dim_z = VT.shape[-1]
#         s = np.exp(log_s)
#         sigma = np.exp(log_sigma)

#         xmb = x - b

#         UT_xmb = householder_prod_transpose(xmb, U)[:dim_z]

#         z = householder_prod_transpose(UT_xmb/s, VT)
#         x_proj = householder_prod(np.pad(UT_xmb, (0, dim_x - dim_z)), U)

#         # Just use 1 sample
#         if(key is not None):
#             noise = random.normal(key, z.shape)
#             z = z + householder_prod_transpose(noise/s, VT)*np.sqrt(sigma)

#         log_hx = -0.5/sigma*np.dot(xmb, xmb - x_proj)
#         log_hx -= np.sum(log_s)
#         log_hx -= 0.5*(dim_x - dim_z)*(np.log(2*np.pi) + log_sigma)
#         return z, log_hx

#     def forward(params, state, log_px, x, condition, **kwargs):
#         (((U, log_s, VT), (b, log_sigma)), flow_params) = params
#         _, flow_state = state
#         dim_x, dim_z = U.shape[-1], VT.shape[-1]
#         s = np.exp(log_s)
#         sigma = np.exp(log_sigma)

#         # Find the pseudo inverse and the projection
#         xmb = x - b
#         UT_xmb = householder_prod_transpose(xmb, U)[:dim_z]
#         z = householder_prod_transpose(UT_xmb/s, VT)
#         x_proj = householder_prod(np.pad(UT_xmb, (0, dim_x - dim_z)), U)

#         # Get the terms that don't depend on z
#         log_hx = -0.5/sigma*np.dot(xmb, xmb - x_proj)
#         log_hx -= np.sum(log_s)
#         log_hx -= 0.5*(dim_x - dim_z)*(np.log(2*np.pi) + log_sigma)

#         # Sample from N(z|\mu(x),\Sigma(x))
#         key = kwargs.pop('key', None)
#         if(key is not None):
#             k1, k2 = random.split(key, 2)
#             kwargs['key'] = k2
#             n_importance_samples = kwargs.get('n_importance_samples', n_training_importance_samples)

#             noise = random.normal(k1, (n_importance_samples,) + z.shape)
#             z = z[None, ...] + householder_prod_transpose(noise/s, VT)*np.sqrt(sigma)
#         else:
#             # We're only using the mean, but put it on an axis so that we can use vmap
#             z_samples = z[None]

#         filled_forward = partial(forward_single_x, U, log_s, VT, b, log_sigma, key=k2)
#         if(x.ndim == 2):
#             filled_forward = vmap(filled_forward)
#         z, log_hx = filled_forward(x)

#         log_px, z, updated_flow_state = _forward(flow_params, flow_state, log_px + log_hx, z, condition, **kwargs)
#         return log_px, z, ((), updated_flow_state)

#     def inverse_single_z(U, log_s, VT, b, log_sigma, z, key):
#         dim_x = U.shape[-1]
#         dim_z = VT.shape[-1]
#         sigma = np.exp(log_sigma)

#         x = householder_prod(z, VT)
#         x = x*np.exp(log_s)
#         x = householder_prod(np.pad(x, (0, dim_x - dim_z)), U)
#         x += b

#         if(key is not None):
#             noise = random.normal(key, x.shape)
#             x = x + noise*np.sqrt(sigma)

#         return x

#     def inverse(params, state, log_pz, z, condition, **kwargs):
#         (((U, log_s, VT), (b, log_sigma)), flow_params) = params
#         sigma = np.exp(log_sigma)
#         _, flow_state = state

#         key = kwargs.pop('key', None)
#         test = kwargs.get('test', TRAIN)
#         if((key is not None) and (is_testing(test) == False)):
#             k1, k2 = random.split(key, 2)
#         else:
#             k2 = None

#         log_pz, z, updated_flow_state = _inverse(flow_params, flow_state, log_pz, z, condition, **kwargs)

#         filled_inverse = partial(inverse_single_z, U, log_s, VT, b, log_sigma, key=k2)
#         if(z.ndim == 2):
#             x = vmap(filled_inverse)(z)
#         else:
#             x = filled_inverse(z)

#         return log_pz, x, ((), updated_flow_state)

#     return init_fun, forward, inverse

################################################################################################################

def marginal_test(flow, x, key, n_keys=100, n_z=100, **kwargs):
    # x should have shape (1, x_dim)
    # Assumes that we just have one component of the flow.
    # Will verify that E_{p(z)}[N(x|Az+b, \Sigma)] = h(x)E_{N(z|\mu(x),\Sigma(x))}[p(z)]
    init_fun, forward, inverse = flow
    _, out_shape, params, state = init_fun(key, x.shape[1:], ())

    _, z, _ = forward(params, state, np.zeros(x.shape[0]), x, (), key=key, **kwargs)
    z_dim = z.shape[-1]

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
    z_samples = random.normal(random.split(key, 2)[1], (n_z, z_dim))
    expectation2 = vmap(inverse_w_z_and_key, in_axes=(None, 0))(z_samples, keys)
    return expectation1, expectation2