import jax
import jax.numpy as jnp
from jax import vmap, random, jit
import jax.nn.initializers as jaxinit
from functools import partial
import src.util as util
import src.flows.base as base

@base.auto_batch
def UnitGaussianPrior(name='unit_gaussian_prior'):
    # language=rst
    """
    Prior for the normalizing flow.

    :param axis - Axes to reduce over
    """
    dim = None

    def forward(params, state, inputs, **kwargs):
        x = inputs['x']
        inputs['log_det'] = -0.5*jnp.sum(x**2) + -0.5*dim*jnp.log(2*jnp.pi)
        return inputs, state

    def inverse(params, state, inputs, **kwargs):
        # Usually we're sampling z from a Gaussian, so if we want to do Monte Carlo
        # estimation, ignore the value of N(z|0,I).
        x = inputs['x']
        if(kwargs.get('sample', False)):
            inputs['log_det'] = 0.0
        else:
            inputs['log_det'] = -0.5*jnp.sum(x**2) + -0.5*dim*jnp.log(2*jnp.pi)
        return inputs, state

    def init_fun(key, input_shapes):
        x_shape = input_shapes['x']
        nonlocal dim
        dim = jnp.prod(x_shape)
        params, state = {}, {}

        output_shapes = {}
        output_shapes.update(input_shapes)
        output_shapes['log_det_shape'] = (1,)
        return base.Flow(name, input_shapes, output_shapes, params, state, forward, inverse)

    return init_fun, base.data_independent_init(init_fun)

################################################################################################################

def AffineGaussianPriorFullCov(out_dim, A_init=jaxinit.glorot_normal(), Sigma_chol_init=jaxinit.normal(), name='unnamed'):
    """ Analytic solution to int N(z|0,I)N(x|Az,Sigma)dz.
        Allows normalizing flow to start in different dimension.

        Args:
    """
    triangular_indices = None
    def init_fun(key, input_shape):
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

    def forward(params, state, x, **kwargs):
        A, Sigma_chol_flat = params
        x_dim, z_dim = A.shape

        # Need to make the diagonal positive
        Sigma_chol = Sigma_chol_flat[triangular_indices]

        diag = jnp.diag(Sigma_chol)
        Sigma_chol = index_update(Sigma_chol, jnp.diag_indices(Sigma_chol.shape[0]), jnp.exp(diag))

        # In case we want to change the noise model
        sigma = kwargs.get('sigma', 1.0)
        Sigma_chol = sigma*Sigma_chol

        Sigma_inv_A = jax.scipy.linalg.cho_solve((Sigma_chol, True), A)
        ATSA = jnp.eye(z_dim) + A.T@Sigma_inv_A
        ATSA_inv = jnp.linalg.inv(ATSA)

        if(x.ndim == 1):
            z = jnp.einsum('ij,j->i', ATSA_inv@Sigma_inv_A.T, x)
            x_proj = jnp.einsum('ij,j->i', Sigma_inv_A, z)
            a = util.upper_cho_solve(Sigma_chol, x)
        elif(x.ndim == 2):
            z = jnp.einsum('ij,bj->bi', ATSA_inv@Sigma_inv_A.T, x)
            x_proj = jnp.einsum('ij,bj->bi', Sigma_inv_A, z)
            a = vmap(partial(util.upper_cho_solve, Sigma_chol))(x)
        else:
            assert 0, 'Got an invalid shape.  x.shape: %s'%(str(x.shape))

        log_hx = -0.5*jnp.sum(x*(a - x_proj), axis=-1)
        log_hx -= 0.5*jnp.linalg.slogdet(ATSA)[1]
        log_hx -= diag.sum()
        log_hx -= 0.5*x_dim*jnp.log(2*jnp.pi)
        return log_hx, z, state

    def inverse(params, state, z, **kwargs):
        # Passing back through the network, we just need to sample from N(x|Az,Sigma).
        # Assume we have already sampled z ~ N(0,I)
        A, Sigma_chol_flat = params

        # Compute Az
        if(z.ndim == 1):
            x = jnp.einsum('ij,j->i', A, z)
        elif(z.ndim == 2):
            x = jnp.einsum('ij,bj->bi', A, z)
        else:
            assert 0, 'Got an invalid shape.  z.shape: %s'%(str(z.shape))

        Sigma_chol = Sigma_chol_flat[triangular_indices]
        diag = jnp.diag(Sigma_chol)
        Sigma_chol = jax.ops.index_update(Sigma_chol, jnp.diag_indices(Sigma_chol.shape[0]), jnp.exp(diag))

        key = kwargs.pop('key', None)
        if(key is not None):
            sigma = kwargs.get('sigma', 1.0)
            noise = random.normal(key, x.shape)*sigma
            x += jnp.dot(noise, Sigma_chol.T)
        else:
            noise = jnp.zeros_like(x)

        # Compute N(x|Az+b, Sigma)
        log_px = util.gaussian_diag_cov_logpdf(noise, jnp.zeros_like(noise), log_diag_cov)
        return log_px, x, state

    return init_fun, forward, inverse

################################################################################################################

def AffineGaussianPriorDiagCov(out_dim, A_init=jaxinit.glorot_normal(), name='unnamed'):
    """ Analytic solution to int N(z|0,I)N(x|Az,Sigma)dz.
        Allows normalizing flow to start in different dimension.

        Args:
    """
    def init_fun(key, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        A = A_init(key, (input_shape[-1], out_dim))
        log_diag_cov = jnp.zeros(input_shape[-1])
        params = (A, log_diag_cov)
        state = ()
        return name, output_shape, params, state

    def forward(params, state, x, **kwargs):
        A, log_diag_cov = params

        # In case we want to change the noise model.  This equation corresponds
        # to how we are changing noise in the inverse section
        sigma = kwargs.get('sigma', 1.0)
        log_diag_cov = log_diag_cov + 2*jnp.log(sigma)
        diag_cov = jnp.exp(log_diag_cov)

        x_dim, z_dim = A.shape
        ATSA = jnp.eye(z_dim) + (A.T/diag_cov)@A
        ATSA_inv = jnp.linalg.inv(ATSA)

        if(x.ndim == 1):
            z = jnp.einsum('ij,j->i', ATSA_inv@A.T/diag_cov, x)
            x_proj = A@z/diag_cov
        elif(x.ndim == 2):
            z = jnp.einsum('ij,bj->bi', ATSA_inv@A.T/diag_cov, x)
            x_proj = jnp.einsum('ij,bj->bi', A, z)/diag_cov
        else:
            assert 0, 'Got an invalid shape.  x.shape: %s'%(str(x.shape))

        log_hx = -0.5*jnp.sum(x*(x/diag_cov - x_proj), axis=-1)
        log_hx -= 0.5*jnp.linalg.slogdet(ATSA)[1]
        log_hx -= 0.5*log_diag_cov.sum()
        log_hx -= 0.5*x_dim*jnp.log(2*jnp.pi)

        # In case we want to retrieve the manifold penalty
        get_manifold_penalty = kwargs.get('get_manifold_penalty', False)
        if(get_manifold_penalty):
            _, mp, _ = util.tall_affine_posterior_diag_cov(x, jnp.zeros_like(x), A, log_diag_cov, 1.0)
            state = (mp,)

        return log_hx, z, state

    def inverse(params, state, z, **kwargs):
        # Passing back through the network, we just need to sample from N(x|Az,Sigma).
        # Assume we have already sampled z ~ N(0,I)
        A, log_diag_cov = params

        log_diag_cov = jnp.zeros_like(log_diag_cov)

        # Compute Az
        if(z.ndim == 1):
            x = jnp.einsum('ij,j->i', A, z)
        elif(z.ndim == 2):
            x = jnp.einsum('ij,bj->bi', A, z)
        else:
            assert 0, 'Got an invalid shape.  z.shape: %s'%(str(z.shape))

        key = kwargs.pop('key', None)
        if(key is not None):
            sigma = kwargs.get('sigma', 1.0)
            noise = random.normal(key, x.shape)*jnp.exp(0.5*log_diag_cov)*sigma
            x += noise

            # Compute N(x|Az+b, Sigma)
            log_px = util.gaussian_diag_cov_logpdf(noise, jnp.zeros_like(noise), log_diag_cov)

        else:
            noise = x*0.0

            # Otherwise we're just computing the manifold pdf
            log_px = -0.5*jnp.linalg.slogdet(A.T@A)[1]

        return log_px, x, state

    return init_fun, forward, inverse

################################################################################################################

__all__ = ['UnitGaussianPrior',
           'AffineGaussianPriorFullCov',
           'AffineGaussianPriorDiagCov']
