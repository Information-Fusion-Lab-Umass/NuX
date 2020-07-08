import jax
import jax.numpy as jnp
from jax import vmap, random, jit
import jax.nn.initializers as jaxinit
from functools import partial
import nux.util as util
import nux.flows.base as base

@base.auto_batch
def UnitGaussianPrior(name='unit_gaussian_prior'):
    # language=rst
    """
    Prior for the normalizing flow.

    :param axis - Axes to reduce over
    """
    dim = None

    def apply_fun(params, state, inputs, reverse=False, compute_base=False, **kwargs):
        x = inputs['x']
        t = state['t']
        outputs = {'x': x}
        if(reverse == False or compute_base == True):
            outputs['log_pz'] = -0.5*jnp.sum(x**2)/t + -0.5*dim*jnp.log(2*jnp.pi)
            # outputs['log_pz'] = -0.5*jnp.sum(x**2, axis=-1)/t + -0.5*dim*jnp.log(2*jnp.pi)
        else:
            outputs['log_pz'] = 0.0
        return outputs, state

    def create_params_and_state(key, input_shapes):
        x_shape = input_shapes['x']
        nonlocal dim
        dim = jnp.prod(x_shape)
        params, state = {}, {'t': 1.0}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

@base.auto_batch
def AffineGaussianPriorFullCov(out_dim, A_init=jaxinit.glorot_normal(), Sigma_chol_init=jaxinit.normal(), name='affine_gaussian_prior_full_cov'):
    """ Analytic solution to int N(z|0,I)N(x|Az,Sigma)dz.
        Allows normalizing flow to start in different dimension.

        Args:
    """
    triangular_indices = None

    def forward(params, state, inputs, **kwargs):
        x = inputs['x']
        assert x.ndim == 1

        A, Sigma_chol_flat = params['A'], params['Sigma_chol_flat']
        x_dim, z_dim = A.shape

        # Need to make the diagonal positive
        Sigma_chol = Sigma_chol_flat[triangular_indices]

        diag = jnp.diag(Sigma_chol)
        Sigma_chol = jax.ops.index_update(Sigma_chol, jnp.diag_indices(Sigma_chol.shape[0]), jnp.exp(diag))

        # In case we want to change the noise model
        sigma = state['sigma']
        Sigma_chol = sigma*Sigma_chol

        Sigma_inv_A = jax.scipy.linalg.cho_solve((Sigma_chol, True), A)
        IpL = jnp.eye(z_dim) + A.T@Sigma_inv_A
        IpL_inv = jnp.linalg.inv(IpL)

        z = jnp.einsum('ij,j->i', IpL_inv@Sigma_inv_A.T, x)
        x_proj = jnp.einsum('ij,j->i', Sigma_inv_A, z)
        a = util.upper_cho_solve(Sigma_chol, x)

        log_hx = -0.5*jnp.sum(x*(a - x_proj), axis=-1)
        log_hx -= 0.5*jnp.linalg.slogdet(IpL)[1]
        log_hx -= diag.sum()
        log_hx -= 0.5*x_dim*jnp.log(2*jnp.pi)

        outputs['x'] = z
        outputs['log_pz'] = log_hx
        return outputs, state

    def inverse(params, state, inputs, **kwargs):
        z = inputs['x']
        assert z.ndim == 1

        A, Sigma_chol_flat = params['A'], params['Sigma_chol_flat']
        # Passing back through the network, we just need to sample from N(x|Az,Sigma).
        # Assume we have already sampled z ~ N(0,I)

        # Compute Az
        x = A@z

        Sigma_chol = Sigma_chol_flat[triangular_indices]
        diag = jnp.diag(Sigma_chol)
        Sigma_chol = jax.ops.index_update(Sigma_chol, jnp.diag_indices(Sigma_chol.shape[0]), jnp.exp(diag))

        key = kwargs.pop('key', None)
        if(key is not None):
            sigma = state['sigma']
            noise = random.normal(key, x.shape)*sigma
            x += jnp.dot(noise, Sigma_chol.T)
        else:
            noise = jnp.zeros_like(x)

        # Compute N(x|Az+b, Sigma)
        log_px = util.gaussian_chol_cov_logpdf(noise, jnp.zeros_like(noise), log_diag_cov)

        outputs['x'] = x
        outputs['log_pz'] = log_px
        return outputs, state

    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        if(reverse == False):
            return forward(params, state, inputs, **kwargs)
        return inverse(params, state, inputs, **kwargs)

    def create_params_and_state(key, input_shapes):
        x_shape = input_shapes['x']
        output_shape = x_shape[:-1] + (out_dim,)
        k1, k2 = random.split(key, 2)

        # Initialize the affine matrix
        A = A_init(k1, (x_shape[-1], out_dim))

        # Initialize the cholesky decomposition of the covariance matrix
        nonlocal triangular_indices
        dim = x_shape[-1]
        triangular_indices = util.upper_triangular_indices(dim)
        flat_dim = util.n_elts_upper_triangular(dim)
        Sigma_chol_flat = Sigma_chol_init(k2, (flat_dim,))

        params = {'A': A, 'Sigma_chol_flat': Sigma_chol_flat}
        state = {'sigma': 1.0}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

@base.auto_batch
def AffineGaussianPriorDiagCov(out_dim, A_init=jaxinit.glorot_normal(), name='affine_gaussian_prior_diag_cov'):
    """ Analytic solution to int N(z|0,I)N(x|Az,Sigma)dz.
        Allows normalizing flow to start in different dimension.

        Args:
    """
    def forward(params, state, inputs, s=1.0, **kwargs):
        x = inputs['x']
        assert x.ndim == 1
        A, log_diag_cov = params['A'], params['log_diag_cov']

        # In case we want to change the noise model.  This equation corresponds
        # to how we are changing noise in the inverse section
        # s = state['s']
        log_diag_cov = log_diag_cov + 2*jnp.log(s)
        diag_cov = jnp.exp(log_diag_cov)

        # I+Lambda
        x_dim, z_dim = A.shape
        IpL = jnp.eye(z_dim) + (A.T/diag_cov)@A

        # (I+Lambda)^{-1}
        IpL_inv = jnp.linalg.inv(IpL)

        # Compute everything else
        z = (IpL_inv@A.T/diag_cov)@x
        x_proj = A@z/diag_cov

        # Manifold penalty term
        log_hx = -0.5*jnp.sum(x*(x/diag_cov - x_proj), axis=-1)
        log_hx -= 0.5*jnp.linalg.slogdet(IpL)[1]
        log_hx -= 0.5*log_diag_cov.sum()
        log_hx -= 0.5*x_dim*jnp.log(2*jnp.pi)

        outputs = {}
        outputs['x'] = z
        outputs['log_pz'] = log_hx
        return outputs, state

    def inverse(params, state, inputs, s=1.0, t=1.0, compute_base=False, **kwargs):
        # Passing back through the network, we just need to sample from N(x|Az,Sigma).
        # Assume we have already sampled z ~ N(0,I)
        z = inputs['x']
        assert z.ndim == 1
        A, log_diag_cov = params['A'], params['log_diag_cov']

        # Compute Az
        Az = A@z

        key = kwargs.pop('key', None)
        if(key is not None):
            # Sample from N(x|Az,Sigma)
            # s = state['s']
            noise = random.normal(key, Az.shape)*jnp.exp(0.5*log_diag_cov)*s
            x = Az + noise

            # Compute N(x|Az+b, Sigma)
            log_px = util.gaussian_diag_cov_logpdf(x, Az, log_diag_cov)
        else:
            x = Az

            # Otherwise we're just using an injective flow
            log_px = -0.5*jnp.linalg.slogdet(A.T@A)[1]

        if(compute_base == True):
            log_px += -0.5*jnp.sum(z**2, axis=-1)/t + -0.5*z.shape[-1]*jnp.log(2*jnp.pi)

        outputs = {}
        outputs['x'] = x
        outputs['log_pz'] = log_px
        return outputs, state

    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        if(reverse == False):
            return forward(params, state, inputs, **kwargs)
        return inverse(params, state, inputs, **kwargs)

    def create_params_and_state(key, input_shapes):
        x_shape = input_shapes['x']
        output_shape = x_shape[:-1] + (out_dim,)
        A = A_init(key, (x_shape[-1], out_dim))
        log_diag_cov = jnp.ones(x_shape[-1])

        params = {'A': A, 'log_diag_cov': log_diag_cov}
        state = {'s': 1.0}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

__all__ = ['UnitGaussianPrior',
           'AffineGaussianPriorFullCov',
           'AffineGaussianPriorDiagCov']
