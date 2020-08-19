import jax
import jax.numpy as jnp
from jax import vmap, random, jit
import jax.nn.initializers as jaxinit
from functools import partial
import nux.util as util
import nux.flows
import nux.flows.base as base
from jax.scipy.special import gammaln, logsumexp

@base.auto_batch
def UnitGaussianPrior(name='unit_gaussian_prior'):
    # language=rst
    """
    Prior for the normalizing flow.
    """
    dim = None

    def apply_fun(params,
                  state,
                  inputs,
                  reverse=False,
                  compute_base=False,
                  prior_sample=False,
                  negative_entropy=False,
                  key=None,
                  t=1.0,
                  **kwargs):
        x = inputs['x']
        outputs = {'x': x}
        if(reverse == False or compute_base == True):
            if(negative_entropy == False):
                outputs['log_pz'] = -0.5*jnp.sum(x**2)/t**2 + -0.5*dim*jnp.log(2*jnp.pi*t**2)
            else:
                outputs['log_pz'] = -0.5*dim*(1 + jnp.log(2*jnp.pi))
        else:
            outputs['log_pz'] = 0.0

        if(reverse == True and prior_sample == True):
            assert key is not None
            x = random.normal(key, x.shape)*t

            if(compute_base == True):
                if(negative_entropy == False):
                    outputs['log_pz'] = -0.5*jnp.sum(x**2)/t**2 + -0.5*dim*jnp.log(2*jnp.pi*t**2)
                else:
                    outputs['log_pz'] = -0.5*dim*(1 + jnp.log(2*jnp.pi))

            outputs['x'] = x

        return outputs, state

    def create_params_and_state(key, input_shapes):
        x_shape = input_shapes['x']
        nonlocal dim
        dim = jnp.prod(jnp.array(x_shape))
        params, state = {}, {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

@base.auto_batch
def UniformDirichletPrior(name='uniform_dirichlet_prior'):
    # language=rst
    """
    Dirichlet prior with alpha = 1.  Can optionally pass labels too.
    """
    def apply_fun(params,
                  state,
                  inputs,
                  reverse=False,
                  compute_base=False,
                  prior_sample=False,
                  key=None,
                  scale=1.0,
                  **kwargs):
        x = inputs['x']
        y = inputs.get('y', -1)
        outputs = {'x': x}

        alpha = jnp.ones_like(x)*scale

        if(reverse == False or compute_base == True):
            # Compute p(x,y) = p(y|x)p(x) if we have a label, p(x) otherwise
            outputs['log_pz'] = jax.lax.cond(y >= 0, lambda a: jnp.log(x[y]), lambda a: 0.0, None)
            outputs['log_pz'] += jnp.sum((alpha - 1)*jnp.log(x)) + gammaln(alpha.sum()) - gammaln(alpha).sum()
        else:
            outputs['log_pz'] = 0.0

        if(reverse == True and prior_sample == True):
            assert key is not None

            if(y >= 0):
                # Just sample from a dirichlet with a different alpha
                alpha = jnp.ones_like(x)
                alpha = jax.ops.index_update(alpha, y, 5)
                x = random.dirichlet(key, alpha)

                if(compute_base == True):
                    outputs['log_pz'] = jnp.log(x[y]) + jnp.sum((alpha - 1)*jnp.log(x)) + gammaln(alpha.sum()) - gammaln(alpha).sum()

            else:
                x = random.dirichlet(key, alpha)

                if(compute_base == True):
                    outputs['log_pz'] = jnp.sum((alpha - 1)*jnp.log(x)) + gammaln(alpha.sum()) - gammaln(alpha).sum()

            outputs['x'] = x

        outputs['prediction'] = jnp.argmax(x)
        return outputs, state

    def create_params_and_state(key, input_shapes):
        assert len(input_shapes['x']) == 1
        params, state = {}, {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

@base.auto_batch
def GMMPrior(n_classes, name='gmm_prior'):
    # language=rst
    """
    Gaussian mixture model prior with fixed means and covariances.  Can optionally pass labels too.
    """
    def apply_fun(params,
                  state,
                  inputs,
                  reverse=False,
                  compute_base=False,
                  prior_sample=False,
                  key=None, **kwargs):
        means, log_diag_covs = state['means'], state['log_diag_covs']
        x = inputs['x']
        y = inputs.get('y', -1)
        outputs = {'x': x}

        # Compute the log pdfs of each mixture component
        gmm = vmap(partial(util.gaussian_diag_cov_logpdf, x))
        log_pdfs = gmm(means, log_diag_covs)

        if(reverse == False or compute_base == True):
            # Compute p(x,y) = p(x|y)p(y) if we have a label, p(x) otherwise
            outputs['log_pz'] = jax.lax.cond(y >= 0,
                                             lambda a: log_pdfs[y] + jnp.log(n_classes),
                                             lambda a: logsumexp(log_pdfs) - jnp.log(n_classes),
                                             None)
        else:
            outputs['log_pz'] = 0.0

        if(reverse == True and prior_sample == True):
            assert key is not None

            if(y >= 0):
                # Sample from a certain cluster
                x = random.normal(key, x.shape) + means[y]

                if(compute_base == True):
                    # Compute the likelihoods
                    gmm = vmap(partial(util.gaussian_diag_cov_logpdf, x))
                    log_pdfs = gmm(means, log_diag_covs)

                    outputs['log_pz'] = log_pdfs[y] - jnp.log(n_classes)
            else:
                # Sample from any cluster
                y = random.randint(key, minval=0, maxval=n_classes, shape=(1,))[0]
                x = random.normal(key, x.shape) + means[y]

                if(compute_base == True):
                    outputs['log_pz'] = logsumexp(log_pdfs) - jnp.log(n_classes)

            outputs['x'] = x

        outputs['prediction'] = jnp.argmax(log_pdfs)

        return outputs, state

    def create_params_and_state(key, input_shapes):
        x_shape = input_shapes['x']
        assert len(x_shape) == 1

        means = random.normal(key, (n_classes, x_shape[-1]))
        log_diag_covs = jnp.zeros((n_classes, x_shape[-1]))

        params = {}
        state = {'means': means, 'log_diag_covs': log_diag_covs}
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
        a = util.lower_cho_solve(Sigma_chol, x)

        log_hx = -0.5*jnp.sum(x*(a - x_proj), axis=-1)
        log_hx -= 0.5*jnp.linalg.slogdet(IpL)[1]
        log_hx -= diag.sum()
        log_hx -= 0.5*x_dim*jnp.log(2*jnp.pi)

        outputs['x'] = z
        outputs['log_pz'] = log_hx
        return outputs, state

    def inverse(params, state, inputs, key=None, **kwargs):
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
        A = util.whiten(A)
        log_diag_cov = jnp.zeros(x_shape[-1])

        params = {'A': A, 'log_diag_cov': log_diag_cov}
        state = {'s': 1.0}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

__all__ = ['UnitGaussianPrior',
           'UniformDirichletPrior',
           'GMMPrior',
           'AffineGaussianPriorFullCov',
           'AffineGaussianPriorDiagCov']
