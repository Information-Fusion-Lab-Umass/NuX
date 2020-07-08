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

@partial(jit, static_argnums=(1,))
def upsample(z, repeats):
    x = z
    is_batched = int(x.ndim == 2 or x.ndim == 4)
    for i, r in enumerate(repeats):
        x = jnp.repeat(x, r, axis=i + is_batched)
    return x

@partial(jit, static_argnums=(1,))
def magic(x, repeats):
    hr, wr, cr = repeats
    assert cr == 1
    Hx, Wx, C = x.shape
    assert Hx%hr == 0 and Wx%wr == 0
    H, W = Hx//hr, Wx//wr
    return x.reshape((H, hr, W, wr, C)).transpose((0, 2, 4, 1, 3)).reshape((H, W, C, hr*wr)).sum(axis=-1)

################################################################################################################

@partial(jit, static_argnums=(1,))
def stochastic_inverse(x, repeats, b, log_diag_cov):
    """ Posterior of N(x|Az + b, Sigma) where A is an upsample matrix defined by repeats"""
    assert x.shape == b.shape
    assert x.shape == log_diag_cov.shape
    assert x.ndim == 3
    xmb = x - b
    one_over_diag_cov = jnp.exp(-log_diag_cov)

    # Compute the diagonal of the riemannian metric.  This is the diagonal of (A^T Sigma^{-1} A)
    hr, wr, cr = repeats; assert cr == 1 # Haven't tested cr != 1
    Hx, Wx, C = x.shape
    H, W = Hx//hr, Wx//wr
    rm_diag = magic(one_over_diag_cov, repeats)

    # Compute the mean of z
    z_mean = magic(xmb*one_over_diag_cov, repeats)/rm_diag
    x_proj = upsample(z_mean, repeats)*one_over_diag_cov
    dim_x = jnp.prod(x.shape)
    dim_z = jnp.prod(z_mean.shape)

    # Compute the manifold error term
    log_hx = -0.5*jnp.sum(xmb*(xmb*one_over_diag_cov - x_proj))
    log_hx -= 0.5*jnp.sum(jnp.log(rm_diag))
    log_hx -= 0.5*log_diag_cov.sum()
    log_hx -= 0.5*(dim_x - dim_z)*jnp.log(2*jnp.pi)

    return z_mean, log_hx, rm_diag

################################################################################################################

@partial(jit, static_argnums=(1,))
def sample_stochastic_inverse(x, repeats, b, log_diag_cov, s, key):
    # language=rst
    """
    Sample from the stochastic inverse of N(x|Az + b, diag(exp(log_diag_cov)))
    """

    # Sample from the stochastic inverse q(z|x)
    z, log_hx, rm_diag = stochastic_inverse(x, repeats, b, log_diag_cov)

    # Sample z and compute the manifold term
    if(key is not None):
        noise = random.normal(key, z.shape)/jnp.sqrt(rm_diag)
        z += noise
        log_det = log_hx
    else:
        # Treat this as an injective flow and use the log determinant
        log_det = jnp.prod(z.shape)/jnp.prod(repeats)

    # Compute the reconstruction error and stochastic inverse likelihood
    if(key is not None):
        mean = upsample(z, repeats) + b
        log_pxgz = util.gaussian_diag_cov_logpdf(x.ravel(), mean.ravel(), log_diag_cov.ravel())
        log_qzgx = util.gaussian_diag_cov_logpdf(noise.ravel(), jnp.zeros_like(noise.ravel()), -jnp.log(rm_diag).ravel())
    else:
        log_pxgz = 0.0
        log_qzgx = 0.0

    outputs = {'x': z, 'log_det': log_det, 'log_pxgz': log_pxgz, 'log_qzgx': log_qzgx}
    return outputs

@partial(jit, static_argnums=(1,))
def generate(z, repeats, b, log_diag_cov, s, key):
    # language=rst
    """
    Sample from N(x|Az + b, diag(exp(log_diag_cov)))
    """

    # Transform onto the manifold
    x = upsample(z, repeats) + b

    # Sample from N(x|Az + b, Sigma)
    if(key is not None):
        noise = jnp.exp(0.5*log_diag_cov)*random.normal(key, x.shape)*s
        x += noise

        # Compute the likelihood p(x|z)
        log_det = util.gaussian_diag_cov_logpdf(noise.ravel(), jnp.zeros_like(noise.ravel()), log_diag_cov.ravel())
    else:
        # Treat this as an injective flow and use the log determinant
        log_det = jnp.prod(z.shape)/jnp.prod(repeats)

    outputs = {'x': x, 'log_det': log_det}

    return outputs

@partial(jit, static_argnums=(2,))
def forward_mc(z, target_x, repeats, b, log_diag_cov, s):
    # language=rst
    """
    Compute N(target_x|Az + b, diag(exp(log_diag_cov)))
    """
    # Transform onto the manifold
    x = upsample(z, repeats) + b

    # Evaluate the log pdf of a given x.  This is used for testing.
    log_pxgz = util.gaussian_diag_cov_logpdf(target_x.ravel(), x.ravel(), log_diag_cov.ravel())

    outputs = {'x': x, 'log_pxgz': log_pxgz}
    return outputs

################################################################################################################

@base.auto_batch
def UpSample(repeats=(2, 2, 1), name='upsample'):
    # language=rst
    """
    Up sample by just repeating consecutive values over specified axes

    :param repeats - The number of times to repeat.  Pass in (2, 1, 2), for example, to repeat twice over
                     the 0th axis, no repeats over the 1st axis, and twice over the 2nd axis
    """
    full_repeats = None

    def apply_fun(params, state, inputs, reverse=False, forward_monte_carlo=False, key=None, **kwargs):
        b, log_diag_cov = params['b'], params['log_diag_cov']
        s = state['s']

        if(reverse == False):
            outputs = sample_stochastic_inverse(inputs['x'], full_repeats, b, log_diag_cov, s, key)
        else:
            if(forward_monte_carlo):
                outputs = forward_mc(inputs['x'], inputs['target_x'], full_repeats, b, log_diag_cov, s)
            else:
                outputs = generate(inputs['x'], full_repeats, b, log_diag_cov, s, key)

        return outputs, state

    def create_params_and_state(key, input_shapes):
        x_shape = input_shapes['x']
        H, W, C = x_shape

        # If we pass in (2, 2) for repeats, interpret this as meaning (2, 2, 1)
        nonlocal full_repeats
        full_repeats = [repeats[i] if i < len(repeats) else 1 for i in range(len(x_shape))]

        # Model is VERY sensitive to these parameters
        log_diag_cov = jnp.zeros(x_shape)
        b = jnp.zeros(x_shape)

        params, state = {'log_diag_cov': log_diag_cov, 'b': b}, {'s': 1.0}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

@base.auto_batch
def CouplingUpSample(repeats=(2, 2, 1),
                     haiku_network1=None,
                     haiku_network2=None,
                     n_channels=256,
                     A_init=jaxinit.glorot_normal(),
                     name='coupling_upsample'):
    ### p(x1, x2) = \int \int p(z1, z2)N(x1|A1@z1+b(x2),\Sigma(x2))N(x2|A2@z2+b(z1),\Sigma(z1))dz1 dz2
    """ General change of dimension.

        Args:
    """
    full_repeats = None
    network1, network2 = None, None

    def apply_fun(params, state, inputs, reverse=False, forward_monte_carlo=False, key=None, **kwargs):
        network_params1, network_params2 = params['hk_params1'], params['hk_params2']
        s = state['s']
        k1, k2 = random.split(key, 2) if key is not None else (None,)*2

        if(reverse == False):
            x = inputs['x']

            # Split x
            x_squeezed = util.dilated_squeeze(x, (2, 2), (1, 1))
            x1, x2 = jnp.split(x_squeezed, 2, axis=-1)

            # Compute the bias and covariance conditioned on x2
            b1, log_diag_cov1 = network1.apply(network_params1, x2, **kwargs)
            outputs1 = sample_stochastic_inverse(x1, full_repeats, b1, log_diag_cov1, s, k1)
            z1 = outputs1.pop('x')

            # Compute the bias and covariance conditioned on z1.  Upsample so that the shapes work out
            b2, log_diag_cov2 = network2.apply(network_params2, upsample(z1, full_repeats), **kwargs)
            outputs2 = sample_stochastic_inverse(x2, full_repeats, b2, log_diag_cov2, s, k2)
            z2 = outputs2.pop('x')

            # Combine the results
            z_squeezed = jnp.concatenate([z1, z2], axis=-1)
            z = util.dilated_unsqueeze(z_squeezed, (2, 2), (1, 1))

            outputs = jax.tree_util.tree_multimap(lambda x, y: x + y, outputs1, outputs2)
            outputs['x'] = z

        else:

            # If we're doing forward monte carlo, need to split the target output
            if(forward_monte_carlo):
                target_x = inputs['target_x']
                x_squeezed = util.dilated_squeeze(target_x, (2, 2), (1, 1))
                target_x1, target_x2 = jnp.split(x_squeezed, 2, axis=-1)

            # Split z
            z = inputs['x']
            z_squeezed = util.dilated_squeeze(z, (2, 2), (1, 1))
            z1, z2 = jnp.split(z_squeezed, 2, axis=-1)

            # Compute the bias and covariance conditioned on z1.  Upsample so that the shapes work out
            b2, log_diag_cov2 = network2.apply(network_params2, upsample(z1, full_repeats), **kwargs)

            if(forward_monte_carlo):
                outputs2 = forward_mc(z2, target_x2, full_repeats, b2, log_diag_cov2, s)
            else:
                outputs2 = generate(z2, full_repeats, b2, log_diag_cov2, s, k2)
            x2 = outputs2.pop('x')

            # Compute the bias and covariance conditioned on x2
            b1, log_diag_cov1 = network1.apply(network_params1, x2, **kwargs)
            if(forward_monte_carlo):
                outputs1 = forward_mc(z1, target_x1, full_repeats, b1, log_diag_cov1, s)
            else:
                outputs1 = generate(z1, full_repeats, b1, log_diag_cov1, s, k1)
            x1 = outputs1.pop('x')

            # Combine the results
            x_squeezed = jnp.concatenate([x1, x2], axis=-1)
            x = util.dilated_unsqueeze(x_squeezed, (2, 2), (1, 1))

            outputs = jax.tree_util.tree_multimap(lambda x, y: x + y, outputs1, outputs2)
            outputs['x'] = x

        return outputs, state

    def create_params_and_state(key, input_shapes):
        x_shape = input_shapes['x']

        keys = random.split(key, 3)

        nonlocal full_repeats
        full_repeats = [repeats[i] if i < len(repeats) else 1 for i in range(len(x_shape))]
        z_shape = []
        for s, r in zip(x_shape, full_repeats):
            assert s%r == 0
            z_shape.append(s//r)
        z_shape = tuple(z_shape)
        Hz, Wz, Cz = z_shape
        Hx, Wx, Cx = x_shape

        # Going to be squeezing the splitting on channel axis
        sq_z1_shape = (Hz//2, Wz//2, Cz*2)
        sq_z2_shape = (Hz//2, Wz//2, Cz*2)

        sq_x1_shape = (Hx//2, Wx//2, Cx*2)
        sq_x2_shape = (Hx//2, Wx//2, Cx*2)

        # Initialize each of the conditioner networks
        nonlocal network1
        input_shape1 = sq_x2_shape
        output_shape1 = sq_x1_shape
        if(haiku_network1 is None):
            network1 = hk.transform(lambda x, **kwargs: util.SimpleConv(output_shape1, n_channels, is_additive=False)(x, **kwargs))
        else:
            network1 = hk.transform(lambda x, **kwargs: haiku_network(output_shape1)(x, **kwargs))
        hk_params1 = network1.init(keys[2], jnp.zeros(input_shape1))

        nonlocal network2
        input_shape2 = sq_z1_shape
        output_shape2 = sq_x2_shape
        if(haiku_network2 is None):
            network2 = hk.transform(lambda x, **kwargs: util.SimpleConv(output_shape2, n_channels, is_additive=False)(x, **kwargs))
        else:
            network2 = hk.transform(lambda x, **kwargs: haiku_network(output_shape2)(x, **kwargs))
        hk_params2 = network2.init(keys[2], jnp.zeros(input_shape2))

        # Compile the parameters and state
        params = {'hk_params1': hk_params1, 'hk_params2': hk_params2}
        state = {'s': 1.0}

        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

__all__ = ['UpSample',
           'CouplingUpSample']