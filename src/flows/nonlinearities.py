import jax.numpy as jnp
import jax
import src.flows.base as base

@base.auto_batch
def LeakyReLU(alpha=0.01, name='leaky_relu'):
    # language=rst
    """
    Leaky ReLU

    :param alpha: Slope for negative inputs
    """
    def forward(params, state, inputs, **kwargs):
        x = inputs['x']
        z = jnp.where(x > 0, x, alpha*x)

        log_dx_dz = jnp.where(x > 0, 0, jnp.log(alpha))
        log_det = log_dx_dz.sum(axis=-1)

        if(log_det.ndim > 1):
            # Then we have an image and have to sum over the height and width axes
            log_det = log_det.sum(axis=(-2, -1))

        outputs = {'x': z, 'log_det': log_det}
        return outputs, state

    def inverse(params, state, inputs, **kwargs):
        z = inputs['x']
        x = jnp.where(z > 0, z, z/alpha)

        log_dx_dz = jnp.where(z > 0, 0, jnp.log(alpha))
        log_det = log_dx_dz.sum(axis=-1)

        if(log_det.ndim > 1):
            # Then we have an image and have to sum over the height and width axes
            log_det = log_det.sum(axis=(-2, -1))

        outputs = {'x': x, 'log_det': log_det}
        return outputs, state

    def init_fun(key, input_shapes):
        params, state = {}, {}

        output_shapes = {}
        output_shapes.update(input_shapes)
        output_shapes['log_det'] = (1,)

        return base.Flow(name, input_shapes, output_shapes, params, state, forward, inverse)

    return init_fun, base.data_independent_init(init_fun)

################################################################################################################

@base.auto_batch
def Sigmoid(lmbda=None, name='sigmoid'):
    # language=rst
    """
    Invertible sigmoid.  The logit function is its inverse.

    :param lmbda: For numerical stability
    """
    safe = lmbda is not None

    def forward(params, state, inputs, **kwargs):
        x = inputs['x']
        z = jax.nn.sigmoid(x)
        log_det = -(jax.nn.softplus(x) + jax.nn.softplus(-x))

        if(safe == True):
            z -= lmbda
            z /= 1.0 - 2*lmbda
            log_det -= jnp.log(1.0 - 2*lmbda)

        log_det = log_det.sum(axis=-1)

        if(log_det.ndim > 1):
            # Then we have an image and have to sum over the height and width axes
            log_det = log_det.sum(axis=(-2, -1))

        outputs = {'x': z, 'log_det': log_det}
        return outputs, state

    def inverse(params, state, inputs, **kwargs):
        z = inputs['x']
        if(safe == True):
            z *= 1.0 - 2*lmbda
            z += lmbda

        x = jax.scipy.special.logit(z)
        log_det = -(jax.nn.softplus(x) + jax.nn.softplus(-x))

        if(safe == True):
            log_det -= jnp.log(1.0 - 2*lmbda)

        log_det = log_det.sum(axis=-1)

        if(log_det.ndim > 1):
            # Then we have an image and have to sum over the height and width axes
            log_det = log_det.sum(axis=(-2, -1))

        outputs = {'x': x, 'log_det': log_det}
        return outputs, state

    def init_fun(key, input_shapes):
        params, state = {}, {}

        output_shapes = {}
        output_shapes.update(input_shapes)
        output_shapes['log_det'] = (1,)

        return base.Flow(name, input_shapes, output_shapes, params, state, forward, inverse)

    return init_fun, base.data_independent_init(init_fun)

@base.auto_batch
def Logit(lmbda=0.05, name='logit'):
    # language=rst
    """
    Inverse of Sigmoid

    :param lmbda: For numerical stability
    """
    safe = lmbda is not None

    def forward(params, state, inputs, **kwargs):
        x = inputs['x']
        if(safe == True):
            x *= (1.0 - 2*lmbda)
            x += lmbda

        z = jax.scipy.special.logit(x)
        log_det = (jax.nn.softplus(z) + jax.nn.softplus(-z))

        if(safe == True):
            log_det += jnp.log(1.0 - 2*lmbda)

        log_det = log_det.sum(axis=-1)
        if(log_det.ndim > 1):
            # Then we have an image and have to sum more
            log_det = log_det.sum(axis=(-2, -1))

        outputs = {'x': z, 'log_det': log_det}
        return outputs, state

    def inverse(params, state, inputs, **kwargs):
        z = inputs['x']
        x = jax.nn.sigmoid(z)
        log_det = (jax.nn.softplus(z) + jax.nn.softplus(-z))

        if(safe == True):
            x -= lmbda
            x /= (1.0 - 2*lmbda)
            log_det += jnp.log(1.0 - 2*lmbda)

        log_det = log_det.sum(axis=-1)
        if(log_det.ndim > 1):
            # Then we have an image and have to sum more
            log_det = log_det.sum(axis=(-2, -1))

        outputs = {'x': x, 'log_det': log_det}
        return outputs, state

    def init_fun(key, input_shapes):
        params, state = {}, {}

        output_shapes = {}
        output_shapes.update(input_shapes)
        output_shapes['log_det'] = (1,)

        return base.Flow(name, input_shapes, output_shapes, params, state, forward, inverse)

    return init_fun, base.data_independent_init(init_fun)

################################################################################################################

__all__ = ['LeakyReLU',
           'Sigmoid',
           'Logit']