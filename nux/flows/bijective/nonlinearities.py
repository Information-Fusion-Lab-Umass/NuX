import jax.numpy as jnp
import jax
import nux.flows.base as base

@base.auto_batch
def LeakyReLU(alpha=0.01, name='leaky_relu'):
    # language=rst
    """
    Leaky ReLU

    :param alpha: Slope for negative inputs
    """
    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        x = inputs['x']

        if(reverse == False):
            z = jnp.where(x > 0, x, alpha*x)
        else:
            z = jnp.where(x > 0, x, x/alpha)

        log_dx_dz = jnp.where(x > 0, 0, jnp.log(alpha))
        log_det = log_dx_dz.sum(axis=-1)

        if(log_det.ndim > 1):
            # Then we have an image and have to sum over the height and width axes
            log_det = log_det.sum(axis=(-2, -1))

        outputs = {'x': z, 'log_det': log_det}
        return outputs, state

    def create_params_and_state(key, input_shapes):
        params, state = {}, {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

@base.auto_batch
def Sigmoid(lmbda=None, name='sigmoid'):
    # language=rst
    """
    Invertible sigmoid.  The logit function is its inverse.

    :param lmbda: For numerical stability
    """
    safe = lmbda is not None

    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        x = inputs['x']

        if(reverse == False):
            z = jax.nn.sigmoid(x)

            if(safe == True):
                z -= lmbda
                z /= 1.0 - 2*lmbda

            log_det = -(jax.nn.softplus(x) + jax.nn.softplus(-x))
        else:
            if(safe == True):
                x *= 1.0 - 2*lmbda
                x += lmbda

            z = jax.scipy.special.logit(x)
            log_det = -(jax.nn.softplus(z) + jax.nn.softplus(-z))

        if(safe == True):
            log_det -= jnp.log(1.0 - 2*lmbda)

        log_det = log_det.sum(axis=-1)

        if(log_det.ndim > 1):
            # Then we have an image and have to sum over the height and width axes
            log_det = log_det.sum(axis=(-2, -1))

        outputs = {'x': z, 'log_det': log_det}
        return outputs, state

    def create_params_and_state(key, input_shapes):
        params, state = {}, {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

@base.auto_batch
def Logit(lmbda=0.05, name='logit'):
    # language=rst
    """
    Inverse of Sigmoid

    :param lmbda: For numerical stability
    """
    safe = lmbda is not None

    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        x = inputs['x']

        if(reverse == False):
            if(safe == True):
                x *= (1.0 - 2*lmbda)
                x += lmbda
            z = jax.scipy.special.logit(x)
            log_det = (jax.nn.softplus(z) + jax.nn.softplus(-z))
        else:
            z = jax.nn.sigmoid(x)
            log_det = (jax.nn.softplus(x) + jax.nn.softplus(-x))

            if(safe == True):
                z -= lmbda
                z /= (1.0 - 2*lmbda)


        if(safe == True):
            log_det += jnp.log(1.0 - 2*lmbda)

        log_det = log_det.sum(axis=-1)
        if(log_det.ndim > 1):
            # Then we have an image and have to sum more
            log_det = log_det.sum(axis=(-2, -1))

        outputs = {'x': z, 'log_det': log_det}
        return outputs, state

    def create_params_and_state(key, input_shapes):
        params, state = {}, {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

__all__ = ['LeakyReLU',
           'Sigmoid',
           'Logit']