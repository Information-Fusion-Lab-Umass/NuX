import jax.numpy as jnp
import jax

def LeakyReLU(alpha=0.01, name='unnamed'):
    # language=rst
    """
    Leaky ReLU

    :param alpha: Slope for negative inputs
    """
    def init_fun(key, input_shape):
        params, state = (), ()
        return name, input_shape, params, state

    def forward(params, state, x, **kwargs):
        z = jnp.where(x > 0, x, alpha*x)

        log_dx_dz = jnp.where(x > 0, 0, jnp.log(alpha))
        log_det = log_dx_dz.sum(axis=-1)

        if(log_det.ndim > 1):
            # Then we have an image and have to sum over the height and width axes
            log_det = log_det.sum(axis=(-2, -1))

        return log_det, z, state

    def inverse(params, state, z, **kwargs):
        x = jnp.where(z > 0, z, z/alpha)

        log_dx_dz = jnp.where(z > 0, 0, jnp.log(alpha))
        log_det = log_dx_dz.sum(axis=-1)

        if(log_det.ndim > 1):
            # Then we have an image and have to sum over the height and width axes
            log_det = log_det.sum(axis=(-2, -1))

        return log_det, x, state

    return init_fun, forward, inverse

################################################################################################################

def Sigmoid(lmbda=None, name='unnamed'):
    # language=rst
    """
    Invertible sigmoid.  The logit function is its inverse.  Remember to apply sigmoid before logit so that
    the input ranges are as expected!

    :param lmbda: For numerical stability
    """
    safe = lmbda is not None
    def init_fun(key, input_shape):
        params, state = (), ()
        return name, input_shape, params, state

    def forward(params, state, x, **kwargs):
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

        return log_det, z, state

    def inverse(params, state, z, **kwargs):
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

        return log_det, x, state

    return init_fun, forward, inverse

def Logit(lmbda=0.05, name='unnamed'):
    # language=rst
    """
    Inverse of Sigmoid

    :param lmbda: For numerical stability
    """
    safe = lmbda is not None
    def init_fun(key, input_shape):
        params, state = (), ()
        return name, input_shape, params, state

    def forward(params, state, x, **kwargs):

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
        return log_det, z, state

    def inverse(params, state, z, **kwargs):

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

        return log_det, x, state

    return init_fun, forward, inverse

################################################################################################################

__all__ = ['LeakyReLU',
           'Sigmoid',
           'Logit']