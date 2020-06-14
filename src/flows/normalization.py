import jax
import jax.nn.initializers as jaxinit
import jax.numpy as jnp
from jax import random
import src.util as util

def ActNorm(log_s_init=jaxinit.zeros, b_init=jaxinit.zeros, name='unnamed'):
    # language=rst
    """
    Act norm normalization.  Act norm requires a data seed in order to properly initialize
    its parameters.  This will be done at runtime.

    :param axis: Batch axis
    """

    def init_fun(key, input_shape):
        k1, k2 = random.split(key)
        log_s = log_s_init(k1, (input_shape[-1],))
        b = b_init(k2, (input_shape[-1],))

        params = (log_s, b)
        state = ()
        return name, input_shape, params, state

    def forward(params, state, x, **kwargs):
        log_s, b = params

        # Check to see if we're seeding this function
        actnorm_seed = kwargs.get('actnorm_seed', False)
        if(actnorm_seed == True):
            # The initial parameters should normalize the input
            # We want it to be normalized over the channel dimension!
            axes = tuple(jnp.arange(len(x.shape) - 1))
            mean = jnp.mean(x, axis=axes)
            std = jnp.std(x, axis=axes) + 1e-5
            log_s = jnp.log(std)
            b = mean
            updated_state = (log_s, b)
        else:
            updated_state = ()

        z = (x - b)*jnp.exp(-log_s)
        log_det = -log_s.sum()

        # Need to multiply by the height/width!
        if(z.ndim == 4 or z.ndim == 3):
            height, width, channel = z.shape[-3], z.shape[-2], z.shape[-1]
            log_det *= height*width

        return log_det, z, updated_state

    def inverse(params, state, z, **kwargs):
        log_s, b = params
        x = jnp.exp(log_s)*z + b
        log_det = -log_s.sum()

        # Need to multiply by the height/width!
        if(z.ndim == 4 or z.ndim == 3):
            height, width, channel = z.shape[-3], z.shape[-2], z.shape[-1]
            log_det *= height*width

        return log_det, x, state

    return init_fun, forward, inverse

def BatchNorm(epsilon=1e-5, alpha=0.05, beta_init=jaxinit.zeros, gamma_init=jaxinit.zeros, name='unnamed'):
    # language=rst
    """
    Invertible batch norm.

    :param axis: Batch axis
    :param epsilon: Constant for numerical stability
    :param alpha: Parameter for exponential moving average of population parameters
    """
    def init_fun(key, input_shape):
        k1, k2 = random.split(key)
        beta, log_gamma = beta_init(k1, input_shape), gamma_init(k2, input_shape)
        running_mean = jnp.zeros(input_shape)
        running_var = jnp.ones(input_shape)
        params = (beta, log_gamma)
        state = (running_mean, running_var)
        return name, input_shape, params, state

    def get_bn_params(x, test, running_mean, running_var):
        """ Update the batch norm statistics """
        if(util.is_testing(test)):
            mean, var = running_mean, running_var
        else:
            mean = jnp.mean(x, axis=0)
            var = jnp.var(x, axis=0) + epsilon
            running_mean = (1 - alpha)*running_mean + alpha*mean
            running_var = (1 - alpha)*running_var + alpha*var

        return (mean, var), (running_mean, running_var)

    def forward(params, state, x, **kwargs):
        beta, log_gamma = params
        running_mean, running_var = state

        # Check if we're training or testing
        test = kwargs['test'] if 'test' in kwargs else util.TRAIN

        # Update the running population parameters
        (mean, var), (running_mean, running_var) = get_bn_params(x, test, running_mean, running_var)

        # Normalize the inputs
        x_hat = (x - mean) / jnp.sqrt(var)
        z = jnp.exp(log_gamma)*x_hat + beta

        log_det = log_gamma.sum()#*jnp.ones((z.shape[0],))
        log_det += -0.5*jnp.log(var).sum()

        updated_state = (running_mean, running_var)
        return log_det, z, updated_state

    def inverse(params, state, z, **kwargs):
        beta, log_gamma = params
        running_mean, running_var = state

        # Check if we're training or testing
        test = kwargs['test'] if 'test' in kwargs else util.TRAIN

        # Update the running population parameters
        (mean, var), (running_mean, running_var) = get_bn_params(z, test, running_mean, running_var)

        # Normalize the inputs
        z_hat = (z - beta)*jnp.exp(-log_gamma)
        x = z_hat*jnp.sqrt(var) + mean

        log_det = log_gamma.sum()#*jnp.ones((z.shape[0],))
        log_det += -0.5*jnp.log(var).sum()

        updated_state = (running_mean, running_var)
        return log_det, x, updated_state

    return init_fun, forward, inverse

################################################################################################################

__all__ = ['ActNorm',
           'BatchNorm']
