import jax
import jax.nn.initializers as jaxinit
import jax.numpy as jnp
from jax import random
import src.util as util
import src.flows.base as base

@base.auto_batch
def ActNorm(log_s_init=jaxinit.zeros, b_init=jaxinit.zeros, name='act_norm'):

    def forward(params, state, inputs, **kwargs):
        x = inputs['x']

        z = (x - params['b'])*jnp.exp(-params['log_s'])
        log_det = -params['log_s'].sum()

        # Need to multiply by the height/width!
        if(z.ndim == 3):
            height, width, channel = z.shape[-3], z.shape[-2], z.shape[-1]
            log_det *= height*width

        outputs = {'x': z, 'log_det': log_det}
        return outputs, state

    def inverse(params, state, inputs, **kwargs):
        z = inputs['x']
        x = jnp.exp(params['log_s'])*z + params['b']
        log_det = -params['log_s'].sum()

        # Need to multiply by the height/width!
        if(z.ndim == 3):
            height, width, channel = z.shape[-3], z.shape[-2], z.shape[-1]
            log_det *= height*width

        outputs = {'x': x, 'log_det': log_det}
        return outputs, state

    def init_fun(key, input_shapes):
        x_shape = input_shapes['x']

        k1, k2 = random.split(key)
        params = {'b'    : b_init(k1, x_shape),
                  'log_s': log_s_init(k2, x_shape)}
        state = {}

        output_shapes = {}
        output_shapes.update(input_shapes)
        output_shapes['log_det'] = (1,)

        return base.Flow(name, input_shapes, output_shapes, params, state, forward, inverse)

    def data_dependent_init_fun(key, inputs, **kwargs):
        x = inputs['x']
        axes = tuple(jnp.arange(len(x.shape) - 1))
        params = {'b'    : jnp.mean(x, axis=axes),
                  'log_s': jnp.log(jnp.std(x, axis=axes) + 1e-5)}
        state = {}

        input_shapes = util.tree_shapes(inputs)
        outputs, state = forward(params, state, inputs)
        output_shapes = util.tree_shapes(outputs)

        return outputs, base.Flow(name, input_shapes, output_shapes, params, state, forward, inverse)

    return init_fun, data_dependent_init_fun

# Don't use autobatching!
def BatchNorm(epsilon=1e-5, alpha=0.05, beta_init=jaxinit.zeros, gamma_init=jaxinit.zeros, name='batch_norm'):
    # language=rst
    """
    Invertible batch norm.

    :param axis: Batch axis
    :param epsilon: Constant for numerical stability
    :param alpha: Parameter for exponential moving average of population parameters
    """
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

    def forward(params, state, inputs, **kwargs):
        x = inputs['x']
        beta, log_gamma = params['beta'], params['log_gamma']
        running_mean, running_var = state['running_mean'], state['running_var']

        # Check if we're training or testing
        test = kwargs['test'] if 'test' in kwargs else util.TRAIN

        # Update the running population parameters
        (mean, var), (running_mean, running_var) = get_bn_params(x, test, running_mean, running_var)

        # Normalize the inputs
        x_hat = (x - mean) / jnp.sqrt(var)
        z = jnp.exp(log_gamma)*x_hat + beta

        log_det = log_gamma.sum()
        log_det += -0.5*jnp.log(var).sum()

        state['running_mean'] = running_mean
        state['running_var'] = running_var
        outputs = {'x': z, 'log_det': log_det}
        return outputs, state

    def inverse(params, state, inputs, **kwargs):
        z = inputs['x']
        beta, log_gamma = params['beta'], params['log_gamma']
        running_mean, running_var = state['running_mean'], state['running_var']

        # Check if we're training or testing
        test = kwargs['test'] if 'test' in kwargs else util.TRAIN

        # Update the running population parameters
        (mean, var), (running_mean, running_var) = get_bn_params(z, test, running_mean, running_var)

        # Normalize the inputs
        z_hat = (z - beta)*jnp.exp(-log_gamma)
        x = z_hat*jnp.sqrt(var) + mean

        log_det = log_gamma.sum()
        log_det += -0.5*jnp.log(var).sum()

        state['running_mean'] = running_mean
        state['running_var'] = running_var
        outputs = {'x': x, 'log_det': log_det}
        return outputs, state

    def init_fun(key, input_shapes):
        x_shape = input_shapes['x']
        k1, k2 = random.split(key)
        beta, log_gamma = beta_init(k1, x_shape), gamma_init(k2, x_shape)
        running_mean = jnp.zeros(x_shape)
        running_var = jnp.ones(x_shape)

        params = {'beta': beta,
                  'log_gamma': log_gamma}

        state = {'running_mean': running_mean,
                 'running_var': running_var}

        output_shapes = {}
        output_shapes.update(input_shapes)
        output_shapes['log_det'] = (1,)

        return base.Flow(name, input_shapes, output_shapes, params, state, forward, inverse)

    return init_fun, base.data_independent_init(init_fun)

################################################################################################################

__all__ = ['ActNorm',
           'BatchNorm']
