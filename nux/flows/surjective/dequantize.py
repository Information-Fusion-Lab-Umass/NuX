import jax.numpy as jnp
from jax import random
import nux.flows.base as base

@base.auto_batch
def UniformDequantization(noise_scale=None, scale=256.0, name='uniform_dequantization'):
    # language=rst
    """
    Dequantization for images.

    :param noise_scale: An array that tells us how much noise to add to each dimension
    :param scale: What to divide the image by
    """
    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        x = inputs['x']

        # Add uniform noise
        # key_name = '%s_key'%name if name != 'unnamed' else 'key'
        key = kwargs.pop('key', None)
        if(reverse == False):
            assert key is not None
        if(key is None):
            noise = jnp.zeros_like(x)
        else:
            noise = random.uniform(key, x.shape)*state['noise_scale_array']

        log_det = -jnp.log(scale)
        log_det *= jnp.prod(x.shape)

        if(reverse == False):
            z = (x + noise)/scale
        else:
            z = x*scale

        outputs = {'x': z, 'log_det': log_det}
        return outputs, state

    def create_params_and_state(key, input_shapes):
        x_shape = input_shapes['x']
        params, state = {}, {}

        if(noise_scale is None):
            state['noise_scale_array'] = jnp.ones(x_shape)
        else:
            assert noise_scale.shape == x_shape
            state['noise_scale_array'] = noise_scale

        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

__all__ = ['UniformDequantization']