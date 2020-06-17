import jax.numpy as jnp
from jax import random
import src.flows.base as base

@base.auto_batch
def UniformDequantization(noise_scale=None, scale=256.0, name='uniform_dequantization'):
    # language=rst
    """
    Dequantization for images.

    :param noise_scale: An array that tells us how much noise to add to each dimension
    :param scale: What to divide the image by
    """
    def forward(params, state, inputs, **kwargs):
        x = inputs['x']

        # Add uniform noise
        key_name = '%s_key'%name if name != 'unnamed' else 'key'
        key = kwargs.pop(key_name, None)
        if(key is None):
            noise = jnp.zeros_like(x)
        else:
            noise = random.uniform(key, x.shape)*state['noise_scale_array']

        log_det = -jnp.log(scale)
        log_det *= jnp.prod(x.shape)

        z = (x + noise)/scale

        outputs = {'x': z, 'log_det': log_det}
        return outputs, state

    def inverse(params, state, inputs, **kwargs):
        z = inputs['x']

        # Put the image back on the set of integers between 0 and 255
        x = z*scale
        # x = jnp.floor(z*scale).astype(jnp.int32)

        log_det = -jnp.log(scale)
        log_det *= jnp.prod(z.shape)

        outputs = {'x': x, 'log_det': log_det}
        return outputs, state

    def init_fun(key, input_shapes):
        x_shape = input_shapes['x']
        params, state = {}, {}

        if(noise_scale is None):
            state['noise_scale_array'] = jnp.ones(x_shape)
        else:
            assert noise_scale.shape == x_shape
            state['noise_scale_array'] = noise_scale

        output_shapes = {}
        output_shapes.update(input_shapes)
        output_shapes['log_det'] = (1,)

        return base.Flow(name, input_shapes, output_shapes, params, state, forward, inverse)

    return init_fun, base.data_independent_init(init_fun)

################################################################################################################

__all__ = ['UniformDequantization']