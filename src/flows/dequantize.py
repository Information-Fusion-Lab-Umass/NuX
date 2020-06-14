import jax.numpy as jnp
from jax import random

def UniformDequantization(noise_scale=None, scale=256.0, name='unnamed'):
    # language=rst
    """
    Dequantization for images.

    :param noise_scale: An array that tells us how much noise to add to each dimension
    :param scale: What to divide the image by
    """
    noise_scale_array = None
    def init_fun(key, input_shape):
        params, state = (), ()
        if(noise_scale is None):
            nonlocal noise_scale_array
            noise_scale_array = jnp.ones(input_shape)
        else:
            assert noise_scale.shape == input_shape
            noise_scale_array = noise_scale
        return name, input_shape, params, state

    def forward(params, state, x, **kwargs):
        # Add uniform noise
        key_name = '%s_key'%name if name != 'unnamed' else 'key'
        key = kwargs.pop(key_name, None)
        if(key is None):
            noise = jnp.zeros_like(x)
        else:
            noise = random.uniform(key, x.shape)*noise_scale_array

        log_det = -jnp.log(scale)
        log_det *= jnp.prod(x.shape)

        return log_det, (x + noise)/scale, state

    def inverse(params, state, z, **kwargs):
        # Put the image back on the set of integers between 0 and 255
        z = z*scale
        # z = jnp.floor(z*scale).astype(jnp.int32)

        log_det = -jnp.log(scale)
        log_det *= jnp.prod(z.shape)

        return log_det, z, state

    return init_fun, forward, inverse

################################################################################################################

__all__ = ['UniformDequantization']