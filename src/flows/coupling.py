import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial
import haiku as hk
import src.util as util

################################################################################################################

def Coupling(haiku_network=None, hidden_layer_sizes=[1024]*4, kind='affine', axis=-1, name='unnamed'):
    # language=rst
    """
    Apply an arbitrary transform to half of the input vector.
    Use a squeeze before this if you want a checkerboard pattern coupling transform.

    :param network: An uninitialized Haiku network
    :param axis: Axis to split on
    """
    assert kind == 'affine' or kind == 'additive'
    network = None

    def init_fun(key, input_shape):
        ax = axis % len(input_shape)
        assert input_shape[-1]%2 == 0
        half_split_dim = input_shape[ax]//2

        # Find the split shape
        split_input_shape = input_shape[:ax] + (half_split_dim,) + input_shape[ax + 1:]

        # Build the network if it isn't passed in
        nonlocal network
        if(haiku_network is None):
            if(len(input_shape) == 3):
                network = hk.transform(lambda x, **kwargs: util.SimpleConv(split_input_shape, 256, kind=='additive')(x, **kwargs))
            else:
                network = hk.transform(lambda x, **kwargs: util.SimpleMLP(split_input_shape, hidden_layer_sizes, kind=='additive')(x, **kwargs))
        else:
            network = hk.transform(lambda x, **kwargs: haiku_network(split_input_shape)(x, **kwargs))

        params = network.init(key, jnp.zeros(split_input_shape))

        return name, input_shape, params, ()

    def forward(params, state, x, **kwargs):
        xa, xb = jnp.split(x, 2, axis=axis)

        # Apply the transformation
        if(kind == 'affine'):
            t, log_s = network.apply(params, xb, **kwargs)
            za = xa*jnp.exp(log_s) + t
            log_det = jnp.sum(log_s)
        else:
            t = network.apply(params, xb, **kwargs)
            za = xa + t
            log_det = 0.0

        # Recombine
        z = jnp.concatenate([za, xb], axis=axis)
        return log_det, z, state

    def inverse(params, state, z, **kwargs):
        za, zb = jnp.split(z, 2, axis=axis)

        # Apply the transformation
        if(kind == 'affine'):
            t, log_s = network.apply(params, zb, **kwargs)
            xa = (za - t)*jnp.exp(-log_s)
            log_det = jnp.sum(log_s)
        else:
            t = network.apply(params, zb, **kwargs)
            xa = za - t
            log_det = 0.0

        x = jnp.concatenate([xa, zb], axis=axis)
        return log_det, x, state

    return init_fun, forward, inverse

################################################################################################################

__all__ = ['Coupling']