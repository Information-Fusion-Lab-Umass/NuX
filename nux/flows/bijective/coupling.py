import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial
import haiku as hk
import nux.util as util
import nux.flows.base as base

################################################################################################################

@base.auto_batch
def Coupling(haiku_network=None, hidden_layer_sizes=[1024]*4, kind='affine', axis=-1, name='coupling', n_channels=256):
    # language=rst
    """
    Apply an arbitrary transform to half of the input vector.
    Use a squeeze before this if you want a checkerboard pattern coupling transform.

    :param network: An uninitialized Haiku network
    :param axis: Axis to split on
    """
    assert kind == 'affine' or kind == 'additive'
    network = None

    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        x = inputs['x']
        xa, xb = jnp.split(x, 2, axis=axis)
        network_params = params['hk_params']

        # Apply the transformation
        if(kind == 'affine'):
            t, log_s = network.apply(network_params, None, xb, **kwargs)
            if(reverse == False):
                za = (xa - t)*jnp.exp(-log_s)
            else:
                za = xa*jnp.exp(log_s) + t
            log_det = -jnp.sum(log_s)
        else:
            t = network.apply(network_params, None, xb, **kwargs)
            if(reverse == False):
                za = xa - t
            else:
                za = xa + t
            log_det = 0.0

        # Recombine
        z = jnp.concatenate([za, xb], axis=axis)

        outputs = {'x': z, 'log_det': log_det}
        return outputs, state

    def create_params_and_state(key, input_shapes):
        x_shape = input_shapes['x']
        ax = axis % len(x_shape)
        assert x_shape[-1]%2 == 0
        half_split_dim = x_shape[ax]//2

        # Find the split shape
        split_input_shape = x_shape[:ax] + (half_split_dim,) + x_shape[ax + 1:]

        # Build the network if it isn't passed in
        nonlocal network
        if(haiku_network is None):
            if(len(x_shape) == 3):
                network = hk.transform(lambda x, **kwargs: util.SimpleConv(split_input_shape, n_channels, kind=='additive')(x, **kwargs), apply_rng=True)
            else:
                network = hk.transform(lambda x, **kwargs: util.SimpleMLP(split_input_shape, hidden_layer_sizes, kind=='additive')(x, **kwargs), apply_rng=True)
        else:
            network = hk.transform(lambda x, **kwargs: haiku_network(split_input_shape)(x, **kwargs), apply_rng=True)

        params = {'hk_params': network.init(key, jnp.zeros(split_input_shape))}
        state = {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

__all__ = ['Coupling']