import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial
import haiku as hk
import src.util as util
import src.flows.base as base

################################################################################################################

@base.auto_batch
def Coupling(haiku_network=None, hidden_layer_sizes=[1024]*4, kind='affine', axis=-1, name='coupling'):
    # language=rst
    """
    Apply an arbitrary transform to half of the input vector.
    Use a squeeze before this if you want a checkerboard pattern coupling transform.

    :param network: An uninitialized Haiku network
    :param axis: Axis to split on
    """
    assert kind == 'affine' or kind == 'additive'
    network = None

    def forward(params, state, inputs, **kwargs):
        x = inputs['x']
        xa, xb = jnp.split(x, 2, axis=axis)
        network_params = params['hk_params']

        # Apply the transformation
        if(kind == 'affine'):
            t, log_s = network.apply(network_params, xb, **kwargs)
            za = xa*jnp.exp(log_s) + t
            log_det = jnp.sum(log_s)
        else:
            t = network.apply(network_params, xb, **kwargs)
            za = xa + t
            log_det = 0.0

        # Recombine
        z = jnp.concatenate([za, xb], axis=axis)

        outputs = {'x': z, 'log_det': log_det}
        return outputs, state

    def inverse(params, state, inputs, **kwargs):
        z = inputs['x']
        za, zb = jnp.split(z, 2, axis=axis)
        network_params = params['hk_params']

        # Apply the transformation
        if(kind == 'affine'):
            t, log_s = network.apply(network_params, zb, **kwargs)
            xa = (za - t)*jnp.exp(-log_s)
            log_det = jnp.sum(log_s)
        else:
            t = network.apply(network_params, zb, **kwargs)
            xa = za - t
            log_det = 0.0

        x = jnp.concatenate([xa, zb], axis=axis)

        outputs = {'x': x, 'log_det': log_det}
        return outputs, state

    def init_fun(key, input_shapes):
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
                network = hk.transform(lambda x, **kwargs: util.SimpleConv(split_input_shape, 256, kind=='additive')(x, **kwargs))
            else:
                network = hk.transform(lambda x, **kwargs: util.SimpleMLP(split_input_shape, hidden_layer_sizes, kind=='additive')(x, **kwargs))
        else:
            network = hk.transform(lambda x, **kwargs: haiku_network(split_input_shape)(x, **kwargs))

        params = {'hk_params': network.init(key, jnp.zeros(split_input_shape))}
        state = {}

        output_shapes = {}
        output_shapes.update(input_shapes)
        output_shapes['log_det'] = (1,)

        return base.Flow(name, input_shapes, output_shapes, params, state, forward, inverse)

    return init_fun, base.data_independent_init(init_fun)

################################################################################################################

__all__ = ['Coupling']