import jax.numpy as jnp
from jax import vmap
from functools import partial
import nux.util as util
import nux.flows.base as base

################################################################################################################

@base.auto_batch
def Squeeze(name='squeeze'):
    # language=rst
    """
    Squeeze transformation
    """
    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        x = inputs['x']
        if(reverse == False):
            z = util.dilated_squeeze(x, (2, 2), (1, 1))
        else:
            z = util.dilated_unsqueeze(x, (2, 2), (1, 1))
        outputs = {'x': z, 'log_det': 0.0}
        return outputs, state

    def create_params_and_state(key, input_shapes):
        H, W, C = input_shapes['x']
        assert H%2 == 0
        assert W%2 == 0
        params, state = {}, {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

@base.auto_batch
def UnSqueeze(name='unsqueeze'):
    # language=rst
    """
    Squeeze transformation
    """
    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        x = inputs['x']
        if(reverse == True):
            z = util.dilated_squeeze(x, (2, 2), (1, 1))
        else:
            z = util.dilated_unsqueeze(x, (2, 2), (1, 1))
        outputs = {'x': z, 'log_det': 0.0}
        return outputs, state

    def create_params_and_state(key, input_shapes):
        H, W, C = input_shapes['x']
        assert C%4 == 0
        params, state = {}, {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

@base.auto_batch
def Transpose(axis_order, name='transpose'):
    # language=rst
    """
    Transpose an input
    """
    order = None
    order_inverse = None

    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        x = inputs['x']
        if(reverse == False):
            z = x.transpose(order)
        else:
            z = x.transpose(order_inverse)
        outputs = {'x': z, 'log_det': 0.0}
        return outputs, state

    def create_params_and_state(key, input_shapes):
        x_shape = input_shapes['x']
        nonlocal order
        order = [ax%len(axis_order) for ax in axis_order]
        assert len(order) == len(x_shape)
        assert len(set(order)) == len(order)

        nonlocal order_inverse
        order_inverse = [order.index(i) for i in range(len(order))]
        params, state = {}, {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

@base.auto_batch
def Reshape(shape, name='reshape'):
    # language=rst
    """
    Prior for the normalizing flow.

    :param shape - Shape to reshape to
    """

    # Need to keep track of the original shape in order to invert
    original_shape = None
    assert len([1 for s in shape if s < 0]) < 2, 'Can only have 1 negative shape'
    has_negative1 = jnp.any(jnp.array(shape) < 0)

    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        x = inputs['x']
        if(reverse == False):
            z = x.reshape(shape)
        else:
            z = x.reshape(original_shape)
        outputs = {'x': z, 'log_det': 0.0}
        return outputs, state

    def create_params_and_state(key, input_shapes):
        x_shape = input_shapes['x']
        # If there is a negative 1, then figure out what to do
        nonlocal shape
        if(has_negative1):
            total_dim = jnp.prod(x_shape)
            given_dim = jnp.prod([s for s in shape if s > 0])
            remaining_dim = total_dim//given_dim
            shape = [s if s > 0 else remaining_dim for s in shape]

        nonlocal original_shape
        original_shape = x_shape
        assert jnp.prod(x_shape) == jnp.prod(shape), 'x_shape %s shape: %s'%(str(x_shape), str(shape))

        params, state = {}, {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

@base.auto_batch
def Flatten(name='flatten'):
    # Need to keep track of the original shape in order to invert
    original_shape = None
    shape = None

    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        x = inputs['x']
        if(reverse == False):
            z = x.reshape(shape)
        else:
            z = x.reshape(original_shape)

        outputs = {'x': z, 'log_det': 0.0}
        return outputs, state

    def create_params_and_state(key, input_shapes):
        x_shape = input_shapes['x']
        nonlocal shape, original_shape
        original_shape = x_shape
        shape = (jnp.prod(x_shape),)
        assert jnp.prod(x_shape) == jnp.prod(shape), 'x_shape %s shape: %s'%(str(x_shape), str(shape))

        params, state = {}, {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

@base.auto_batch
def Reverse(name='reverse'):
    # language=rst
    """
    Reverse an input.
    """
    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        x = inputs['x']
        z = x[...,::-1]
        outputs = {'x': z, 'log_det': 0.0}
        return outputs, state

    def create_params_and_state(key, input_shapes):
        params, state = {}, {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

__all__ = ['Squeeze',
           'UnSqueeze',
           'Transpose',
           'Reshape',
           'Flatten',
           'Reverse']
