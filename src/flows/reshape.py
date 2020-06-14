import jax.numpy as jnp
from jax import vmap
from functools import partial
import src.util as util

################################################################################################################

def Squeeze(name='unnamed'):
    # language=rst
    """
    """
    def init_fun(key, input_shape):
        H, W, C = input_shape
        assert H%2 == 0
        assert W%2 == 0
        output_shape = (H//2, W//2, C*4)
        params, state = (), ()
        return name, output_shape, params, state

    def forward(params, state, x, **kwargs):
        z = util.dilated_squeeze(x, (2, 2), (1, 1))
        return 0.0, z, state

    def inverse(params, state, z, **kwargs):
        x = util.dilated_unsqueeze(z, (2, 2), (1, 1))
        return 0.0, x, state

    return init_fun, forward, inverse

def UnSqueeze(name='unnamed'):
    # language=rst
    """
    """
    def init_fun(key, input_shape):
        H, W, C = input_shape
        assert C%4 == 0
        output_shape = (H*2, W*2, C//4)
        params, state = (), ()
        return name, output_shape, params, state

    def forward(params, state, x, **kwargs):
        z = util.dilated_unsqueeze(x, (2, 2), (1, 1))
        return 0.0, z, state

    def inverse(params, state, z, **kwargs):
        x = util.dilated_squeeze(z, (2, 2), (1, 1))
        return 0.0, x, state

    return init_fun, forward, inverse

################################################################################################################

def Transpose(axis_order, name='unnamed'):
    # language=rst
    """
    Transpose an input
    """
    order = None
    order_inverse = None

    def init_fun(key, input_shape):
        nonlocal order
        order = [ax%len(axis_order) for ax in axis_order]
        assert len(order) == len(input_shape)
        assert len(set(order)) == len(order)
        params, state = (), ()
        output_shape = [input_shape[ax] for ax in order]

        nonlocal order_inverse
        order_inverse = [order.index(i) for i in range(len(order))]

        return name, output_shape, params, state

    def forward(params, state, x, **kwargs):
        z = x.transpose(order)
        return 0.0, z, state

    def inverse(params, state, z, **kwargs):
        x = z.transpose(order_inverse)
        return 0.0, x, state

    return init_fun, forward, inverse

################################################################################################################

def Reshape(shape, name='unnamed'):
    # language=rst
    """
    Prior for the normalizing flow.

    :param shape - Shape to reshape to
    """

    # Need to keep track of the original shape in order to invert
    original_shape = None
    assert len([1 for s in shape if s < 0]) < 2, 'Can only have 1 negative shape'
    has_negative1 = jnp.any(jnp.array(shape) < 0)

    def init_fun(key, input_shape):
        # If there is a negative 1, then figure out what to do
        nonlocal shape
        if(has_negative1):
            total_dim = jnp.prod(input_shape)
            given_dim = jnp.prod([s for s in shape if s > 0])
            remaining_dim = total_dim//given_dim
            shape = [s if s > 0 else remaining_dim for s in shape]

        nonlocal original_shape
        original_shape = input_shape
        assert jnp.prod(input_shape) == jnp.prod(shape), 'input_shape %s shape: %s'%(str(input_shape), str(shape))
        params, state = (), ()
        return name, shape, params, state

    def forward(params, state, x, **kwargs):
        z = x.reshape(shape)
        return 0.0, z, state

    def inverse(params, state, z, **kwargs):
        x = z.reshape(original_shape)
        return 0.0, x, state

    return init_fun, forward, inverse

################################################################################################################

def Flatten(name='unnamed'):
    # Need to keep track of the original shape in order to invert
    original_shape = None
    shape = None

    def init_fun(key, input_shape):
        nonlocal shape, original_shape
        original_shape = input_shape
        shape = (jnp.prod(input_shape),)
        assert jnp.prod(input_shape) == jnp.prod(shape), 'input_shape %s shape: %s'%(str(input_shape), str(shape))
        params, state = (), ()
        return name, shape, params, state

    def forward(params, state, x, **kwargs):
        z = x.reshape(shape)
        return 0.0, z, state

    def inverse(params, state, z, **kwargs):
        x = z.reshape(original_shape)
        return 0.0, x, state

    return init_fun, forward, inverse

################################################################################################################

def Reverse(name='unnamed'):
    # language=rst
    """
    Reverse an input.
    """
    def init_fun(key, input_shape):
        params, state = (), ()
        return name, input_shape, params, state

    def forward(params, state, x, **kwargs):
        return 0.0, x[...,::-1], state

    def inverse(params, state, z, **kwargs):
        return 0.0, z[...,::-1], state

    return init_fun, forward, inverse

################################################################################################################

__all__ = ['Squeeze',
           'UnSqueeze',
           'Transpose',
           'Reshape',
           'Flatten',
           'Reverse']
