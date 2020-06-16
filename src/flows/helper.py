import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from functools import partial

################################################################################################################

def key_wrap(flow, key):
    # language=rst
    """
    Add the ability to specify a key to initialize
    """
    _init_fun, forward, inverse = flow

    def init_fun(unused_key, input_shape):
        name, output_shape, params, state = _init_fun(key, input_shape)
        return name, output_shape, params, state

    return init_fun, forward, inverse

def named_wrap(flow, name='unnamed'):
    _init_fun, _forward, _inverse = flow

    def init_fun(key, input_shape):
        _name, output_shape, params, state = _init_fun(key, input_shape)
        return name, output_shape, params, state

    def forward(params, state, x, **kwargs):
        log_det, z, updated_state = _forward(params, state, x, **kwargs)
        return log_det, z, updated_state

    def inverse(params, state, z, **kwargs):
        log_det, x, updated_state = _inverse(params, state, z, **kwargs)
        return log_det, x, updated_state

    return init_fun, forward, inverse

################################################################################################################

def Debug(message,
          print_init_shape=True,
          print_forward_shape=False,
          print_inverse_shape=False,
          compare_vals=False,
          name='unnamed'):
    # language=rst
    """
    Help debug shapes

    :param print_init_shape: Print the shapes
    :param print_forward_shape: Print the shapes
    :param print_inverse_shape: Print the shapes
    :param compare_vals: Print the difference between the value of the forward pass and the reconstructed
    """

    saved_val = None

    def init_fun(key, input_shape):
        if(print_init_shape):
            print(message, 'input_shape', input_shape)
        return name, input_shape, (), ()

    def forward(params, state, x, **kwargs):
        if(print_forward_shape):
            print(message, 'x shapes', [_x.shape for _x in x])

        if(compare_vals):
            nonlocal saved_val
            saved_val = x

        return 0.0, x, state

    def inverse(params, state, z, **kwargs):
        if(print_inverse_shape):
            print(message, 'z shapes', [_z.shape for _z in z])

        if(compare_vals):
            if(isinstance(z, tuple) or isinstance(z, list)):
                print(message, 'jnp.linalg.norm(z - saved_val)', [jnp.linalg.norm(_z - _x) for _x, _z in zip(saved_val, z)])
            else:
                print(message, 'jnp.linalg.norm(z - saved_val)', jnp.linalg.norm(z - saved_val))

        return 0.0, z, state

    return init_fun, forward, inverse

################################################################################################################

__all__ = ['key_wrap',
           'named_wrap',
           'Debug']