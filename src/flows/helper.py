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
    _init_fun, _data_dependent_init_fun = flow

    def init_fun(unused_key, input_shapes):
        return _init_fun(key, input_shapes)

    def data_dependent_init_fun(unused_key, inputs, **kwargs):
        return _data_dependent_init_fun(key, inputs, **kwargs)

    return init_fun, data_dependent_init_fun

################################################################################################################

def Debug(message='',
          print_init_shape=True,
          print_forward_shape=False,
          print_inverse_shape=False,
          compare_vals=False,
          name='debug'):
    # language=rst
    """
    Help debug shapes

    :param print_init_shape: Print the shapes
    :param print_forward_shape: Print the shapes
    :param print_inverse_shape: Print the shapes
    :param compare_vals: Print the difference between the value of the forward pass and the reconstructed
    """
    saved_val = None

    def forward(params, state, inputs, **kwargs):
        x = inputs['x']
        if(print_forward_shape):
            print(message, 'x shapes', [_x.shape for _x in x])

        if(compare_vals):
            nonlocal saved_val
            saved_val = x

        outputs = {'x': x, 'log_det': log_det}
        return outputs, state

    def inverse(params, state, inputs, **kwargs):
        z = inputs['x']
        if(print_inverse_shape):
            print(message, 'z shapes', [_z.shape for _z in z])

        if(compare_vals):
            if(isinstance(z, tuple) or isinstance(z, list)):
                print(message, 'jnp.linalg.norm(z - saved_val)', [jnp.linalg.norm(_z - _x) for _x, _z in zip(saved_val, z)])
            else:
                print(message, 'jnp.linalg.norm(z - saved_val)', jnp.linalg.norm(z - saved_val))

        outputs = {'x': z, 'log_det': log_det}
        return outputs, state

    def init_fun(key, input_shapes):
        if(print_init_shape):
            print(message, 'input_shapes', input_shapes)

        output_shapes = {}
        output_shapes.update(input_shapes)
        output_shapes['log_det'] = (1,)
        params, state = {}, {}
        return base.Flow(name, input_shapes, output_shapes, params, state, forward, inverse)

    return init_fun, base.data_independent_init(init_fun)

################################################################################################################

__all__ = ['key_wrap',
           'Debug']