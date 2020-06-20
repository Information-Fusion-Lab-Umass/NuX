import jax.numpy as jnp
import jax
from jax import random, jit, vmap
import src.util as util
import src.flows.base as base

################################################################################################################

def Identity(name='Identity'):
    # language=rst
    """
    Identity transformation
    """
    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        outputs = {'x': inputs['x'], 'log_det': 0.0}
        return outputs, state

    def create_params_and_state(key, input_shapes):
        params = {}
        state = {}
        return params, state

    return base.data_independent_init(name, apply_fun, create_params_and_state)

__all__ = ['Identity']
