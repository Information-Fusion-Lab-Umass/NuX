from collections import OrderedDict, namedtuple
from functools import partial, wraps
import jax.nn.initializers as jaxinit
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
import src.util as util

Flow = namedtuple('Flow', ['name', 'input_shapes', 'output_shapes', 'params', 'state', 'forward', 'inverse'])

def auto_batch(layer):

    @wraps(layer)
    def call_layer(*args, **kwargs):

        _init_fun, data_dependent_init_fun = layer(*args, **kwargs)

        def init_fun(key, input_shapes):
            # Initialize the flow layer
            flow = _init_fun(key, input_shapes)

            # Keep track of the expected dimensions
            expected_input_x_dim  = len(input_shapes['x'])
            expected_output_x_dim = len(flow.output_shapes['x'])

            def forward(params, state, inputs, **kwargs):
                if(inputs['x'].ndim > expected_input_x_dim):
                    return vmap(partial(forward, params, state, **kwargs))(inputs)
                return flow.forward(params, state, inputs, **kwargs)

            def inverse(params, state, inputs, **kwargs):
                if(inputs['x'].ndim > expected_output_x_dim):
                    return vmap(partial(inverse, params, state, **kwargs))(inputs)
                return flow.inverse(params, state, inputs, **kwargs)

            return Flow(flow.name, flow.input_shapes, flow.output_shapes, flow.params, flow.state, forward, inverse)

        return init_fun, data_dependent_init_fun

    return call_layer

def data_independent_init(init_fun):

    def ddi(key, inputs, **kwargs):
        input_shapes = util.tree_shapes(inputs)
        flow = init_fun(key, input_shapes)
        outputs, _ = flow.forward(flow.params, flow.state, inputs, **kwargs)
        return outputs, flow

    return ddi