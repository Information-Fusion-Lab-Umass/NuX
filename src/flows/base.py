from collections import OrderedDict, namedtuple
from functools import partial, wraps
import jax.nn.initializers as jaxinit
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
import src.util as util

Flow = namedtuple('Flow', ['name', 'input_shapes', 'output_shapes', 'params', 'state', 'apply'])

################################################################################################################

def auto_batch(layer):

    @wraps(layer)
    def call_layer(*args, **kwargs):

        name = kwargs.get('name')

        _init_fun = layer(*args, **kwargs)

        def init_fun(key, inputs, batched=False, **kwargs):
            # Initialize the flow layer
            outputs, flow = _init_fun(key, inputs, batched=batched, **kwargs)

            # Keep track of the expected dimensions
            expected_input_x_dim  = len(flow.input_shapes['x'])
            expected_output_x_dim = len(flow.output_shapes['x'])

            def apply_fun(params, state, inputs, reverse=False, **kwargs):
                input_dim = jax.tree_util.tree_leaves(inputs['x'])[0].ndim
                expected_dim = expected_input_x_dim if reverse == False else expected_output_x_dim

                if(input_dim > expected_dim):
                    outputs, updated_state = vmap(partial(apply_fun, params, state, reverse=reverse, **kwargs))(inputs)
                    # Because we're using auto batch, we should expect that the state update doesn't depend on the batch
                    updated_state = jax.tree_util.tree_map(lambda x: x[0], updated_state)
                    return outputs, updated_state

                return flow.apply(params, state, inputs, reverse=reverse, **kwargs)

            new_flow = Flow(flow.name, flow.input_shapes, flow.output_shapes, flow.params, flow.state, apply_fun)
            return outputs, new_flow

        return init_fun

    return call_layer

################################################################################################################

def data_independent_init(name, apply_fun, create_params_and_state):
    # language=rst
    """
    Data dependent init function that does not do any special initialization using data.
    """
    def init_fun(key, inputs, batched=False, **kwargs):
        if(batched == False):
            inputs = jax.tree_util.tree_map(lambda x: x[None], inputs)

        # Retrieve the shapes of the inputs
        unbatched_inputs = jax.tree_util.tree_map(lambda x: x[0], inputs)
        input_shapes = util.tree_shapes(unbatched_inputs)

        # Initialize the parameters and state
        params, state = create_params_and_state(key, input_shapes)

        # Pass the dummy inputs to forward
        outputs, _ = vmap(partial(apply_fun, params, state, **kwargs))(inputs)

        # Retrieve the output shapes
        unbatched_outputs = jax.tree_util.tree_map(lambda x: x[0], outputs)
        output_shapes = util.tree_shapes(unbatched_outputs)

        # Get the flow instance
        flow = Flow(name, input_shapes, output_shapes, params, state, apply_fun)

        if(batched == False):
            outputs = jax.tree_util.tree_map(lambda x: x[0], outputs)

        return outputs, flow

    return init_fun

################################################################################################################

def Debug(message='', name='debug'):
    # language=rst
    """
    Debug by looking at shapes
    """
    n_dims = None

    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        x = inputs['x']
        dims = x.ndim

        if(dims > n_dims):
            log_det = jnp.zeros(x.shape[0])
        else:
            log_det = 0.0

        outputs = {'x': x, 'log_det': log_det}
        inputs_shapes = util.tree_shapes(inputs)
        print(message, 'inputs_shapes', inputs_shapes)
        return outputs, state

    def create_params_and_state(key, input_shapes):
        nonlocal n_dims
        n_dims = len(input_shapes['x'])

        params = {}
        state = {}
        return params, state

    return data_independent_init(name, apply_fun, create_params_and_state)

################################################################################################################

def ensure_dictionaries(layer):

    @wraps(layer)
    def call_layer(*args, **kwargs):

        _init_fun, data_dependent_init_fun = layer(*args, **kwargs)
        original_input_shape = None

        def init_fun(key, input_shapes):
            assert 'x' in input_shapes
            assert len(input_shapes.keys()) == 1

            # Initialize the flow layer
            flow = _init_fun(key, input_shapes)

            def forward(params, state, inputs, **kwargs):
                assert 'x' in inputs
                assert len(input_shapes.keys()) == 1

                return flow.forward(params, state, inputs, **kwargs)

            def inverse(params, state, inputs, **kwargs):
                assert 'x' in inputs
                assert len(input_shapes.keys()) == 1

                return flow.inverse(params, state, inputs, **kwargs)

            return Flow(flow.name, flow.input_shapes, flow.output_shapes, flow.params, flow.state, forward, inverse)

        return init_fun, data_dependent_init_fun

    return call_layer
