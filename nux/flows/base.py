from collections import OrderedDict, namedtuple
from functools import partial, wraps
import jax.nn.initializers as jaxinit
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
import nux.util as util
fast_tree_leaves = jit(jax.tree_util.tree_leaves)

Flow = namedtuple('Flow', ['name',
                           'input_shapes',
                           'output_shapes',
                           'input_ndims', # Include ndims because cannot get this easily for all shapes.
                           'output_ndims', # Include ndims because cannot get this easily for all shapes.
                           'params',
                           'state',
                           'apply'])

################################################################################################################

def auto_batch(layer):
    # language=rst
    """
    Automatically handle extra leading dimensions on an input.
    """
    @wraps(layer)
    def call_layer(*args, **kwargs):

        name = kwargs.get('name')

        _init_fun = layer(*args, **kwargs)

        def init_fun(key, inputs, batched=False, batch_depth=1, **kwargs):
            # Initialize the flow layer
            outputs, flow = _init_fun(key, inputs, batched=batched, batch_depth=batch_depth, **kwargs)

            # Keep track of the expected dimensions
            expected_input_x_dim = fast_tree_leaves(flow.input_ndims['x'])[0]
            expected_output_x_dim = fast_tree_leaves(flow.output_ndims['x'])[0]

            # The new apply fun will vmap when needed
            def apply_fun(params, state, inputs, key=None, reverse=False, **kwargs):
                input_dim = fast_tree_leaves(inputs['x'])[0].ndim # Assume all inputs are batched the same!!
                expected_dim = expected_input_x_dim if reverse == False else expected_output_x_dim

                # Recursively vmap
                if(input_dim > expected_dim):

                    # Need to split keys when we vmap!
                    if(key is not None):
                        N = fast_tree_leaves(inputs['x'])[0].shape[0]
                        keys = random.split(key, N)
                        outputs, updated_state = vmap(partial(apply_fun, params, state, reverse=reverse, **kwargs))(inputs, keys)
                    else:
                        outputs, updated_state = vmap(partial(apply_fun, params, state, reverse=reverse, **kwargs))(inputs)

                    # Average the state.  Not sure if this is the best way to do this.
                    updated_state = jax.tree_util.tree_map(lambda x: x.mean(axis=0), updated_state)
                    return outputs, updated_state

                return flow.apply(params, state, inputs, key=key, reverse=reverse, **kwargs)

            new_flow = Flow(flow.name, flow.input_shapes, flow.output_shapes, flow.input_ndims, flow.output_ndims, flow.params, flow.state, apply_fun)
            return outputs, new_flow

        return init_fun

    return call_layer

################################################################################################################

def initialize(name, apply_fun, create_params_and_state, data_dependent=False, start_hook=None, end_hook=None):
    # language=rst
    """
    Data dependent init function that does not do any special initialization using data.
    """
    def init_fun(key, inputs, batched=False, batch_depth=1, **kwargs):
        k1, k2 = random.split(key, 2)
        if(batched == False):
            for i in range(batch_depth):
                inputs = jax.tree_util.tree_map(lambda x: x[None], inputs)

        # Retrieve the shapes of the inputs
        unbatched_inputs = inputs
        for i in range(batch_depth):
            unbatched_inputs = jax.tree_util.tree_map(lambda x: x[0], unbatched_inputs)
        input_shapes = util.tree_shapes(unbatched_inputs)
        input_ndims = util.tree_ndims(unbatched_inputs)

        # Initialize the parameters and state
        if(data_dependent):
            params, state = create_params_and_state(k1, inputs, batch_depth)
        else:
            params, state = create_params_and_state(k1, input_shapes)

        # If we need things from the initialization function, we can pass a hook
        if(start_hook is not None):
            start_hook(input_shapes, input_ndims)

        # Pass the inputs to forward
        vmapped_fun = partial(apply_fun, params, state, key=k2, **kwargs)
        for i in range(batch_depth):
            vmapped_fun = vmap(vmapped_fun)
        outputs, _ = vmapped_fun(inputs)

        # Retrieve the output shapes
        unbatched_outputs = outputs
        for i in range(batch_depth):
            unbatched_outputs = jax.tree_util.tree_map(lambda x: x[0], unbatched_outputs)
        output_shapes = util.tree_shapes(unbatched_outputs)
        output_ndims = util.tree_ndims(unbatched_outputs)

        # Get the flow instance
        flow = Flow(name, input_shapes, output_shapes, input_ndims, output_ndims, params, state, apply_fun)

        # If we need things from the initialization function, we can pass a hook
        if(end_hook is not None):
            end_hook(flow)

        if(batched == False):
            outputs = unbatched_outputs

        return outputs, flow

    return init_fun

################################################################################################################

def Debug(message='', name='debug'):
    # language=rst
    """
    Debug by looking at input shapes
    """
    n_dims = None

    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        x = inputs['x']
        dims = x.ndim

        inputs_shapes = util.tree_shapes(inputs)
        print(message, 'inputs_shapes', inputs_shapes)

        if(dims > n_dims):
            log_det = jnp.zeros(x.shape[:dims - n_dims])
        else:
            log_det = 0.0

        outputs = {'x': x, 'log_det': log_det}

        return outputs, state

    def create_params_and_state(key, input_shapes):
        params, state = {}, {}
        return params, state

    def hook(input_shapes, input_ndims):
        nonlocal n_dims
        n_dims = input_ndims['x']

    return initialize(name, apply_fun, create_params_and_state, start_hook=hook)
