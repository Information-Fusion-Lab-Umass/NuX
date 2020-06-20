import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from functools import partial
from jax.ops import index, index_add, index_update
import src.util as util
import src.flows.base as base
from collections import OrderedDict

################################################################################################################

def sequential(*init_funs, name='sequential'):
    n_layers = len(init_funs)

    def init_fun(key, original_inputs, batched=False, **kwargs):
        # Use a new dictionary so that we don't modify the existing one
        inputs = {}
        inputs.update(original_inputs)

        keys = random.split(key, n_layers)

        # Retrieve the shape of the inputs
        if(batched == True):
            unbatched_inputs = jax.tree_util.tree_map(lambda x: x[0], inputs)
            actual_input_shape = util.tree_shapes(unbatched_inputs)
        else:
            actual_input_shape = util.tree_shapes(inputs)

        # Initialize each function
        params, state = OrderedDict(), OrderedDict()
        apply_funs = []
        log_det = 0.0
        used_names = {}
        for key, init_fun in zip(keys, init_funs):

            # Initialize the flow and handle passing the inputs to the next flow accordingly
            outputs, flow = init_fun(key, inputs, batched=batched, **kwargs)
            log_det += outputs['log_det']
            inputs.update(outputs)

            # Can't repeat names!
            if(flow.name in used_names):
                index = used_names[flow.name]
                used_names[flow.name] += 1
                flow = flow._replace(name='%s_%d'%(flow.name, index))
            else:
                used_names[flow.name] = 0

            # Update everything
            apply_funs.append(flow.apply)
            params[flow.name] = flow.params
            state[flow.name] = flow.state

        # Finalize the things we need in the flow
        output_shapes = flow.output_shapes
        outputs['log_det'] = log_det

        def apply_fun(params, state, original_inputs, reverse=False, **kwargs):
            # Use a new dictionary so that we don't modify the existing one
            inputs = {}
            inputs.update(original_inputs)

            funs = apply_funs
            names = list(params.keys())
            if(reverse):
                funs = funs[::-1]
                names = names[::-1]

            key = kwargs.pop('key', None)
            keys = random.split(key, len(funs)) if key is not None else (None,)*len(funs)
            log_det = 0.0
            updated_state = OrderedDict()

            for fun, name, key in zip(funs, names, keys):
                # Run the function
                outputs, uptd_state = fun(params[name], state[name], inputs, key=key, reverse=reverse, **kwargs)
                # Update the log determinant and state
                log_det += outputs['log_det']
                updated_state[name] = uptd_state

                # Update the input for the next iteration
                inputs.update(outputs)

            inputs['log_det'] = log_det
            return inputs, updated_state

        flow = base.Flow(name, actual_input_shape, output_shapes, params, state, apply_fun)
        return outputs, flow

    return init_fun

################################################################################################################

def factored(*init_funs, name='factored'):
    n_factors = len(init_funs)

    def init_fun(key, inputs, batched=False, **kwargs):
        keys = random.split(key, n_factors)

        # Retrieve the shape of the inputs
        if(batched == True):
            unbatched_inputs = jax.tree_util.tree_map(lambda x: x[0], inputs)
            actual_input_shape = util.tree_shapes(unbatched_inputs)
        else:
            actual_input_shape = util.tree_shapes(inputs)

        # Initialize each function
        params, state = OrderedDict(), OrderedDict()
        apply_funs = []
        log_det = 0.0
        xs = []
        used_names = {}
        for key, init_fun, x in zip(keys, init_funs, inputs['x']):

            # Create a new input dictionary
            single_input = inputs.copy()
            single_input['x'] = x

            # Initialize the flow
            outputs, flow = init_fun(key, single_input, batched=batched, **kwargs)
            log_det += outputs['log_det']
            xs.append(outputs['x'])

            # Can't repeat names!
            if(flow.name in used_names):
                index = used_names[flow.name]
                used_names[flow.name] += 1
                flow = flow._replace(name='%s_%d'%(flow.name, index))
            else:
                used_names[flow.name] = 0

            # Update everything
            apply_funs.append(flow.apply)
            params[flow.name] = flow.params
            state[flow.name] = flow.state

        # Finalize the things we need in the flow
        outputs = inputs.copy()
        outputs['x'] = xs
        outputs['log_det'] = log_det
        output_shapes = util.tree_shapes(outputs)

        def apply_fun(params, state, original_inputs, reverse=False, **kwargs):
            # Use a new dictionary so that we don't modify the existing one
            inputs = {}
            inputs.update(original_inputs)

            input_xs = inputs['x']
            funs = apply_funs
            names = list(params.keys())
            if(reverse):
                funs = funs[::-1]
                names = names[::-1]
                input_xs = input_xs[::-1]

            key = kwargs.pop('key', None)
            keys = random.split(key, len(funs)) if key is not None else (None,)*len(funs)
            log_det = 0.0
            xs = []
            updated_state = OrderedDict()

            for fun, name, key, x in zip(funs, names, keys, input_xs):

                # Create a new input dictionary
                single_input = inputs.copy()
                single_input['x'] = x

                # Run the function
                outputs, uptd_state = fun(params[name], state[name], single_input, key=key, reverse=reverse, **kwargs)

                # Update the log determinant and state
                log_det += outputs['log_det']
                xs.append(outputs['x'])
                updated_state[name] = uptd_state

            if(reverse):
                xs = xs[::-1]

            outputs = inputs.copy()
            outputs['x'] = xs
            outputs['log_det'] = log_det
            return outputs, updated_state

        flow = base.Flow(name, actual_input_shape, output_shapes, params, state, apply_fun)
        return outputs, flow

    return init_fun

################################################################################################################

def ChainRule(split_idx, axis=-1, factor=True, name='chain_rule'):
    # language=rst
    """
    Split/recombine a vector.  Use this to set up chain rule
    """
    n_dims = None

    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        x = inputs['x']
        if(reverse != factor):
            z = jnp.split(x, split_idx, axis)
            dims = x.ndim
            batch_size = x.shape[0]
        else:
            z = jnp.concatenate(x, axis)
            dims = z.ndim
            batch_size = z.shape[0]

        if(dims > n_dims):
            log_det = jnp.zeros(batch_size)
        else:
            log_det = 0.0

        outputs = {'x': z, 'log_det': log_det}
        return outputs, state

    def create_params_and_state(key, input_shapes):
        params, state = {}, {}
        return params, state

    def init_fun(key, inputs, batched=False, **kwargs):

        if(batched == False):
            inputs = jax.tree_util.tree_map(lambda x: x[None], inputs)

        nonlocal n_dims
        n_dims = jax.tree_util.tree_leaves(inputs['x'])[0].ndim - 1

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
        flow = base.Flow(name, input_shapes, output_shapes, params, state, apply_fun)

        if(batched == False):
            outputs = jax.tree_util.tree_map(lambda x: x[0], outputs)

        return outputs, flow

    return init_fun

################################################################################################################

from src.flows.reshape import Squeeze, UnSqueeze
from src.flows.basic import Identity

def multi_scale(flow, existing_flow):
    return sequential(flow,
                      Squeeze(),
                      ChainRule(2, factor=True),
                      factored(existing_flow, Identity()),
                      ChainRule(2, factor=False),
                      UnSqueeze())

        # def multi_scale(i, flow):
        #     if(isinstance(self.n_filters, int)):
        #         n_filters = self.n_filters
        #     else:
        #         n_filters = self.n_filters[i]

        #     if(isinstance(self.n_blocks, int)):
        #         n_blocks  = self.n_blocks
        #     else:
        #         n_blocks  = self.n_blocks[i]

        #     return nf.sequential_flow(nf.Squeeze(),
        #                               GLOWComponent(name_iter, n_filters, n_blocks),
        #                               nf.FactorOut(2),
        #                               nf.factored_flow(flow, nf.Identity()),
        #                               nf.FanInConcat(2),
        #                               nf.UnSqueeze())

################################################################################################################

__all__ = ['sequential',
           'factored',
           'ChainRule',
           # 'Augment',
           'multi_scale']

# @base.auto_batch
# @base.ensure_dictionaries
# def Augment(flow, sampler, name='augment'):
#     # language=rst
#     """
#     Run a normalizing flow in an augmented space https://arxiv.org/pdf/2002.07101.pdf

#     :param flow: The normalizing flow
#     :param sampler: Function to sample from the convolving distribution
#     """
#     _init_fun, _data_dependent_init = flow
#     # _init_fun, _forward, _inverse = flow

#     def forward(params, state, inputs, **kwargs):
#         x = inputs['x']
#         key = kwargs.pop('key', None)
#         if(key is None):
#             assert 0, 'Need a key for this'
#         k1, k2 = random.split(key, 2)

#         # Sample e and concatenate it to x
#         e = random.normal(k1, x.shape)
#         xe = jnp.concatenate([x, e], axis=-1)

#         return _forward(params, state, xe, key=k2, **kwargs)

#     def inverse(params, state, inputs, **kwargs):
#         z = inputs['x']
#         key = kwargs.pop('key', None)
#         if(key is None):
#             assert 0, 'Need a key for this'
#         k1, k2 = random.split(key, 2)

#         x, e = jnp.split(z, axis=-1)

#         return _inverse(params, state, x, key=k2, **kwargs)

#     def init_fun(key, input_shapes):
#         x_shape = input_shapes['x']
#         augmented_input_shape = x_shape[:-1] + (2*x_shape[-1],)

#         return _init_fun(key, {'x': augmented_input_shape})


#     return init_fun, base.data_independent_init(init_fun)
