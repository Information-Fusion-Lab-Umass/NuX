import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from functools import partial
from jax.ops import index, index_add, index_update
import src.util as util

################################################################################################################

def sequential(*layers):
    # language=rst
    """
    Sequential flow builder.  Like spp.sequential, but also passes density and works in reverse.
    forward transforms data, x, into a latent variable, z.
    inverse transforms a latent variable, z, into data, x.
    We can also pass a condition in order to compute logp(x|condition)

    :param layers - An unpacked list of (init_fun, apply_fun)

    **Example**

    .. code-block:: python

        from jax import random
        from normalizing_flows import sequential, MAF, BatchNorm, UnitGaussianPrior
        from util import TRAIN, TEST
        key = random.PRNGKey(0)

        # Create the flow
        input_shape = (5,)
        flow = sequential(MAF([1024]), Reverse(), BatchNorm(), MAF([1024]), UnitGaussianPrior())

        # Initialize it
        init_fun, forward, inverse = flow
        names, output_shape, params, state = init_fun(key, input_shape)

        # Run an input through the flow
        inputs = jnp.ones((10, 5))
        log_det1, z, updated_state = forward(params, state, inputs)
        log_det2, fz, _ = inverse(params, state, z)

        assert jnp.allclose(fz, x)
        assert jnp.allclose(log_det1, log_det2)
    """
    n_layers = len(layers)
    init_funs, forward_funs, inverse_funs = zip(*layers)

    def init_fun(key, input_shape):
        names, params, states = [], [], []
        keys = random.split(key, len(init_funs))
        for key, init_fun in zip(keys, init_funs):
            # Conditioning can only be added in a factor call or at the top level call
            name, input_shape, param, state = init_fun(key, input_shape)
            names.append(name)
            params.append(param)
            states.append(state)
        return tuple(names), input_shape, tuple(params), tuple(states)

    def evaluate(apply_funs, params, state, inputs, **kwargs):

        # Need to store the ouputs of the functions and the updated state
        updated_states = []

        # Need to pop so that we don't resuse random keys!
        key = kwargs.pop('key', None)
        keys = random.split(key, n_layers) if key is not None else (None,)*n_layers

        # Evaluate each function and store the updated static parameters
        log_det = 0.0
        for fun, param, s, key in zip(apply_funs, params, state, keys):
            _log_det, inputs, updated_state = fun(param, s, inputs, key=key, **kwargs)
            updated_states.append(updated_state)
            log_det += _log_det

        return log_det, inputs, tuple(updated_states)

    def forward(params, state, x, **kwargs):
        return evaluate(forward_funs, params, state, x, **kwargs)

    def inverse(params, state, z, **kwargs):
        return evaluate(inverse_funs[::-1], params[::-1], state[::-1], z, **kwargs)

    return init_fun, forward, inverse

def factored(*layers, condition_on_results=False):
    # language=rst
    """
    Parallel flow builder.  Like spp.parallel, but also passes density and works in reverse.
    forward transforms data, x, into a latent variable, z.
    inverse transforms a latent variable, z, into data, x.
    This function exploits the chain rule p(x) = p([x_1,x_2,...x_N]) = p(x_1)p(x_2|x_1)*...*p(x_N|x_N-1,...,x_1)
    The result of each distribution is passed as a new conditioner to the next distribution.

    :param layers - An unpacked list of (init_fun, apply_fun)

    **Example**

    .. code-block:: python

        from jax import random
        from normalizing_flows import sequential, MAF, FactorOut, FanInConcat, UnitGaussianPrior
        from util import TRAIN, TEST
        key = random.PRNGKey(0)

        # Create the flow
        input_shape = (6,)
        flow = sequential(Factor(2),
                               factored(MAF([1024])
                                        MAF([1024])),
                               FanInConcat(2),
                               UnitGaussianPrior())

        # Initialize it
        init_fun, forward, inverse = flow
        names, output_shape, params, state = init_fun(key, input_shape)

        # Run an input through the flow
        inputs = jnp.ones((10, 5))
        log_det1, z, updated_state = forward(params, state inputs)
        log_det2, fz, _ = inverse(params, state, z)

        assert jnp.allclose(fz, x)
        assert jnp.allclose(log_det1, log_det2)
    """
    n_layers = len(layers)
    init_funs, forward_funs, inverse_funs = zip(*layers)

    # Feature extract network
    fe_apply_fun = None

    def init_fun(key, input_shape):
        keys = random.split(key, n_layers + 1)

        # Find the shapes of all of the conditionals
        names, output_shapes, params, states = [], [], [], []

        # Split these up so that we can evaluate each of the parallel items together
        condition_shape = ()
        for init_fun, key, shape in zip(init_funs, keys, input_shape):
            conditioned_shape = shape + condition_shape
            name, output_shape, param, state = init_fun(key, conditioned_shape)
            names.append(name)
            output_shapes.append(output_shape)
            params.append(param)
            states.append(state)

            if(condition_on_results):
                condition_shape = condition_shape + (output_shape,)

        return tuple(names), output_shapes, tuple(params), tuple(states)

    def forward(params, state, x, **kwargs):

        # Need to pop so that we don't resuse random keys!
        key = kwargs.pop('key', None)
        n_keys = n_layers if fe_apply_fun is None else n_layers*2
        keys = random.split(key, n_keys) if key is not None else (None,)*n_keys
        key_iter = iter(keys)

        # We need to store each of the outputs and state
        log_det = 0.0
        outputs, states = [], []
        for apply_fun, param, s, inp in zip(forward_funs, params, state, x):
            _log_det, output, s = apply_fun(param, s, inp, key=next(key_iter), **kwargs)
            log_det += _log_det
            outputs.append(output)
            states.append(s)

            if(condition_on_results):
                condition = condition + (output,)

        return log_det, outputs, tuple(states)

    def inverse(params, state, z, **kwargs):

        # Need to pop so that we don't resuse random keys!
        key = kwargs.pop('key', None)
        n_keys = n_layers if fe_apply_fun is None else n_layers*2
        keys = random.split(key, n_keys) if key is not None else (None,)*n_keys
        key_iter = iter(keys)

        # We need to store each of the outputs and state
        log_det = 0.0
        outputs, states = [], []
        for apply_fun, param, s, inp in zip(inverse_funs, params, state, z):
            _log_det, output, updated_state = apply_fun(param, s, inp, key=next(key_iter), **kwargs)
            log_det += _log_det
            outputs.append(output)
            states.append(updated_state)

            # Conditioners are inputs during the inverse pass
            if(condition_on_results):
                condition = condition + (inp,)

        return log_det, outputs, tuple(states)

    return init_fun, forward, inverse

################################################################################################################

def UnitGaussianPrior(axis=(-1,), name='unnamed'):
    # language=rst
    """
    Prior for the normalizing flow.

    :param axis - Axes to reduce over
    """
    def init_fun(key, input_shape):
        params, state = (), ()
        return name, input_shape, params, state

    def forward(params, state, x, **kwargs):
        dim = jnp.prod([x.shape[ax] for ax in axis])
        log_det = -0.5*jnp.sum(x**2, axis=axis) + -0.5*dim*jnp.log(2*jnp.pi)
        return log_det, x, state

    def inverse(params, state, z, **kwargs):
        # Usually we're sampling z from a Gaussian, so if we want to do Monte Carlo
        # estimation, ignore the value of N(z|0,I).
        return 0.0, z, state

    return init_fun, forward, inverse

################################################################################################################

def Identity(name='unnamed'):
    # language=rst
    """
    Just pass an input forward.
    """
    def init_fun(key, input_shape):
        params, state = (), ()
        return name, input_shape, params, state

    def forward(params, state, x, **kwargs):
        return 0.0, x, state

    def inverse(params, state, z, **kwargs):
        return 0.0, z, state

    return init_fun, forward, inverse

################################################################################################################

def ReverseInputs(name='unnamed'):
    # language=rst
    """
    Reverse the order of inputs.  Not the same as reversing an array!
    """
    def init_fun(key, input_shape):
        params, state = (), ()
        return name, input_shape[::-1], params, state

    def forward(params, state, x, **kwargs):
        return 0.0, x[::-1], state[::-1]

    def inverse(params, state, z, **kwargs):
        return 0.0, z[::-1], state[::-1]

    return init_fun, forward, inverse

################################################################################################################

def Split(split_idx, axis=-1, name='unnamed'):
    # language=rst
    """
    Split a vector
    """
    def init_fun(key, input_shape):
        assert len(split_idx) == 1 # For the moment
        ax = axis % len(input_shape)
        out_shape1, out_shape2 = list(input_shape), list(input_shape)
        out_shape1[ax] = split_idx[0]
        out_shape2[ax] = input_shape[ax] - split_idx[0]
        params, state = (), ()
        return name, (tuple(out_shape1), tuple(out_shape2)), params, state

    def forward(params, state, x, **kwargs):
        z_components = jnp.split(x, split_idx, axis)
        zs = z_components

        return 0.0, zs, state

    def inverse(params, state, z, **kwargs):
        x = jnp.concatenate(z, axis)
        return 0.0, x, state

    return init_fun, forward, inverse

def Concat(axis=-1, name='unnamed'):
    """
    Going to unify this and FanInConcat later
    """
    split_idx = None

    def init_fun(key, input_shape):
        assert len(input_shape) == 2
        ax = axis % len(input_shape[0])
        out_shape = list(input_shape[0])

        nonlocal split_idx
        split_idx = [input_shape[0][ax]]

        out_shape[ax] = input_shape[0][ax] + input_shape[1][ax]
        params, state = (), ()
        return name, tuple(out_shape), params, state

    def forward(params, state, x, **kwargs):
        z = jnp.concatenate(x, axis)
        return 0.0, z, state

    def inverse(params, state, z, **kwargs):
        xs = jnp.split(z, split_idx, axis)
        return 0.0, xs, state

    return init_fun, forward, inverse

################################################################################################################

def FactorOut(num, axis=-1, name='unnamed'):
    # language=rst
    """
    Factor p(z_{1..N}) = p(z_1)p(z_2|z_1)...p(z_N|z_{1..N-1}) using an even split

    :param num: Number of components to split into
    :param axis: Axis to split
    """
    def init_fun(key, input_shape):
        ax = axis % len(input_shape)

        # Split evenly
        assert input_shape[ax]%num == 0
        split_shape = list(input_shape)
        split_shape[ax] = input_shape[ax]//num
        split_shape = tuple(split_shape)

        params, state = (), ()
        return name, [split_shape]*num, params, state

    def forward(params, state, x, **kwargs):
        zs = jnp.split(x, num, axis)
        return 0.0, zs, state

    def inverse(params, state, z, **kwargs):
        x = jnp.concatenate(z, axis)
        return 0.0, x, state

    return init_fun, forward, inverse

def FanInConcat(num, axis=-1, name='unnamed'):
    # language=rst
    """
    Inverse of FactorOut

    :param num: Number of components to split into
    :param axis: Axis to split
    """
    def init_fun(key, input_shape):
        # Make sure that each of the inputs are the same size
        assert num == len(input_shape)
        for shape in input_shape:
            assert shape == input_shape[0], input_shape
        ax = axis % len(input_shape[0])
        concat_size = sum(shape[ax] for shape in input_shape)
        out_shape = input_shape[0][:ax] + (concat_size,) + input_shape[0][ax+1:]
        params, state = (), ()
        return name, out_shape, params, state

    def forward(params, state, x, **kwargs):
        zs = jnp.concatenate(x, axis)
        return 0.0, zs, state

    def inverse(params, state, z, **kwargs):
        x = jnp.split(z, num, axis)
        return 0.0, x, state

    return init_fun, forward, inverse

################################################################################################################

from src.flows.reshape import Squeeze, UnSqueeze

def multi_scale(flow, existing_flow):
    return sequential(Squeeze(),
                      flow,
                      FactorOut(2),
                      factored(existing_flow, Identity()),
                      FanInConcat(2),
                      UnSqueeze())

################################################################################################################

__all__ = ['sequential',
           'factored',
           'UnitGaussianPrior',
           'Identity',
           'ReverseInputs',
           'Split',
           'Concat',
           'FactorOut',
           'FanInConcat',
           'multi_scale']
