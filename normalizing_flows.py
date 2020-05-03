import numpy as onp
import jax
from jax import random, jit, vmap, jacobian, grad, value_and_grad, pmap
import jax.nn
import jax.numpy as np
from functools import partial, reduce
from jax.experimental import stax
from jax.nn.initializers import glorot_normal, normal, ones, zeros
from jax.ops import index, index_add, index_update
import staxplusplus as spp
from jax.scipy.special import logsumexp
from util import is_testing, TRAIN, TEST, householder_prod, householder_prod_transpose
import util
from lds_svi import easy_niw_nat, easy_mniw_nat, mniw_params_to_stats, niw_nat_to_std, mniw_nat_to_std, lds_svi, mniw_sample, easy_niw_params, easy_mniw_params, niw_sample, niw_params_to_stats
from non_dim_preserving import *
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_multimap, tree_flatten
ravel_pytree = jit(ravel_pytree)
from tqdm.notebook import tnrange
from tqdm import tqdm

################################################################################################################

def sequential_flow(*layers):
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
        from normalizing_flows import sequential_flow, MAF, BatchNorm, UnitGaussianPrior
        from util import TRAIN, TEST
        key = random.PRNGKey(0)

        # Create the flow
        input_shape = (5,)
        condition_shape = ()
        flow = sequential_flow(MAF([1024]), Reverse(), BatchNorm(), MAF([1024]), UnitGaussianPrior())

        # Initialize it
        init_fun, forward, inverse = flow
        names, output_shape, params, state = init_fun(key, input_shape)

        # Run an input through the flow
        inputs = np.ones((10, 5))
        log_px = np.zeros(inputs.shape[0]) # Need to pass in a correctly shaped initial density
        condition = ()
        log_px, z, updated_state = forward(params, state, log_px, inputs, condition, test=TEST)
        log_pfz, fz, _ = inverse(params, state, np.zeros(inputs.shape[0]), z, condition, test=TEST)

        assert np.allclose(fz, x)
        assert np.allclose(log_pfz, log_px)
    """
    n_layers = len(layers)
    init_funs, forward_funs, inverse_funs = zip(*layers)

    def init_fun(key, input_shape, condition_shape):
        names, params, states = [], [], []
        for init_fun in init_funs:
            key, *keys = random.split(key, 2)
            # Conditioning can only be added in a factor call or at the top level call
            name, input_shape, param, state = init_fun(keys[0], input_shape, condition_shape)
            names.append(name)
            params.append(param)
            states.append(state)
        return tuple(names), input_shape, tuple(params), tuple(states)

    def evaluate(apply_funs, params, state, log_px, inputs, condition, **kwargs):

        # Need to store the ouputs of the functions and the updated state
        updated_states = []

        # Need to pop so that we don't resuse random keys!
        key = kwargs.pop('key', None)
        keys = random.split(key, n_layers) if key is not None else (None,)*n_layers

        # Evaluate each function and store the updated static parameters
        for fun, param, s, key in zip(apply_funs, params, state, keys):
            log_px, inputs, updated_state = fun(param, s, log_px, inputs, condition, key=key, **kwargs)
            updated_states.append(updated_state)

        return log_px, inputs, tuple(updated_states)

    def forward(params, state, log_px, x, condition, **kwargs):
        return evaluate(forward_funs, params, state, log_px, x, condition, **kwargs)

    def inverse(params, state, log_pz, z, condition, **kwargs):
        return evaluate(inverse_funs[::-1], params[::-1], state[::-1], log_pz, z, condition, **kwargs)

    return init_fun, forward, inverse

def factored_flow(*layers, condition_on_results=False):
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
        from normalizing_flows import sequential_flow, MAF, FactorOut, FanInConcat, UnitGaussianPrior
        from util import TRAIN, TEST
        key = random.PRNGKey(0)

        # Create the flow
        input_shape = (6,)
        condition_shape = ()
        flow = sequential_flow(Factor(2),
                               factored_flow(MAF([1024])
                                             MAF([1024])),
                               FanInConcat(2),
                               UnitGaussianPrior())

        # Initialize it
        init_fun, forward, inverse = flow
        names, output_shape, params, state = init_fun(key, input_shape)

        # Run an input through the flow
        inputs = np.ones((10, 5))
        log_px = np.zeros(inputs.shape[0]) # Need to pass in a correctly shaped initial density
        condition = ()
        log_px, z, updated_state = forward(params, state, log_px, inputs, condition, test=TEST)
        log_pfz, fz, _ = inverse(params, state, np.zeros(inputs.shape[0]), z, condition, test=TEST)

        assert np.allclose(fz, x)
        assert np.allclose(log_pfz, log_px)
    """
    n_layers = len(layers)
    init_funs, forward_funs, inverse_funs = zip(*layers)

    # Feature extract network
    fe_apply_fun = None

    def init_fun(key, input_shape, condition_shape):
        keys = random.split(key, n_layers + 1)

        # Find the shapes of all of the conditionals
        names, output_shapes, params, states = [], [], [], []

        # Split these up so that we can evaluate each of the parallel items together
        for init_fun, key, shape in zip(init_funs, keys, input_shape):
            name, output_shape, param, state = init_fun(key, shape, condition_shape)
            names.append(name)
            output_shapes.append(output_shape)
            params.append(param)
            states.append(state)

            if(condition_on_results):
                condition_shape = condition_shape + (output_shape,)

        return tuple(names), output_shapes, tuple(params), tuple(states)

    def forward(params, state, log_px, x, condition, **kwargs):

        # Need to pop so that we don't resuse random keys!
        key = kwargs.pop('key', None)
        n_keys = n_layers if fe_apply_fun is None else n_layers*2
        keys = random.split(key, n_keys) if key is not None else (None,)*n_keys
        key_iter = iter(keys)

        # We need to store each of the outputs and state
        densities, outputs, states = [], [], []
        for apply_fun, param, s, lpx, inp in zip(forward_funs, params, state, log_px, x):
            lpx, output, s = apply_fun(param, s, lpx, inp, condition, key=next(key_iter), **kwargs)
            densities.append(lpx)
            outputs.append(output)
            states.append(s)

            if(condition_on_results):
                condition = condition + (output,)

        return densities, outputs, tuple(states)

    def inverse(params, state, log_pz, z, condition, **kwargs):

        # Need to pop so that we don't resuse random keys!
        key = kwargs.pop('key', None)
        n_keys = n_layers if fe_apply_fun is None else n_layers*2
        keys = random.split(key, n_keys) if key is not None else (None,)*n_keys
        key_iter = iter(keys)

        # We need to store each of the outputs and state
        densities, outputs, states = [], [], []
        for apply_fun, param, s, lpz, inp in zip(inverse_funs, params, state, log_pz, z):
            lpz, output, updated_state = apply_fun(param, s, lpz, inp, condition, key=next(key_iter), **kwargs)
            densities.append(lpz)
            outputs.append(output)
            states.append(updated_state)

            # Conditioners are inputs during the inverse pass
            if(condition_on_results):
                condition = condition + (inp,)

        return densities, outputs, tuple(states)

    return init_fun, forward, inverse

def ReverseInputs(name='unnamed'):
    # language=rst
    """
    Reverse the order of inputs.  Not the same as reversing an array!
    """
    def init_fun(key, input_shape, condition_shape):
        params, state = (), ()
        return name, input_shape[::-1], params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        return log_px[::-1], x[::-1], state[::-1]

    def inverse(params, state, log_pz, z, condition, **kwargs):
        return log_pz[::-1], z[::-1], state[::-1]

    return init_fun, forward, inverse

################################################################################################################

def UnitGaussianPrior(axis=(-1,), name='unnamed'):
    # language=rst
    """
    Prior for the normalizing flow.

    :param axis - Axes to reduce over
    """
    def init_fun(key, input_shape, condition_shape):
        params, state = (), ()
        return name, input_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        dim = np.prod([x.shape[ax] for ax in axis])
        log_px += -0.5*np.sum(x**2, axis=axis) + -0.5*dim*np.log(2*np.pi)
        return log_px, x, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        # Usually we're sampling z from a Gaussian, so if we want to do Monte Carlo
        # estimation, ignore the value of N(z|0,I).
        return log_pz, z, state

    return init_fun, forward, inverse

def Dequantization(noise_scale=None, scale=256.0, name='unnamed'):
    # language=rst
    """
    Dequantization for images.

    :param noise_scale: An array that tells us how much noise to add to each dimension
    :param scale: What to divide the image by
    """
    noise_scale_array = None
    def init_fun(key, input_shape, condition_shape):
        params, state = (), ()
        if(noise_scale is None):
            nonlocal noise_scale_array
            noise_scale_array = np.ones(input_shape)
        else:
            assert noise_scale.shape == input_shape
            noise_scale_array = noise_scale
        return name, input_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        # Add uniform noise
        key = kwargs.pop('dq_key', None)
        if(key is None):
            # Try again
            key = kwargs.pop('key', None)
        if(key is None):
            noise = np.zeros_like(x)
        else:
            noise = random.uniform(key, x.shape)*noise_scale_array

        log_det = -np.log(scale)
        if(x.ndim > 2):
            if(x.ndim == 4):
                log_det *= np.prod(x.shape[1:])
            else:
                log_det *= np.prod(x.shape)

        return log_px + log_det, (x + noise)/scale, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        # Put the image back on the set of integers between 0 and 255
        z = z*scale
        # z = np.floor(z*scale).astype(np.int32)

        log_det = -np.log(scale)
        if(z.ndim > 2):
            if(z.ndim == 4):
                log_det *= np.prod(z.shape[1:])
            else:
                log_det *= np.prod(z.shape)

        return log_pz + log_det, z, state

    return init_fun, forward, inverse

def Augment(flow, sampler, name='unnamed'):
    # language=rst
    """
    Run a normalizing flow in an augmented space https://arxiv.org/pdf/2002.07101.pdf

    :param flow: The normalizing flow
    :param sampler: Function to sample from the convolving distribution
    """
    _init_fun, _forward, _inverse = flow

    def init_fun(key, input_shape, condition_shape):
        augmented_input_shape = input_shape[:-1] + (2*input_shape[-1],)
        return _init_fun(key, augmented_input_shape, condition_shape)

    def forward(params, state, log_px, x, condition, **kwargs):
        key = kwargs.pop('key', None)
        if(key is None):
            assert 0, 'Need a key for this'
        k1, k2 = random.split(key, 2)

        # Sample e and concatenate it to x
        e = random.normal(k1, x.shape)
        xe = np.concatenate([x, e], axis=-1)

        return _forward(params, state, log_px, xe, condition, key=k2, **kwargs)

    def inverse(params, state, log_pz, z, condition, **kwargs):
        key = kwargs.pop('key', None)
        if(key is None):
            assert 0, 'Need a key for this'
        k1, k2 = random.split(key, 2)

        x, e = np.split(z, axis=-1)

        return _inverse(params, state, log_pz, x, condition, key=k2, **kwargs)

    return init_fun, forward, inverse

def Convolve(flow, conv_sampler, n_training_importance_samples=1, name='unnamed'):
    # language=rst
    """
    Convolve a normalizing flow with noise

    :param flow: The normalizing flow
    :param conv_sampler: Function to sample from the convolving distribution
    """
    _init_fun, _forward, _inverse = flow

    def init_fun(key, input_shape, condition_shape):
        return _init_fun(key, input_shape, condition_shape)

    def apply_forward(params, state, log_px, x, condition, key, **kwargs):
        k1, k2 = random.split(key, 2)

        epsilon = conv_sampler(k1, x.shape, **kwargs)
        return _forward(params, state, log_px, x - epsilon, condition, key=k2, **kwargs)

    def forward(params, state, log_px, x, condition, **kwargs):
        # Sample epsilon, subtract it from x and then pass that through the rest of the flow

        key = kwargs.pop('key', None)
        if(key is None):
            assert 0, 'Need a key for this'

        filled_forward = partial(apply_forward, params, state, log_px, x, condition, **kwargs)

        n_importance_samples = kwargs.get('n_importance_samples', n_training_importance_samples)
        n_importance_samples = 1
        if(n_importance_samples == 1):
            return filled_forward(key)

        keys = random.split(key, n_importance_samples)
        log_pxs, zs, updated_states = vmap(filled_forward)(keys)
        return logsumexp(log_pxs, axis=0) - np.log(n_importance_samples), np.mean(zs, axis=0), updated_states#[0]

    def apply_inverse(params, state, log_pz, z, condition, key, **kwargs):
        k1, k2 = random.split(key, 2)

        # The forward pass does not need a determinant
        _, fz, updated_state = _inverse(params, state, log_pz, z, condition, key=k1, **kwargs)

        # Sample from p(x|f(z))
        epsilon = conv_sampler(k1, fz.shape, **kwargs)
        return log_pz, fz + epsilon, updated_state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        # Want to sample p(x|f(z))

        key = kwargs.pop('key', None)
        if(key is None):
            assert 0, 'Need a key for this'

        filled_inverse = partial(apply_inverse, params, state, log_pz, z, condition, **kwargs)

        n_importance_samples = kwargs.get('n_importance_samples', n_training_importance_samples)
        if(n_importance_samples == 1):
            return filled_inverse(key)

        keys = random.split(key, n_importance_samples)
        log_pxs, xs, updated_states = vmap(filled_inverse)(keys)
        return logsumexp(log_pxs, axis=0) - np.log(n_importance_samples), np.mean(xs, axis=0), updated_states#[0]

    return init_fun, forward, inverse

def OneWay(transform, name='unnamed'):
    # language=rst
    """
    Wrapper for staxplusplus networks.

    :param transform: spp network
    """
    _init_fun, _apply_fun = transform

    def init_fun(key, input_shape, condition_shape):
        transform_input_shape = input_shape if len(condition_shape) == 0 else (input_shape,) + condition_shape
        return _init_fun(key, transform_input_shape)

    def forward(params, state, log_px, x, condition, **kwargs):
        network_input = x if len(condition) == 0 else (x, *condition)
        z, updated_state = _apply_fun(params, state, network_input, **kwargs)
        return log_px, z, updated_state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        assert 0, 'Not invertible'

    return init_fun, forward, inverse

################################################################################################################

def LDSPrior(z_dim, name='unnamed'):
    # language=rst
    """
    Linear dynamical system prior.

    :param z_dim: Latent state dim
    """
    priors = None

    def init_fun(key, input_shape, condition_shape):
        x_dim = input_shape[-1]
        output_shape = input_shape[:-1] + (z_dim,)

        # No choice to initialize prior yet
        keys = random.split(key, 6)
        nonlocal priors
        state = (easy_niw_nat(keys[0], z_dim), easy_mniw_nat(keys[1], z_dim, z_dim), easy_mniw_nat(keys[2], x_dim, z_dim))
        priors = (easy_niw_nat(keys[3], z_dim), easy_mniw_nat(keys[4], z_dim, z_dim), easy_mniw_nat(keys[5], x_dim, z_dim))
        params = ()

        return name, input_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        # Must work with time series
        assert x.ndim == 2
        T = kwargs.get('T', x.shape[0])

        rho = kwargs.get('rho', 0.2)
        us = kwargs.get('us', np.zeros(1))
        mask = kwargs.get('mask', np.zeros(1))
        if(us.shape == (1,)):
            us = np.zeros((x.shape[0], z_dim))
        if(mask.shape == (1,)):
            mask = np.ones(x.shape[0]).astype(bool)

        # Run the kalman filter
        marginal, z, new_state = lds_svi(us, mask, priors, T, rho, state, x)

        # Don't update the initial state
        state = (state[0], new_state[1], new_state[2])

        return marginal + log_px.sum(), z, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        assert z.ndim == 2
        T = kwargs.get('T', z.shape[0])


        # Sample forward using the emission paramters
        # C = mniw_sample(*mniw_nat_to_std(*state[2]))[0]
        C = mniw_nat_to_std(*state[2])[0]
        x = np.einsum('ij,tj->ti', C, z)

        rho = kwargs.get('rho', 0.2)
        us = kwargs.get('us', np.zeros(1))
        mask = kwargs.get('mask', np.zeros(1))
        if(us.shape == (1,)):
            us = np.zeros_like(z)
        if(mask.shape == (1,)):
            mask = np.ones(x.shape[0]).astype(bool)

        marginal, _, _ = lds_svi(us, mask, priors, T, rho, state, x)
        return (marginal + log_pz), x, state

    return init_fun, forward, inverse

################################################################################################################

def Identity(name='unnamed'):
    # language=rst
    """
    Just pass an input forward.
    """
    def init_fun(key, input_shape, condition_shape):
        params, state = (), ()
        return name, input_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        return log_px, x, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        return log_pz, z, state

    return init_fun, forward, inverse

def Debug(name='unnamed'):
    # language=rst
    """
    Help debug shapes
    """
    def init_fun(key, input_shape, condition_shape):
        print(name, 'input_shape', input_shape)
        return name, input_shape, (), ()

    def forward(params, state, log_px, x, condition, **kwargs):
        return log_px, x, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        return log_pz, z, state

    return init_fun, forward, inverse

def Reverse(name='unnamed'):
    # language=rst
    """
    Reverse an input.
    """
    def init_fun(key, input_shape, condition_shape):
        params, state = (), ()
        return name, input_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        return log_px, x[...,::-1], state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        return log_pz, z[...,::-1], state

    return init_fun, forward, inverse

def Transpose(axis_order, name='unnamed'):
    # language=rst
    """
    Transpose an input
    """
    order = None
    order_inverse = None
    batched_order = None
    batched_order_inverse = None

    def init_fun(key, input_shape, condition_shape):

        nonlocal order
        nonlocal batched_order
        order = [ax%len(axis_order) for ax in axis_order]
        batched_order = [0] + [o + 1 for o in order]
        assert len(order) == len(input_shape)
        assert len(set(order)) == len(order)
        params, state = (), ()
        output_shape = [input_shape[ax] for ax in order]

        nonlocal order_inverse
        nonlocal batched_order_inverse
        order_inverse = [order.index(i) for i in range(len(order))]
        batched_order_inverse = [0] + [o + 1 for o in order_inverse]

        return name, output_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        if(x.ndim == 2 or x.ndim == 4):
            z = x.transpose(batched_order)
        else:
            z = x.transpose(order)
        return log_px, z, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        if(z.ndim == 2 or z.ndim == 4):
            x = z.transpose(batched_order_inverse)
        else:
            x = z.transpose(order_inverse)
        return log_pz, x, state

    return init_fun, forward, inverse


def Reshape(shape, name='unnamed'):
    # language=rst
    """
    Prior for the normalizing flow.

    :param shape - Shape to reshape to
    """

    # Need to keep track of the original shape in order to invert
    original_shape = None

    def init_fun(key, input_shape, condition_shape):
        nonlocal original_shape
        original_shape = input_shape
        params, state = (), ()
        return name, shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        if(x.ndim > len(original_shape)):
            z = x.reshape((-1,) + shape)
        else:
            z = x.reshape(shape)

        return log_px, z, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        if(z.ndim > len(shape)):
            x = z.reshape((-1,) + original_shape)
        else:
            x = z.reshape(original_shape)

        return log_pz, x, state

    return init_fun, forward, inverse

################################################################################################################

def FactorOut(num, axis=-1, name='unnamed'):
    # language=rst
    """
    Factor p(z_{1..N}) = p(z_1)p(z_2|z_1)...p(z_N|z_{1..N-1})

    :param num: Number of components to split into
    :param axis: Axis to split
    """
    def init_fun(key, input_shape, condition_shape):
        ax = axis % len(input_shape)

        # For the moment, ensure we split evenly
        assert input_shape[ax]%num == 0

        split_shape = list(input_shape)
        split_shape[ax] = input_shape[ax]//num
        split_shape = tuple(split_shape)

        params, state = (), ()
        return name, [split_shape]*num, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        z_components = np.split(x, num, axis)

        # Only send the total density through one component.  The density will be recombined later
        log_pxs = [log_px if i == 0 else np.zeros_like(log_px) for i, z_i in enumerate(z_components)]
        zs = z_components

        return log_pxs, zs, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        log_pz = sum(log_pz)
        x = np.concatenate(z, axis)
        return log_pz, x, state

    return init_fun, forward, inverse

def FanInConcat(num, axis=-1, name='unnamed'):
    # language=rst
    """
    Inverse of FactorOut

    :param num: Number of components to split into
    :param axis: Axis to split
    """
    def init_fun(key, input_shape, condition_shape):
        # Make sure that each of the inputs are the same size
        assert num == len(input_shape)
        for shape in input_shape:
            assert shape == input_shape[0]
        ax = axis % len(input_shape[0])
        concat_size = sum(shape[ax] for shape in input_shape)
        out_shape = input_shape[0][:ax] + (concat_size,) + input_shape[0][ax+1:]
        params, state = (), ()
        return name, out_shape, params, state

    _, inverse, forward = FactorOut(num, axis=axis)

    return init_fun, forward, inverse

################################################################################################################

def indexer_and_shape_from_mask(mask):
    # language=rst
    """
    Given a 2d mask array, create an array that can index into a vector with the same number of elements
    as nonzero elements in mask and result in an array of the same size as mask, but with the elements
    specified from the vector. Also return the shape of the resulting array when mask is applied.

    :param mask: 2d boolean mask array
    """
    index = onp.zeros_like(mask, dtype=int)
    non_zero_indices = np.nonzero(mask)
    index[non_zero_indices] = np.arange(len(non_zero_indices[0])) + 1

    nonzero_x, nonzero_y = non_zero_indices
    n_rows = onp.unique(nonzero_x).size
    assert nonzero_x.size%n_rows == 0
    n_cols = nonzero_x.size // n_rows
    shape = (n_rows, n_cols)
    return index, shape

def check_mask(mask):
    # language=rst
    """
    Check if the 2d boolean mask is valid

    :param mask: 2d boolean mask array
    """
    if(np.any(mask) == False):
        assert 0, 'Empty mask!  Reduce num'
    if(np.sum(mask)%2 == 1):
        assert 0, 'Need masks with an even number!  Choose a different num'

def checkerboard_masks(num, shape):
    # language=rst
    """
    Finds masks to factor an array with a given shape so that each pixel will be
    present in the resulting masks.  Also return indices that will help reverse
    the factorization.

    :param masks: A list of 2d boolean mask array whose union is equal to np.ones(shape, dtype=bool)
    :param indices: A list of index matrices that undo the application of image[mask]
    """
    masks = []
    indices = []
    shapes = []

    for i in range(2*num):
        start = 2**i
        step = 2**(i + 1)

        # Staggered indices
        mask = onp.zeros(shape, dtype=bool)
        mask[start::step,::step] = True
        mask[::step,start::step] = True
        check_mask(mask)
        masks.append(mask)
        index, new_shape = indexer_and_shape_from_mask(mask)
        indices.append(index)
        shapes.append(new_shape)

        if(len(masks) + 1 == num):
            break

        # Not staggered indices
        mask = onp.zeros(shape, dtype=bool)
        mask[start::step,start::step] = True
        mask[start::step,start::step] = True
        check_mask(mask)
        masks.append(mask)
        index, new_shape = indexer_and_shape_from_mask(mask)
        indices.append(index)
        shapes.append(new_shape)

        if(len(masks) + 1 == num):
            break

    used = sum(masks).astype(bool)
    mask = ~used
    masks.append(mask)
    index, new_shape = indexer_and_shape_from_mask(mask)
    indices.append(index)
    shapes.append(new_shape)

    return masks, indices, shapes

def recombine(z, index):
    # language=rst
    """
    Use a structured set of indices to create a matrix from a vector

    :param z: Flat input that contains the elements of the output matrix
    :param indices: An array of indices that correspond to values in z
    """
    return np.pad(z.ravel(), (1, 0))[index]

def CheckerboardFactor(num, name='unnamed'):
    # language=rst
    """
    Factor an image using a checkerboard pattern.  Basically each split will be a lower resolution
    image of the original.  See Figure 3 here https://arxiv.org/pdf/1605.08803.pdf for details on checkerboard.
    Only produces the checkerboard factors, does not concatenate them!

    :param num: Number of components to split into
    """
    masks, indices, shapes = None, None, None

    def init_fun(key, input_shape, condition_shape):
        height, width, channel = input_shape

        nonlocal masks, indices, shapes
        masks, indices, shapes = checkerboard_masks(num, (height, width))

        # Swap the order so that our factors go from lowest to highest resolution
        masks, indices, shapes = masks[::-1], indices[::-1], shapes[::-1]

        # Add the channel dim back in
        shapes = [(h, w, channel) for (h, w) in shapes]

        params, state = (), ()
        return name, shapes, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        if(x.ndim == 4):
            # To make things easier, vmap over the batch dimension
            return vmap(partial(forward, params, state, **kwargs), in_axes=(0, 0, None))(log_px, x, condition)

        assert x.ndim == 3

        # Split the pixels into disjoint sets
        zs = []
        for mask, shape in zip(masks, shapes):
            z = x[mask].reshape(shape)
            zs.append(z)

        # Only send the total density through one component.  The density will be recombined later
        log_pxs = [log_px if i == 0 else np.zeros_like(log_px) for i in range(num)]

        return log_pxs, zs, state

    def inverse(params, state, log_pz, z, condition, **kwargs):

        # Add the densities
        log_pz = sum(log_pz)

        # Recombine the pixels into an image
        recombine_vmapped = vmap(recombine, in_axes=(2, None), out_axes=2)

        # If z is batched, then need an extra vmap
        if(z[0].ndim == 4):
            recombine_vmapped = vmap(recombine_vmapped, in_axes=(0, None), out_axes=0)

        x = recombine_vmapped(z[0], indices[0])
        for elt, index in zip(z[1:], indices[1:]):
            x += recombine_vmapped(elt, index)

        return log_pz, x, state

    return init_fun, forward, inverse

def CheckerboardCombine(num, name='unnamed'):
    # language=rst
    """
    Inverse of CheckerboardFactor

    :param num: Number of components to split into
    """
    masks, indices, shapes = None, None, None

    def init_fun(key, input_shape, condition_shape):
        assert num == len(input_shape)

        # By construction, the height of the last shape is the height of the total image
        height, _, channel = input_shape[-1]

        # Count the total number of pixels
        total_pixels = 0
        for h, w, c in input_shape:
            total_pixels += h*w

        assert total_pixels%height == 0
        width = total_pixels // height

        output_shape = (height, width, channel)

        # Need to know ahead of time what the masks are and the indices and shapes to undo the masks are
        nonlocal masks, indices, shapes
        masks, indices, shapes = checkerboard_masks(num, (height, width))

        # Swap the order so that our factors go from lowest to highest resolution
        masks, indices, shapes = masks[::-1], indices[::-1], shapes[::-1]

        # Add the channel dim back in
        shapes = [(h, w, channel) for (h, w) in shapes]

        params, state = (), ()
        return name, output_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):

        # Add the densities
        log_px = sum(log_px)

        # Recombine the pixels into an image
        recombine_vmapped = vmap(recombine, in_axes=(2, None), out_axes=2)

        # If x is batched, then need an extra vmap
        if(x[0].ndim == 4):
            recombine_vmapped = vmap(recombine_vmapped, in_axes=(0, None), out_axes=0)

        z = recombine_vmapped(x[0], indices[0])
        for elt, index in zip(x[1:], indices[1:]):
            z += recombine_vmapped(elt, index)

        return log_px, z, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        if(z.ndim == 4):
            # To make things easier, vmap over the batch dimension
            return vmap(partial(inverse, params, state, **kwargs), in_axes=(0, 0, None))(log_pz, z, condition)

        assert z.ndim == 3

        # Split the pixels into disjoint sets
        xs = []
        for mask, shape in zip(masks, shapes):
            x = z[mask].reshape(shape)
            xs.append(x)

        # Only send the total density through one component.  The density will be recombined later
        log_pzs = [log_pz if i == 0 else np.zeros_like(log_pz) for i in range(num)]

        return log_pzs, xs, state

    return init_fun, forward, inverse

################################################################################################################

def CheckerboardSqueeze(num=2, name='unnamed'):
    # language=rst
    """

    :param num: Number of components to split into
    """
    assert num == 2, 'More not implemented yet'
    masks, indices, shapes = None, None, None

    def init_fun(key, input_shape, condition_shape):
        height, width, channel = input_shape

        # Find the masks in order to split the image correctly
        nonlocal masks, indices, shapes
        masks, indices, shapes = checkerboard_masks(2, (height, width))
        shapes = [(h, w, channel) for (h, w) in shapes]

        # We need to get the same shapes
        assert shapes[0] == shapes[1]

        # Get the output shape
        out_height, out_width, _ = shapes[0]
        output_shape = (out_height, out_width, 2*channel)

        # This is by construction!
        assert out_height == height

        params, state = (), ()
        return name, output_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        if(x.ndim == 4):
            # To make things easier, vmap over the batch dimension
            return vmap(partial(forward, params, state, **kwargs), in_axes=(0, 0, None))(log_px, x, condition)

        assert x.ndim == 3

        # Split the image and concatenate along the channel dimension
        z1 = x[masks[0]].reshape(shapes[0])
        z2 = x[masks[1]].reshape(shapes[1])
        z = np.concatenate([z1, z2], axis=-1)

        # This is basically a permuation matrix, so the log abs determinant is 0
        return log_px, z, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        # Split the image on the channel dimension
        x1, x2 = np.split(z, 2, axis=-1)

        # Recombine the pixels into an image
        recombine_vmapped = vmap(recombine, in_axes=(2, None), out_axes=2)

        # If z is batched, then need an extra vmap
        if(x1.ndim == 4):
            recombine_vmapped = vmap(recombine_vmapped, in_axes=(0, None), out_axes=0)

        x = recombine_vmapped(x1, indices[0]) + recombine_vmapped(x2, indices[1])

        return log_pz, x, state

    return init_fun, forward, inverse

def CheckerboardUnSqueeze(num=2, name='unnamed'):
    # language=rst
    """
    Inverse of CheckerboardSqueeze.

    :param num: Number of components to split into
    """
    assert num == 2, 'More not implemented yet'
    masks, indices, shapes = None, None, None

    def init_fun(key, input_shape, condition_shape):
        # Height remained unchanged
        height, width, channel = input_shape
        assert channel%2 == 0

        # Find the output shape
        out_height = height
        out_width = width*2
        out_channel = channel // 2
        output_shape = (out_height, out_width, out_channel)

        # Create the masks
        nonlocal masks, indices, shapes
        masks, indices, shapes = checkerboard_masks(2, (out_height, out_width))
        shapes = [(h, w, out_channel) for (h, w) in shapes]

        params, state = (), ()
        return name, output_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        # Split the image on the channel dimension
        z1, z2 = np.split(x, 2, axis=-1)

        # Recombine the pixels into an image
        recombine_vmapped = vmap(recombine, in_axes=(2, None), out_axes=2)

        # If x is batched, then need an extra vmap
        if(z1.ndim == 4):
            recombine_vmapped = vmap(recombine_vmapped, in_axes=(0, None), out_axes=0)

        z = recombine_vmapped(z1, indices[0]) + recombine_vmapped(z2, indices[1])

        return log_px, z, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        if(z.ndim == 4):
            # To make things easier, vmap over the batch dimension
            return vmap(partial(inverse, params, state, **kwargs), in_axes=(0, 0, None))(log_pz, z, condition)

        assert z.ndim == 3

        # Split the image and concatenate along the channel dimension
        x1 = z[masks[0]].reshape(shapes[0])
        x2 = z[masks[1]].reshape(shapes[1])
        x = np.concatenate([x1, x2], axis=-1)

        # This is basically a permuation matrix, so the log abs determinant is 0
        return log_pz, x, state

    return init_fun, forward, inverse

def Squeeze(name='unnamed'):
    # language=rst
    """
    Inverse of CheckerboardSqueeze.

    :param num: Number of components to split into
    """
    def init_fun(key, input_shape, condition_shape):
        H, W, C = input_shape
        assert H%2 == 0
        assert W%2 == 0
        output_shape = (H//2, W//2, C*4)
        params, state = (), ()
        return name, output_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        if(x.ndim == 4):
            # Handle batching this way
            return vmap(partial(forward, params, state, log_px, **kwargs), in_axes=(0, None))(x, condition)

        # Turn to (C, H, W)
        z = x.transpose((2, 0, 1))

        C, H, W = z.shape
        z = z.reshape((C, H//2, 2, W//2, 2))
        z = z.transpose((0, 2, 4, 1, 3))
        z = z.reshape((C*4, H//2, W//2))

        # Turn back to (H, W, C)
        z = z.transpose((1, 2, 0))
        return log_px, z, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        if(z.ndim == 4):
            # Handle batching this way
            return vmap(partial(inverse, params, state, log_pz, **kwargs), in_axes=(0, None))(z, condition)

        # Turn to (C, H, W)
        x = z.transpose((2, 0, 1))

        C, H, W = x.shape
        x = x.reshape((C//4, 2, 2, H, W))
        x = x.transpose((0, 3, 1, 4, 2))
        x = x.reshape((C//4, H*2, W*2))

        # Turn back to (H, W, C)
        x = x.transpose((1, 2, 0))
        return log_pz, x, state

    return init_fun, forward, inverse

def UnSqueeze(name='unnamed'):
    _, forward, inverse = Squeeze(name=name)

    def init_fun(key, input_shape, condition_shape):
        H, W, C = input_shape
        assert C%4 == 0
        output_shape = (H*2, W*2, C//4)
        params, state = (), ()
        return name, output_shape, params, state

    return init_fun, inverse, forward

################################################################################################################

def UpSample(repeats, name='unnamed'):
    # language=rst
    """
    Up sample by just repeating consecutive values over specified axes

    :param repeats - The number of times to repeat.  Pass in (2, 1, 2), for example, to repeat twice over
                     the 0th axis, no repeats over the 1st axis, and twice over the 2nd axis
    """
    full_repeats = None
    def init_fun(key, input_shape, condition_shape):
        nonlocal full_repeats
        full_repeats = [repeats[i] if i < len(repeats) else 1 for i in range(len(input_shape))]
        output_shape = []
        for s, r in zip(input_shape, full_repeats):
            assert s%r == 0
            output_shape.append(s//r)
        output_shape = tuple(output_shape)
        log_sigma = -1.0
        params, state = (log_sigma), ()
        return name, output_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        log_sigma = params
        sigma = np.exp(log_sigma)
        is_batched = int(x.ndim == 2 or x.ndim == 4)

        # The pseudo-inverse is the sliced input
        slices = [slice(0, None, r) for r in full_repeats]

        if(is_batched):
            z = x[[slice(0, None, 1)] + slices]
        else:
            z = x[slices]

        # Find the flat dimensions
        if(x.ndim <= 2):
            dim_z = z.shape[-1]
            dim_x = x.shape[-1]
        elif(x.ndim <= 4):
            dim_z = np.prod(z.shape[-3:])
            dim_x = np.prod(x.shape[-3:])

        # Find the projection
        x_proj = z
        for i, r in enumerate(repeats):
            x_proj = np.repeat(x_proj, r, axis=i + is_batched)

        # Compute -0.5|J^T J|
        repeat_prod = np.prod(repeats)
        log_det = -0.5*np.log(repeat_prod)*dim_x

        log_hx = -0.5/sigma*np.sum(x*(x - x_proj), axis=-1)
        log_hx -= log_det
        log_hx -= (dim_x - dim_z)*(log_sigma + np.log(2*np.pi))

        # If we have an image, need to sum over more axes
        if(x.ndim >= 3):
            if(log_hx.ndim >= 2):
                log_hx = log_hx.sum(axis=(-1, -2))

        # J^T J is a diagonal matrix where each diagonal element is the
        # the product of repeats
        key = kwargs.pop('key', None)
        if(key is None):
            assert 0, 'Need a key for this'

        # Sample z ~ N(z^+, sigma(J^T J)^{-1})
        noise = random.normal(key, z.shape)
        z += noise*np.sqrt(sigma/repeat_prod)

        return log_px + log_hx, z, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        log_sigma = params
        x = z
        is_batched = int(x.ndim == 2 or x.ndim == 4)
        for i, r in enumerate(repeats):
            x = np.repeat(x, r, axis=i + is_batched)

        # Find the flat dimensions
        if(x.ndim <= 2):
            dim_z = z.shape[-1]
            dim_x = x.shape[-1]
        elif(x.ndim <= 4):
            dim_z = np.prod(z.shape[-3:])
            dim_x = np.prod(x.shape[-3:])

        # Compute -0.5|J^T J|
        repeat_prod = np.prod(repeats)
        log_det = -0.5*np.log(repeat_prod)*dim_x

        # The projection difference is 0!
        log_hx = log_det
        log_hx -= (dim_x - dim_z)*(log_sigma + np.log(2*np.pi))

        # If we have an image, need to sum over more axes
        if(x.ndim >= 3):
            if(log_hx.ndim >= 2):
                log_hx = log_hx.sum(axis=(-1, -2))

        return log_pz + log_hx, x, state

    return init_fun, forward, inverse

################################################################################################################

def flow_data_dependent_init(x, target_param_names, name_tree, params, state, forward, condition, flag_names, **kwargs):
    # language=rst
    """
    Data dependent initialization for a normalizing flow.

    :param x: The data seed
    :param target_param_names: A list of the names of parameters to seed
    :param name_tree: A pytree (nested structure) of names.  This is the first output of an init_fun call
    :param params: The parameter pytree
    :param state: The state pytree
    :param forward: Forward function
    :param flag_names: The names of the flag that will turn on seeding.

    **Example**

    .. code-block:: python
        from jax import random
        import jax.numpy as np
        from normalizing_flows import ActNorm, flow_data_dependent_init
        from util import TRAIN, TEST

        # Create the model
        flow = ActNorm(name='an')

        # Initialize it
        init_fun, forward, inverse = flow
        key = random.PRNGKey(0)
        names, output_shape, params, state = init_fun(key, input_shape=(5, 5, 3), condition_shape=())

        # Seed weight norm and retrieve the new parameters
        data_seed = np.ones((10, 5, 5, 3))
        actnorm_names = ['an']
        params = flow_data_dependent_init(data_seed, actnorm_names, names, params, state, forward, (), 'actnorm_seed')
    """
    def filled_forward_function(params, state, x, **kwargs):
        _, ans, updated_states = forward(params, state, np.zeros(x.shape[0]), x, condition, **kwargs)
        return ans, updated_states

    return spp.data_dependent_init(x, target_param_names, name_tree, params, state, filled_forward_function, flag_names, **kwargs)

def multistep_flow_data_dependent_init(x,
                                       target_param_names,
                                       flow_model,
                                       condition,
                                       flag_names,
                                       key,
                                       n_seed_examples=1000,
                                       batch_size=4,
                                       notebook=True,
                                       **kwargs):
    # language=rst
    """
    Data dependent initialization for a normalizing flow that is split up into multiple steps

    :param x: The data seed
    :param target_param_names: A list of the names of parameters to seed
    :param name_tree: A pytree (nested structure) of names.  This is the first output of an init_fun call
    :param params: The parameter pytree
    :param state: The state pytree
    :param forward: Forward function
    :param flag_names: The names of the flag that will turn on seeding.
    """
    (names, output_shape, params, state), forward, inverse = flow_model

    seed_steps = int(np.ceil(n_seed_examples/batch_size))

    # Get the inital parameters
    flat_params, unflatten = ravel_pytree(params)
    unflatten = jit(unflatten)

    # JIT the forward function.  Need to fill the kwargs before jitting otherwise this will fail.
    if(isinstance(flag_names, list) == False and isinstance(flag_names, tuple) == False):
        flag_names = (flag_names,)
    flag_names = dict([(name, True) for name in flag_names])
    jitted_forward = jit(partial(forward, **flag_names))

    # Define a single gpu slice of the dependent init
    @jit
    def single_gpu_init(params, key, x_batch):
        new_params = flow_data_dependent_init(x_batch, target_param_names, names, params, state, jitted_forward, (), None, key=key)
        new_flat_params, _ = ravel_pytree(new_params)
        return new_flat_params

    # Run the data dependent initialization
    pbar = tnrange(seed_steps) if notebook else tqdm(range(seed_steps))
    for i in pbar:
        key, *keys = random.split(key, 3)

        # Get the next batch of data for each gpu
        batch_idx = random.randint(keys[0], (batch_size,), minval=0, maxval=x.shape[0])
        x_batch = x[batch_idx,:]

        # Compute the seeded parameters
        new_params = flow_data_dependent_init(x_batch, target_param_names, names, params, state, jitted_forward, (), None, key=key)
        new_flat_params, _ = ravel_pytree(new_params)

        # Compute a running mean of the parameters
        flat_params = i/(i + 1)*flat_params + new_flat_params/(i + 1)
        params = unflatten(flat_params)

    return params

def ActNorm(log_s_init=zeros, b_init=zeros, name='unnamed'):
    # language=rst
    """
    Act norm normalization.  Act norm requires a data seed in order to properly initialize
    its parameters.  This will be done at runtime.

    :param axis: Batch axis
    """

    def init_fun(key, input_shape, condition_shape):
        k1, k2 = random.split(key)
        log_s = log_s_init(k1, (input_shape[-1],))
        b = b_init(k2, (input_shape[-1],))

        params = (log_s, b)
        state = ()
        return name, input_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        log_s, b = params

        # Check to see if we're seeding this function
        actnorm_seed = kwargs.get('actnorm_seed', False)
        if(actnorm_seed == True):
            # The initial parameters should normalize the input
            # We want it to be normalized over the channel dimension!
            axes = tuple(np.arange(len(x.shape) - 1))
            mean = np.mean(x, axis=axes)
            std = np.std(x, axis=axes) + 1e-5
            log_s = np.log(std)
            b = mean
            updated_state = (log_s, b)
        else:
            updated_state = ()

        z = (x - b)*np.exp(-log_s)
        log_det = -log_s.sum()

        # Need to multiply by the height/width!
        if(z.ndim == 4 or z.ndim == 3):
            height, width, channel = z.shape[-3], z.shape[-2], z.shape[-1]
            log_det *= height*width

        return log_px + log_det, z, updated_state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        log_s, b = params
        x = np.exp(log_s)*z + b
        log_det = -log_s.sum()

        # Need to multiply by the height/width!
        if(z.ndim == 4 or z.ndim == 3):
            height, width, channel = z.shape[-3], z.shape[-2], z.shape[-1]
            log_det *= height*width

        return log_pz + log_det, x, state

    return init_fun, forward, inverse

def BatchNorm(epsilon=1e-5, alpha=0.05, beta_init=zeros, gamma_init=zeros, name='unnamed'):
    # language=rst
    """
    Invertible batch norm.

    :param axis: Batch axis
    :param epsilon: Constant for numerical stability
    :param alpha: Parameter for exponential moving average of population parameters
    """
    def init_fun(key, input_shape, condition_shape):
        k1, k2 = random.split(key)
        beta, log_gamma = beta_init(k1, input_shape), gamma_init(k2, input_shape)
        running_mean = np.zeros(input_shape)
        running_var = np.ones(input_shape)
        params = (beta, log_gamma)
        state = (running_mean, running_var)
        return name, input_shape, params, state

    def get_bn_params(x, test, running_mean, running_var):
        """ Update the batch norm statistics """
        if(is_testing(test)):
            mean, var = running_mean, running_var
        else:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0) + epsilon
            running_mean = (1 - alpha)*running_mean + alpha*mean
            running_var = (1 - alpha)*running_var + alpha*var

        return (mean, var), (running_mean, running_var)

    def forward(params, state, log_px, x, condition, **kwargs):
        beta, log_gamma = params
        running_mean, running_var = state

        # Check if we're training or testing
        test = kwargs['test'] if 'test' in kwargs else TRAIN

        # Update the running population parameters
        (mean, var), (running_mean, running_var) = get_bn_params(x, test, running_mean, running_var)

        # Normalize the inputs
        x_hat = (x - mean) / np.sqrt(var)
        z = np.exp(log_gamma)*x_hat + beta

        log_det = log_gamma.sum()*np.ones((z.shape[0],))
        log_det += -0.5*np.log(var).sum()

        updated_state = (running_mean, running_var)
        return log_px + log_det, z, updated_state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        beta, log_gamma = params
        running_mean, running_var = state

        # Check if we're training or testing
        test = kwargs['test'] if 'test' in kwargs else TRAIN

        # Update the running population parameters
        (mean, var), (running_mean, running_var) = get_bn_params(z, test, running_mean, running_var)

        # Normalize the inputs
        z_hat = (z - beta)*np.exp(-log_gamma)
        x = z_hat*np.sqrt(var) + mean

        log_det = log_gamma.sum()*np.ones((z.shape[0],))
        log_det += -0.5*np.log(var).sum()

        updated_state = (running_mean, running_var)
        return log_pz + log_det, x, updated_state

    return init_fun, forward, inverse

################################################################################################################

def AffineLDU(L_init=normal(), d_init=ones, U_init=normal(), name='unnamed', return_mat=False):
    # language=rst
    """
    LDU parametrized square dense matrix
    """

    triangular_indices = None

    def init_fun(key, input_shape, condition_shape):
        # This is a square matrix!
        dim = input_shape[-1]

        # Create the fancy indices that we'll use to turn our vectors into triangular matrices
        nonlocal triangular_indices
        triangular_indices = np.pad(util.upper_triangular_indices(dim - 1), ((0, 1), (1, 0)))
        flat_dim = util.n_elts_upper_triangular(dim)

        k1, k2, k3 = random.split(key, 3)
        L_flat, d, U_flat = L_init(k1, (flat_dim,)), d_init(k2, (dim,)), U_init(k3, (flat_dim,))

        params = (L_flat, d, U_flat)
        state = ()
        return name, input_shape, params, state

    def get_LDU(params):
        L_flat, d, U_flat = params

        L = np.pad(L_flat, (1, 0))[triangular_indices]
        L = L + np.eye(d.shape[-1])
        L = L.T

        U = np.pad(U_flat, (1, 0))[triangular_indices]
        U = U + np.eye(d.shape[-1])

        return L, d, U

    def forward(params, state, log_px, x, condition, **kwargs):
        # Go from x to x
        if(x.ndim == 2):
            return vmap(partial(forward, params, state, **kwargs), in_axes=(0, 0, None))(log_px, x, condition)

        L, d, U = get_LDU(params)

        z = np.einsum('ij,j->i', U, x)
        z = z*d
        z = np.einsum('ij,j->i', L, z)

        log_det = np.sum(np.log(np.abs(d)), axis=-1)

        return log_px + log_det, z, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        if(z.ndim == 2):
            return vmap(partial(inverse, params, state, **kwargs), in_axes=(0, 0, None))(log_pz, z, condition)

        L, d, U = get_LDU(params)

        x = util.L_solve(L, z)
        x = x/d
        x = util.U_solve(U, x)

        log_det = np.sum(np.log(np.abs(d)), axis=-1)

        return log_pz + log_det, x, state

    # Have the option to directly get the matrix
    if(return_mat):
        return init_fun, forward, inverse, get_LDU

    return init_fun, forward, inverse

def AffineSVD(n_householders, U_init=glorot_normal(), log_s_init=normal(), VT_init=glorot_normal(), name='name'):
    # language=rst
    """
    Affine matrix with SVD parametrization.  Uses a product of householders to parametrize the orthogonal matrices.

    :param n_householders: Number of householders to use in U and V parametrization.  When n_householders = dim(x),
                           we can represent any orthogonal matrix with det=-1 (I think?)
    """
    def init_fun(key, input_shape, condition_shape):
        keys = random.split(key, 3)
        U = U_init(keys[0], (n_householders, input_shape[-1]))
        log_s = log_s_init(keys[1], (input_shape[-1],))
        VT = VT_init(keys[2], (n_householders, input_shape[-1]))
        params = (U, log_s, VT)
        state = ()
        return name, input_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        U, log_s, VT = params

        if(x.ndim == 2):
            householder_fun = vmap(householder_prod, in_axes=(0, None))
        else:
            householder_fun = householder_prod

        z = householder_fun(x, VT)
        z = z*np.exp(log_s)
        z = householder_fun(z, U)
        log_det = log_s.sum()
        return log_px + log_det, z, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        U, log_s, VT = params

        if(z.ndim == 2):
            householder_fun = vmap(householder_prod_transpose, in_axes=(0, None))
        else:
            householder_fun = householder_prod_transpose

        x = householder_fun(z, U)
        x = x*np.exp(-log_s)
        x = householder_fun(x, VT)
        log_det = log_s.sum()
        return log_pz + log_det, x, state

    return init_fun, forward, inverse

def Affine(*args, mode='LDU', **kwargs):
    # language=rst
    """
    Affine matrix with choice of parametrization

    :param mode: Name of parametrization choice
    """
    if(mode == 'LDU'):
        return AffineLDU(*args, **kwargs)
    elif(mode == 'SVD'):
        return AffineSVD(*args, **kwargs)
    else:
        assert 0, 'Invalid choice of affine backend'

def GeneralAffine(flow, out_dim, prior=1e3, n_training_importance_samples=8, A_init=glorot_normal(), name='unnamed'):
    """ Affine function to change dimensions.  Doesn't actually work!

        Args:
            flow - The normalizing flow between this point and the latent dim
            out_dim - The dimension of "flow"
            prior - Variance of the prior.  Higher numbers -> less impactful prior
    """
    _init_fun, _forward, _inverse = flow

    def init_fun(key, input_shape, condition_shape):
        x_shape = input_shape
        output_shape = x_shape[:-1] + (out_dim,)
        keys = random.split(key, 2)
        A = A_init(keys[0], (x_shape[-1], out_dim))
        flow_name, flow_output_shape, flow_params, flow_state = _init_fun(keys[1], output_shape, condition_shape)
        params = (A, flow_params)
        state = flow_state
        return name, flow_output_shape, params, flow_state

    def apply_forward(params, state, log_px, condition, z, sigma_ATA_chol, key, **kwargs):
        A, flow_params = params
        flow_state = state

        k1, k2 = random.split(key, 2)

        # Sample from N(z|\mu(x),\Sigma(x))
        noise = random.normal(k1, z.shape)
        if(z.ndim == 1):
            z += sigma_ATA_chol@noise
        elif(z.ndim == 2):
            z += np.einsum('ij,bj->bi', sigma_ATA_chol, noise)
        else:
            assert 0, 'Can only handle 1d or 2d inputs'

        return _forward(flow_params, state, log_px, z, condition, key=k2, **kwargs)

    def forward(params, state, log_px, x, condition, **kwargs):
        A, flow_params = params

        key = kwargs.pop('key', None)
        if(key is None):
            assert 0, 'Need a key for this'

        sigma = kwargs.pop('sigma', 0.1)

        # Compute the regularized pseudoinverse
        # ATA = A.T@A
        ATA = sigma/prior*np.eye(A.shape[1]) + A.T@A
        ATA_inv = np.linalg.inv(ATA)
        A_pinv = ATA_inv@A.T
        if(x.ndim == 1):
            z = np.einsum('ij,j->i', A_pinv, x)
            x_proj = np.einsum('ij,j->i', A, z)
        elif(x.ndim == 2):
            z = np.einsum('ij,bj->bi', A_pinv, x)
            x_proj = np.einsum('ij,bj->bi', A, z)
        else:
            assert 0, 'Can only handle 1d or 2d inputs'

        # Get the conv noise
        sigma_ATA_chol = np.linalg.cholesky(sigma*ATA_inv)

        # Get the terms that don't depend on z
        dim_x, dim_z = A.shape
        log_hx = 0.5/sigma*np.sum(x*(x_proj - x), axis=-1)
        log_hx -= 0.5*np.linalg.slogdet(ATA)[1]
        log_hx -= 0.5*(dim_x - dim_z)*np.log(sigma)
        log_hx -= 0.5*dim_x*np.log(2*np.pi)
        log_hx -= 0.5*dim_z*np.log(prior)

        filled_forward = partial(apply_forward, params, state, log_px, condition, z, sigma_ATA_chol, **kwargs)
        n_importance_samples = kwargs.get('n_importance_samples', n_training_importance_samples)
        keys = random.split(key, n_importance_samples)

        # log_pxs, z, updated_states = filled_forward(key)
        # return log_pxs + log_hx, z, updated_states

        log_pxs, zs, updated_states = vmap(filled_forward)(keys)
        return log_hx + logsumexp(log_pxs, axis=0) - np.log(n_importance_samples), np.mean(zs, axis=0), updated_states

    def inverse(params, state, log_pz, z, condition, **kwargs):
        A, flow_params = params
        flow_state = state

        log_pz, z, updated_state = _inverse(flow_params, flow_state, log_pz, z, condition, **kwargs)

        if(z.ndim == 1):
            x = np.einsum('ij,j->i', A, z)
        elif(z.ndim == 2):
            x = np.einsum('ij,bj->bi', A, z)
        else:
            assert 0, 'Got an invalid shape.  z.shape: %s'%(str(z.shape))

        # Ignore the prior!  This leads to biased sampling, so use a weak prior!!!
        ATA = A.T@A

        # Get the conv noise
        sigma = kwargs.pop('sigma', None)

        if(sigma is not None):

            key = kwargs.pop('key', None)
            if(key is None):
                assert 0, 'Need a key for this'
            noise = random.normal(key, x.shape)*np.sqrt(sigma)
            x += noise

        # There is no manifold loss in this direction
        return log_pz - 0.5*np.linalg.slogdet(ATA)[1], x, updated_state

    return init_fun, forward, inverse

################################################################################################################

def OnebyOneConv(name='unnamed'):
    # language=rst
    """
    Invertible 1x1 convolution.  Implemented as matrix multiplication over the channel dimension
    """
    affine_forward, affine_inverse = None, None

    def init_fun(key, input_shape, condition_shape):
        height, width, channel = input_shape

        nonlocal affine_forward, affine_inverse
        affine_init_fun, affine_forward, affine_inverse = AffineLDU()
        _, _, params, state = affine_init_fun(key, (channel,), condition_shape)
        return name, input_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        if(x.ndim == 4):
            # vmap over the batch dim
            z = vmap(partial(forward, params, state, **kwargs), in_axes=(0, 0, None))(log_px, x, condition)
            return z

        # need to vmap over the height and width axes
        assert x.ndim == 3
        over_width = vmap(partial(affine_forward, params, state, **kwargs), in_axes=(None, 0, None))
        over_height_width = vmap(over_width, in_axes=(None, 0, None))

        # Not sure what to do about the updated state in this case
        log_det, z, _ = over_height_width(0, x, condition)
        return log_px + log_det.sum(axis=(-2, -1)), z, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        if(z.ndim == 4):
            # vmap over the batch dim
            x = vmap(partial(inverse, params, state, **kwargs), in_axes=(0, 0, None))(log_pz, z, condition)
            return x

        # need to vmap over the height and width axes
        assert z.ndim == 3
        over_width = vmap(partial(affine_inverse, params, state, **kwargs), in_axes=(None, 0, None))
        over_height_width = vmap(over_width, in_axes=(None, 0, None))

        # Not sure what to do about the updated state in this case
        log_det, x, _ = over_height_width(0, z, condition)
        return log_pz + log_det.sum(axis=(-2, -1)), x, state

    return init_fun, forward, inverse

################################################################################################################

def LeakyReLU(alpha=0.01, name='unnamed'):
    # language=rst
    """
    Leaky ReLU

    :param alpha: Slope for negative inputs
    """
    def init_fun(key, input_shape, condition_shape):
        params, state = (), ()
        return name, input_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        z = np.where(x > 0, x, alpha*x)

        log_dx_dz = np.where(x > 0, 0, np.log(alpha))
        log_det = log_dx_dz.sum(axis=-1)

        if(log_det.ndim > 1):
            # Then we have an image and have to sum over the height and width axes
            log_det = log_det.sum(axis=(-2, -1))

        return log_px + log_det, z, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        x = np.where(z > 0, z, z/alpha)

        log_dx_dz = np.where(z > 0, 0, np.log(alpha))
        log_det = log_dx_dz.sum(axis=-1)

        if(log_det.ndim > 1):
            # Then we have an image and have to sum over the height and width axes
            log_det = log_det.sum(axis=(-2, -1))

        return log_pz + log_det, x, state

    return init_fun, forward, inverse

################################################################################################################

def Sigmoid(lmbda=None, name='unnamed'):
    # language=rst
    """
    Invertible sigmoid.  The logit function is its inverse.  Remember to apply sigmoid before logit so that
    the input ranges are as expected!

    :param lmbda: For numerical stability
    """
    safe = lmbda is not None
    def init_fun(key, input_shape, condition_shape):
        params, state = (), ()
        return name, input_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        z = jax.nn.sigmoid(x)
        log_det = -(jax.nn.softplus(x) + jax.nn.softplus(-x))

        if(safe == True):
            z -= lmbda
            z /= 1.0 - 2*lmbda
            log_det -= np.log(1.0 - 2*lmbda)

        log_det = log_det.sum(axis=-1)

        if(log_det.ndim > 1):
            # Then we have an image and have to sum over the height and width axes
            log_det = log_det.sum(axis=(-2, -1))

        return log_px + log_det, z, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        if(safe == True):
            z *= 1.0 - 2*lmbda
            z += lmbda

        x = jax.scipy.special.logit(z)
        log_det = -(jax.nn.softplus(x) + jax.nn.softplus(-x)).sum(axis=-1)

        if(safe == True):
            log_det -= np.log(1.0 - 2*lmbda)

        log_det = log_det.sum(axis=-1)

        if(log_det.ndim > 1):
            # Then we have an image and have to sum over the height and width axes
            log_det = log_det.sum(axis=(-2, -1))

        return log_pz + log_det, x, state

    return init_fun, forward, inverse

def Logit(lmbda=0.05, name='unnamed'):
    # language=rst
    """
    Inverse of Sigmoid

    :param lmbda: For numerical stability
    """
    safe = lmbda is not None
    def init_fun(key, input_shape, condition_shape):
        params, state = (), ()
        return name, input_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):

        if(safe == True):
            x *= (1.0 - 2*lmbda)
            x += lmbda

        z = jax.scipy.special.logit(x)
        log_det = (jax.nn.softplus(z) + jax.nn.softplus(-z))

        if(safe == True):
            log_det += np.log(1.0 - 2*lmbda)

        log_det = log_det.sum(axis=-1)
        if(log_det.ndim > 1):
            # Then we have an image and have to sum more
            log_det = log_det.sum(axis=(-2, -1))
        return log_px + log_det, z, state

    def inverse(params, state, log_pz, z, condition, **kwargs):

        x = jax.nn.sigmoid(z)
        log_det = (jax.nn.softplus(z) + jax.nn.softplus(-z))

        if(safe == True):
            x -= lmbda
            x /= (1.0 - 2*lmbda)
            log_det += np.log(1.0 - 2*lmbda)

        log_det = log_det.sum(axis=-1)
        if(log_det.ndim > 1):
            # Then we have an image and have to sum more
            log_det = log_det.sum(axis=(-2, -1))

        return log_pz + log_det, x, state

    return init_fun, forward, inverse

################################################################################################################

def GaussianMixtureCDF(n_components=4, weight_logits_init=normal(), mean_init=normal(), variance_init=ones, name='unnamed'):
    # language=rst
    """
    Inverse transform sampling of a Gaussian Mixture Model.  CDF(x|pi,mus,sigmas) = sum[pi_i*erf(x|mu, sigma)]

    :param n_components: The number of components to use in the GMM
    """
    def init_fun(key, input_shape, condition_shape):
        k1, k2, k3 = random.split(key, 3)
        weight_logits = weight_logits_init(k1, (n_components,))
        means = mean_init(k2, (n_components,))
        variances = variance_init(k3, (n_components,))
        params = (weight_logits, means, variances)
        state = ()
        return name, input_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        weights, means, variances = params

        # z is the CDF of x
        dxs = x[...,None] - means[...,:]
        cdfs = 0.5*(1 + jax.scipy.special.erf(dxs/np.sqrt(2*variances[...,:])))
        z = np.sum(np.exp(weights)*cdfs, axis=-1)

        # log_det is log_pdf(x)
        log_pdfs = -0.5*(dxs**2)/variances[...,:] - 0.5*np.log(variances[...,:]) - 0.5*np.log(2*np.pi)
        log_det = logsumexp(weight_logits + log_pdfs, axis=-1)

        # We computed the element-wise log_dets, so sum over the dimension axis
        log_det = log_det.sum(axis=-1)

        return log_px + log_det, z, state

    def inverse(params, state, log_px, x, condition, **kwargs):
        # TODO: Implement iterative method to do this
        assert 0, 'Not implemented'

    return init_fun, forward, inverse

################################################################################################################

def MAF(hidden_layer_sizes,
        reverse=False,
        method='sequential',
        key=None,
        name='unnamed',
        **kwargs):
    # language=rst
    """
    Masked Autoregressive Flow https://arxiv.org/pdf/1705.07057.pdf
    Invertible network that enforces autoregressive property.

    :param hidden_layer_sizes: A list of the size of the feature network
    :param reverse: Whether or not to reverse the inputs
    :param method: Either 'sequential' or 'random'.  Controls how indices are assigned to nodes in each layer
    :param key: JAX random key.  Only needed in random mode
    """

    made_apply_fun = None

    def init_fun(key, input_shape, condition_shape):
        # Ugly, but saves us from initializing when calling function
        nonlocal made_apply_fun
        made_init_fun, made_apply_fun = spp.GaussianMADE(input_shape[-1], hidden_layer_sizes, reverse=reverse, method=method, key=key, name=name, **kwargs)
        _, (mu_shape, alpha_shape), params, state = made_init_fun(key, input_shape)
        return name, input_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        (mu, alpha), updated_state = made_apply_fun(params, state, x, **kwargs)
        z = (x - mu)*np.exp(-alpha)
        log_det = -alpha.sum(axis=-1)
        return log_px + log_det, z, updated_state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        # TODO: Speed this up with lax.while and fixed point iteration (and add vjps for each)
        x = np.zeros_like(z)
        u = z

        # Helper for item assignment
        idx_axis_0 = np.arange(x.shape[0])

        # For each MADE block, need to build output a dimension at a time
        for i in range(1, 1 + z.shape[-1]):
            (mu, alpha), _ = made_apply_fun(params, state, u, **kwargs)
            w = mu + u*np.exp(alpha)
            x = index_update(x, [idx_axis_0, np.ones_like(idx_axis_0)*(i - 1)], w[:,(i - 1)])

        # Do the backwards pass again to get the determinant term
        # Not sure if this is the correct way to update the state though
        (mu, alpha), updated_state = made_apply_fun(params, state, x, **kwargs)
        log_det = -alpha.sum(axis=-1)

        return log_pz + log_det, x, updated_state

    return init_fun, forward, inverse

################################################################################################################

def MaskedAffineCoupling(transform_fun, axis=-1, mask_type='channel_wise', top_left_zero=False, name='unnamed'):
    # language=rst
    """
    Apply an arbitrary transform to half of the input vector.  Uses masking instead of explicitly splitting inputs

    :param transform: A transformation that will act on half of the input vector. Must return 2 vectors!!!
    :param axis: Axis to split on
    :param mask_type: What kind of masking to use.  For images, can use checkerboard
    :param top_left_zero: Whether or not top left pixel should be 0.  Basically like the Reverse layer
    """
    apply_fun = None
    reduce_axes = None
    mask = None

    def init_fun(key, input_shape, condition_shape):
        ax = axis % len(input_shape)

        # We need to keep track of the input shape in order to know how to reduce the log det
        nonlocal reduce_axes
        reduce_axes = tuple(range(-1, -(len(input_shape) + 1), -1))

        # Generate the mask we'll use for training
        nonlocal mask
        if(mask_type == 'channel_wise'):
            mask_index = [slice(0, s//2) if i == ax else slice(None) for i, s in enumerate(input_shape)]
            mask_index = tuple(mask_index)
            mask = onp.ones(input_shape)
            mask[mask_index] = 0.0
        elif(mask_type == 'checkerboard'):
            assert len(input_shape) == 3
            height, width, channel = input_shape

            # Mask should be the same shape as the input
            mask = onp.ones(input_shape)

            # Build the checkerboard mask and fill in the mask
            masks, _, _ = checkerboard_masks(2, (height, width))
            if(top_left_zero == False):
                mask[:,:] = masks[0][:,:,None]
            else:
                mask[:,:] = masks[1][:,:,None]
        else:
            assert 0, 'Invalid mask type'

        # Ugly, but saves us from initializing when calling function
        nonlocal apply_fun
        transform_input_shape = input_shape if len(condition_shape) == 0 else (input_shape,) + condition_shape
        _init_fun, apply_fun = transform_fun(out_shape=input_shape)
        _, (log_s_shape, t_shape), transform_params, state = _init_fun(key, transform_input_shape)

        scale = 0.0
        shift = 0.0
        params = (transform_params, scale, shift)

        return name, input_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        transform_params, scale, shift = params

        # Apply the nonlinearity to the masked input
        masked_input = (1.0 - mask)*x
        network_input = masked_input if len(condition) == 0 else (masked_input, *condition)
        (log_s, t), updated_state = apply_fun(transform_params, state, network_input, **kwargs)

        # To control the variance
        log_s = scale*log_s + shift

        # Remask the result and compute the output
        log_s, t = mask*log_s, mask*t
        z = x*np.exp(log_s) + t

        log_det = np.sum(log_s, axis=reduce_axes)

        return log_px + log_det, z, updated_state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        transform_params, scale, shift = params

        # Apply the nonlinearity to the masked input
        masked_input = (1.0 - mask)*z
        network_input = masked_input if len(condition) == 0 else (masked_input, *condition)
        (log_s, t), updated_state = apply_fun(transform_params, state, network_input, **kwargs)

        # To control the variance
        log_s = scale*log_s + shift

        # Remask the result and compute the output
        log_s, t = mask*log_s, mask*t
        x = (z - t)*np.exp(-log_s)

        log_det = np.sum(log_s, axis=reduce_axes)

        return log_pz + log_det, x, updated_state

    return init_fun, forward, inverse

################################################################################################################

def AffineCoupling(transform_fun, axis=-1, name='unnamed'):
    # language=rst
    """
    Apply an arbitrary transform to half of the input vector.  Probably slower than masked version, but is
    more memory efficient.

    :param transform: A transformation that will act on half of the input vector. Must return 2 vectors!!!
    :param axis: Axis to split on
    """
    apply_fun = None
    reduce_axes = None

    def init_fun(key, input_shape, condition_shape):
        ax = axis % len(input_shape)
        assert input_shape[-1]%2 == 0
        half_split_dim = input_shape[ax]//2

        # We need to keep track of the input shape in order to know how to reduce the log det
        nonlocal reduce_axes
        reduce_axes = tuple(range(-1, -(len(input_shape) + 1), -1))

        # Find the split shape
        split_input_shape = input_shape[:ax] + (half_split_dim,) + input_shape[ax + 1:]
        transform_input_shape = split_input_shape if len(condition_shape) == 0 else (split_input_shape,) + condition_shape

        # Ugly, but saves us from initializing when calling function
        nonlocal apply_fun
        _init_fun, apply_fun = transform_fun(out_shape=split_input_shape)
        name, (log_s_shape, t_shape), params, state = _init_fun(key, transform_input_shape)

        return name, input_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):

        xa, xb = np.split(x, 2, axis=axis)
        network_input = xb if len(condition) == 0 else (xb, *condition)
        (log_s, t), updated_state = apply_fun(params, state, xb, **kwargs)

        za = xa*np.exp(log_s) + t
        z = np.concatenate([za, xb], axis=axis)

        log_det = np.sum(log_s, axis=reduce_axes)

        return log_px + log_det, z, updated_state

    def inverse(params, state, log_pz, z, condition, **kwargs):

        za, zb = np.split(z, 2, axis=axis)
        network_input = zb if len(condition) == 0 else (zb, *condition)
        (log_s, t), updated_state = apply_fun(params, state, zb, **kwargs)

        xa = (za - t)*np.exp(-log_s)
        x = np.concatenate([xa, zb], axis=axis)

        log_det = np.sum(log_s, axis=reduce_axes)

        return log_pz + log_det, x, updated_state

    return init_fun, forward, inverse

################################################################################################################

def GLOWBlock(transform_fun, mask_type='channel_wise', top_left_zero=False, name='unnamed'):
    # language=rst
    """
    One step of GLOW https://arxiv.org/pdf/1807.03039.pdf

    :param transform: A transformation that will act on half of the input vector. Must return 2 vectors!!!
    :param mask_type: What kind of masking to use.  For images, can use checkerboard
    """
    return sequential_flow(ActNorm(name='%s_act_norm'%name),
                           OnebyOneConv(name='%s_one_by_one_conv'%name),
                           MaskedAffineCoupling(transform_fun, mask_type=mask_type, top_left_zero=top_left_zero, name='%s_affine_coupling'%name))

################################################################################################################

fft_channel_vmap = vmap(np.fft.fftn, in_axes=(2,), out_axes=2)
ifft_channel_vmap = vmap(np.fft.ifftn, in_axes=(2,), out_axes=2)
fft_double_channel_vmap = vmap(fft_channel_vmap, in_axes=(2,), out_axes=2)

inv_height_vmap = vmap(np.linalg.inv)
inv_height_width_vmap = vmap(inv_height_vmap)

# slogdet_height_width_vmap = vmap(vmap(lambda x: np.linalg.slogdet(x)[1]))
def complex_slogdet(x):
    D = np.block([[x.real, -x.imag], [x.imag, x.real]])
    return 0.25*np.linalg.slogdet(D@D.T)[1]
slogdet_height_width_vmap = vmap(vmap(complex_slogdet))

def CircularConv(filter_size, kernel_init=glorot_normal(), name='unnamed'):
    # language=rst
    """
    Invertible circular convolution

    :param filter_size: (height, width) of kernel
    """
    def init_fun(key, input_shape, condition_shape):
        height, width, channel = input_shape
        kernel = kernel_init(key, filter_size + (channel, channel))
        assert filter_size[0] <= height, 'filter_size: %s, input_shape: %s'%(filter_size, input_shape)
        assert filter_size[1] <= width, 'filter_size: %s, input_shape: %s'%(filter_size, input_shape)
        params = (kernel,)
        state = ()
        return name, input_shape, params, state

    def forward(params, state, log_px, x, condition, **kwargs):
        if(x.ndim == 4):
            return vmap(partial(forward, params, state, **kwargs), in_axes=(0, 0, None))(log_px, x, condition)

        kernel, = params

        # http://developer.download.nvidia.com/compute/cuda/2_2/sdk/website/projects/convolutionFFT2D/doc/convolutionFFT2D.pdf
        x_h, x_w, x_c = x.shape
        kernel_h, kernel_w, kernel_c_out, kernel_c_in = kernel.shape

        # See how much we need to roll the kernel
        kernel_x = (kernel_h - 1) // 2
        kernel_y = (kernel_w - 1) // 2

        # Pad the kernel to match the fft size and roll it so that its center is at (0,0)
        kernel_padded = np.pad(kernel[::-1,::-1,:,:], ((0, x_h - kernel_h), (0, x_w - kernel_w), (0, 0), (0, 0)))
        kernel_padded = np.roll(kernel_padded, (-kernel_x, -kernel_y), axis=(0, 1))

        # Apply the FFT to get the convolution
        image_fft = fft_channel_vmap(x)
        kernel_fft = fft_double_channel_vmap(kernel_padded)
        z_fft = np.einsum('abij,abj->abi', kernel_fft, image_fft)
        z = ifft_channel_vmap(z_fft).real

        # The log determinant is the log det of the frequencies over the channel dims
        log_det = slogdet_height_width_vmap(kernel_fft).sum()

        return log_px + log_det, z, state

    def inverse(params, state, log_pz, z, condition, **kwargs):
        if(z.ndim == 4):
            return vmap(partial(inverse, params, state, **kwargs), in_axes=(0, 0, None))(log_pz, z, condition)

        kernel, = params

        z_h, z_w, z_c = z.shape
        kernel_h, kernel_w, kernel_c_out, kernel_c_in = kernel.shape

        # See how much we need to roll the kernel
        kernel_x = (kernel_h - 1) // 2
        kernel_y = (kernel_w - 1) // 2

        # Pad the kernel to match the fft size and roll it so that its center is at (0,0)
        kernel_padded = np.pad(kernel[::-1,::-1,:,:], ((0, z_h - kernel_h), (0, z_w - kernel_w), (0, 0), (0, 0)))
        kernel_padded = np.roll(kernel_padded, (-kernel_x, -kernel_y), axis=(0, 1))

        # Apply the FFT to get the convolution
        image_fft = fft_channel_vmap(z)
        kernel_fft = fft_double_channel_vmap(kernel_padded)

        # For deconv, we need to invert the kernel over the channel dims
        kernel_fft_inv = inv_height_width_vmap(kernel_fft)

        x_fft = np.einsum('abij,abj->abi', kernel_fft_inv, image_fft)
        x = ifft_channel_vmap(x_fft).real

        # The log determinant is the log det of the frequencies over the channel dims
        log_det = slogdet_height_width_vmap(kernel_fft).sum()

        return log_pz + log_det, x, state

    return init_fun, forward, inverse

################################################################################################################

def flow_test(flow, x, key, **kwargs):
    # language=rst
    """
    Test if a flow implementation is correct.  Checks if the forward and inverse functions are consistent and
    compares the jacobian determinant calculation against an autograd calculation.
    Assumes that there is no prior on the flow!

    :param flow: A normalizing flow
    :param x: A batched input
    :param key: JAX random key
    """
    # Initialize the flow with conditioning.
    init_fun, forward, inverse = flow

    input_shape = x.shape[1:]
    # condition_shape = (input_shape,)
    # cond = (x,)
    condition_shape = ()
    cond = ()
    names, output_shape, params, state = init_fun(key, input_shape, condition_shape)

    # Make sure that the forwards and inverse functions are consistent
    log_px, z, updated_state = forward(params, state, np.zeros(x.shape[0]), x, cond, test=TEST, key=key, **kwargs)
    log_pfz, fz, updated_state = inverse(params, state, np.zeros(z.shape[0]), z, cond, test=TEST, key=key, **kwargs)

    x_diff = np.linalg.norm(x - fz)
    log_px_diff = np.linalg.norm(log_px - log_pfz)
    print('Transform consistency diffs: x_diff: %5.3f, log_px_diff: %5.3f'%(x_diff, log_px_diff))

    # We are assuming theres no prior!!!!!
    log_det = log_px

    # Make sure that the log det terms are correct
    def z_from_x(unflatten, x_flat, cond, **kwargs):
        x = unflatten(x_flat)
        z = forward(params, state, 0, x, cond, test=TEST, key=key, **kwargs)[1]
        return ravel_pytree(z)[0]

    def single_elt_logdet(x, cond, **kwargs):
        x_flat, unflatten = ravel_pytree(x)
        jac = jacobian(partial(z_from_x, unflatten, **kwargs))(x_flat, cond)
        return 0.5*np.linalg.slogdet(jac.T@jac)[1]

    actual_log_det = vmap(single_elt_logdet)(x, cond, **kwargs)

    print('actual_log_det', actual_log_det)
    print('log_det', log_det)

    log_det_diff = np.linalg.norm(log_det - actual_log_det)
    print('Log det diff: %5.3f'%(log_det_diff))
