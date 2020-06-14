import itertools
import functools
import numpy as onp
import jax
from jax import random, jit
import jax.nn
import jax.numpy as np
from functools import partial, reduce
from jax.tree_util import tree_flatten, tree_unflatten
from jax.experimental import stax
from jax.nn.initializers import glorot_normal, normal, ones, zeros
import jax.experimental.stax as stax
from jax.ops import index, index_add, index_update
from util import is_testing, TRAIN, TEST
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten

################################################################################################################

def sequential(*layers):
    # language=rst
    """
    Sequential network builder.  Like stax.serial, but passes names and state too.

    :param layers - An unpacked list of (init_fun, apply_fun)

    **Example**

    .. code-block:: python
        import jax.numpy as np
        from jax import random
        from staxplusplus import sequential, Dense, BatchNorm

        key = random.PRNGKey(0)

        input_shape = (5,)
        network = sequential(Dense(1024), BatchNorm(), Dense(1))
        init_fun, apply_fun = network
        names, output_shape, params, state = init_fun(key, input_shape)

        inputs = np.ones((10, 5))
        output, updated_state = apply_fun(params, state, inputs)
    """

    n_layers = len(layers)
    init_funs, apply_funs = zip(*layers)

    def init_fun(key, input_shape):
        names, params, states = [], [], []
        for init_fun in init_funs:
            key, *keys = random.split(key, 2)
            name, input_shape, param, state = init_fun(keys[0], input_shape)
            names.append(name)
            params.append(param)
            states.append(state)
        return tuple(names), input_shape, tuple(params), tuple(states)

    def apply_fun(params, state, inputs, **kwargs):

        # Need to store the ouputs of the functions and the updated states
        updated_states = []

        # Need to pop so that we don't resuse random keys!
        key = kwargs.pop('key', None)
        keys = random.split(key, n_layers) if key is not None else (None,)*n_layers

        # Evaluate each function and store the updated states
        for fun, param, s, key in zip(apply_funs, params, state, keys):
            inputs, updated_state = fun(param, s, inputs, key=key, **kwargs)
            updated_states.append(updated_state)

        return inputs, tuple(updated_states)

    return init_fun, apply_fun

def parallel(*layers):
    # language=rst
    """
    Parallel network builder.  Like stax.parallel, but passes names and state too.

    :param layers - An unpacked list of (init_fun, apply_fun)

    **Example**

    .. code-block:: python

        from jax import random
        import staxplusplus as spp
        key = random.PRNGKey(0)

        input_shape = (5,)
        network = sequential(Dense(1024), BatchNorm(), Dense(5))
        residual_network = sequential(FanOut(2), parallel(Identity(), network), FanInSum())
        init_fun, apply_fun = residual_network
        names, output_shape, params, state = init_fun(key, input_shape)

        inputs = np.ones((10, 5))
        output = apply_fun(params, state, inputs)
    """

    n_layers = len(layers)
    init_funs, apply_funs = zip(*layers)

    def init_fun(key, input_shape):
        keys = random.split(key, n_layers)
        names, output_shapes, params, states = [], [], [], []
        # Split these up so that we can evaluate each of the parallel items together
        for init_fun, key, shape in zip(init_funs, keys, input_shape):
            name, output_shape, param, state = init_fun(key, shape)
            names.append(name)
            output_shapes.append(output_shape)
            params.append(param)
            states.append(state)
        return tuple(names), output_shapes, tuple(params), tuple(states)

    def apply_fun(params, state, inputs, **kwargs):
        # Need to pop so that we don't resuse random keys!
        key = kwargs.pop('key', None)
        keys = random.split(key, n_layers) if key is not None else (None,)*n_layers

        # We need to store each of the outputs and states
        outputs = []
        updated_states = []
        zipped_iterables = zip(apply_funs, params, state, inputs, keys)
        for apply_fun, param, s, inp, key in zipped_iterables:
            output, updated_state = apply_fun(param, s, inp, key=key, **kwargs)
            outputs.append(output)
            updated_states.append(updated_state)

        return outputs, tuple(updated_states)

    return init_fun, apply_fun

################################################################################################################

def stax_wrapper(fun):
    """ Convenience wrapper around existing stax functions """
    def ret(*args, name='unnamed', **kwargs):

        # Some stax layers don't need to be called
        if(isinstance(fun, tuple)):
            _init_fun, _apply_fun = fun
        else:
            _init_fun, _apply_fun = fun(*args, **kwargs)

        def init_fun(key, input_shape):
            output_shape, params = _init_fun(key, input_shape)
            state = ()
            return name, output_shape, params, state
        def apply_fun(params, state, inputs, **kwargs):
            return _apply_fun(params, inputs, **kwargs), state

        return init_fun, apply_fun

    return ret

Tanh = stax_wrapper(stax.Tanh)
Relu = stax_wrapper(stax.Relu)
Exp = stax_wrapper(stax.Exp)
LogSoftmax = stax_wrapper(stax.LogSoftmax)
Softmax = stax_wrapper(stax.Softmax)
Softplus = stax_wrapper(stax.Softplus)
Sigmoid = stax_wrapper(stax.Sigmoid)
Elu = stax_wrapper(stax.Elu)
LeakyRelu = stax_wrapper(stax.LeakyRelu)
Selu = stax_wrapper(stax.Selu)
Gelu = stax_wrapper(stax.Gelu)
Identity = stax_wrapper(stax.Identity)
FanInSum = stax_wrapper(stax.FanInSum)
FanOut = stax_wrapper(stax.FanOut)
FanInConcat = stax_wrapper(stax.FanInConcat)

################################################################################################################

def Split(num, axis=-1, name='unnamed'):
    # language=rst
    """
    Split an input along an axis

    :param num - Number of pieces to split into
    :param axis - Axis to split on
    """
    def init_fun(key, input_shape):
        ax = axis % len(input_shape)
        assert input_shape[-1]%num == 0
        split_dim = input_shape[ax]//num
        split_input_shape = input_shape[:ax] + (split_dim,) + input_shape[ax + 1:]
        output_shape = (split_input_shape,)*num
        params, states = (), ()
        return name, output_shape, params, states

    def apply_fun(params, state, inputs, **kwargs):
        return np.split(inputs, num, axis=axis), state

    return init_fun, apply_fun

################################################################################################################

def stax_conv_wrapper(fun):
    """ Convenience wrapper around existing stax functions that work on images """
    def ret(*args, name='unnamed', **kwargs):

        # Some stax layers don't need to be called
        if(isinstance(fun, tuple)):
            _init_fun, _apply_fun = fun
        else:
            _init_fun, _apply_fun = fun(*args, **kwargs)

        def init_fun(key, input_shape):
            # JAX conv is weird with batch dims
            assert len(input_shape) == 3
            input_shape = (1,) + input_shape
            output_shape, params = _init_fun(key, input_shape)
            output_shape = output_shape[1:]
            state = ()
            return name, output_shape, params, state

        def apply_fun(params, state, inputs, **kwargs):
            if(inputs.ndim == 3):
                ans = _apply_fun(params, inputs[None], **kwargs)[0]
            else:
                ans = _apply_fun(params, inputs, **kwargs)

            return ans, state

        return init_fun, apply_fun

    return ret

ConvTranspose = stax_conv_wrapper(stax.ConvTranspose)
MaxPool = stax_conv_wrapper(stax.MaxPool)
SumPool = stax_conv_wrapper(stax.SumPool)
AvgPool = stax_conv_wrapper(stax.AvgPool)

def GeneralConv(dimension_numbers, out_chan, filter_shape,
                strides=None, padding='VALID',
                bias=True, weightnorm=False,
                W_init=None, b_init=normal(1e-6), name='unnamed'):
    # language=rst
    """
    Like jax.stax.GeneralConv, but has the option to use a bias and weightnorm

    :param out_channel - Number of output channels
    :param filter_shape - Size of filter
    :param strides - Strides for each axis
    :param padding - Padding for each axis
    """
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * len(filter_shape)
    strides = strides or one
    W_init = W_init or glorot_normal(rhs_spec.index('I'), rhs_spec.index('O'))
    def init_fun(key, input_shape):
        # JAX conv is weird with batch dims
        assert len(input_shape) == 3
        input_shape = (1,) + input_shape

        filter_shape_iter = iter(filter_shape)

        kernel_shape = [out_chan if c == 'O' else input_shape[lhs_spec.index('C')] if c == 'I' else next(filter_shape_iter) for c in rhs_spec]
        output_shape = jax.lax.conv_general_shape_tuple(input_shape, kernel_shape, strides, padding, dimension_numbers)
        output_shape = output_shape[1:]

        bias_shape = [out_chan if c == 'C' else 1 for c in out_spec]
        bias_shape = tuple(itertools.dropwhile(lambda x: x == 1, bias_shape))

        k1, k2, k3 = random.split(key, 3)

        W = W_init(k1, kernel_shape)
        if(bias):
            b = b_init(k2, bias_shape)
        else:
            b = 0.0 # Just use a dummy

        if(weightnorm):
            # Weight norm is defined over each scalar element of the output
            g = np.ones(bias_shape)
        else:
            g = 0.0 # Just use a dummy

        params = (W, b, g)
        state = ()

        return name, output_shape, params, state

    def apply_fun(params, state, inputs, **kwargs):
        W, b, g = params

        if(weightnorm):
            # Apply weight normalization
            W = g*W/np.linalg.norm(W)

        batched = True
        if(inputs.ndim == 3):
            batched = False
            inputs = inputs[None]

        out = jax.lax.conv_general_dilated(inputs, W, strides, padding, one, one, dimension_numbers=dimension_numbers)

        if(batched == False):
            out = out[0]

        if(bias):
            out += params[-1]

        return out, state

    return init_fun, apply_fun
Conv = functools.partial(GeneralConv, ('NHWC', 'HWIO', 'NHWC'))

def data_dependent_init(x, target_param_names, name_tree, params, state, apply_fun, flag_names, **kwargs):
    # language=rst
    """
    Data dependent initialization.

    :param x: The data seed
    :param target_param_names: A list of the names of parameters to seed
    :param name_tree: A pytree (nested structure) of names.  This is the first output of an init_fun call
    :param params: The parameter pytree
    :param states: The states pytree
    :param apply_fun: Apply function
    :param flag_names: The names of the flag that will turn on seeding.

    **Example**

    .. code-block:: python
        from jax import random
        import jax.numpy as np
        from staxplusplus import WeightNormConv, data_dependent_init
        from util import TRAIN, TEST

        # Create the model
        model = WeightNormConv(4, (3, 3), padding='SAME', name='wn')

        # Initialize it
        init_fun, apply_fun = model
        key = random.PRNGKey(0)
        names, output_shape, params, state = init_fun(key, input_shape=(5, 5, 3))

        # Seed weight norm and retrieve the new parameters
        data_seed = np.ones((10, 5, 5, 3))
        weightnorm_names = ['wn']
        params = data_dependent_init(data_seed, weightnorm_names, names, params, state, apply_fun, 'weightnorm_seed')
    """
    if(len(target_param_names) == 0):
        return params

    # Can be None if it is already filled in with partial
    if(flag_names is not None):
        if(isinstance(flag_names, list) == False and isinstance(flag_names, tuple) == False):
            flag_names = (flag_names,)

        # Pass the seed name to the apply function
        for name in flag_names:
            kwargs[name] = True

    # Run the network.  The state should include the seeded parameters
    _, states_with_seed = apply_fun(params, state, x, **kwargs)

    # Replace the parameters with the seeded parameters
    for name in target_param_names:
        seeded_param = get_param(name, name_tree, states_with_seed)
        params = modify_param(name, name_tree, params, seeded_param)
    return params

################################################################################################################

def InstanceNorm(name='unnamed'):
    def init_fun(key, input_shape):
        params, state = (), ()
        return name, input_shape, params, state

    def apply_fun(params, state, inputs, **kwargs):
        x = inputs

        # Make sure that we have the expected shape
        H, W, C = x.shape[-3], x.shape[-2], x.shape[-1]
        mean = np.mean(x, axis=(-3, -2))
        std = np.std(x, axis=(-3, -2)) + 1e-5

        if(x.ndim == 4):
            x = (x - mean[:,None,None,:])/std[:,None,None,:]
        else:
            x = (x - mean[None,None,:])/std[None,None,:]

        return x, state

    return init_fun, apply_fun

def LayerNorm(name='unnamed'):
    def init_fun(key, input_shape):
        H, W, C = input_shape
        g = np.ones(C)
        b = np.zeros(C)
        params, state = (g, b), ()
        return name, input_shape, params, state

    def apply_fun(params, state, inputs, **kwargs):
        g, b = params
        x = inputs

        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)

        x = (x - mean)/np.sqrt(std + 1e-5)*g + b

        return x, state

    return init_fun, apply_fun

def LayerNormSimple(name='unnamed'):
    def init_fun(key, input_shape):
        H, W, C = input_shape
        params, state = (), ()
        return name, input_shape, params, state

    def apply_fun(params, state, inputs, **kwargs):
        x = inputs

        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)

        x = (x - mean)/np.sqrt(std + 1e-5)

        return x, state

    return init_fun, apply_fun

################################################################################################################

def Reshape(shape, name='unnamed'):
    # language=rst
    """
    Reshape an input

    :param shape - The shape to change an input to
    """

    total_dim = np.prod(shape)

    def init_fun(key, input_shape):
        assert np.prod(input_shape) == total_dim
        params = ()
        state = ()
        return name, shape, params, state

    def apply_fun(params, state, inputs, **kwargs):
        if(np.prod(inputs.shape) != total_dim):
            assert np.prod(inputs.shape) % total_dim == 0
            return np.reshape(inputs, (-1,) + shape), state
        return np.reshape(inputs, shape), state

    return init_fun, apply_fun

################################################################################################################

def ScalarMultiply(val, name='unnamed'):
    # language=rst
    """
    Multiply an input by a constant scalar value

    :param val - The constant value
    """
    def init_fun(key, input_shape):
        params = ()
        state = ()
        return name, input_shape, params, state

    def apply_fun(params, state, inputs, **kwargs):
        return inputs*val, state

    return init_fun, apply_fun

def Dense(out_dim, mask_id=None, keep_prob=1.0, W_init=glorot_normal(), b_init=normal(), name='unnamed'):
    # language=rst
    """
    Fully connected layer with dropout and an option to use a mask

    :param out_dim: The output dimension.  Input is filled automatically during initialization
    :param mask_id: A string that indexes into the kwargs to retrieve the mask
    :param keep_prob: The probability of keeping a weight in the matrix during train time
    """

    use_dropout = keep_prob < 0.99999

    def init_fun(key, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(key)
        W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
        params = (W, b)
        state = ()
        return name, output_shape, params, state

    def apply_fun(params, state, inputs, **kwargs):
        W, b = params

        # See if we are testing or training
        test = kwargs.get('test', TRAIN)

        # Dropout
        if(use_dropout and is_testing(test) == False):

            key = kwargs.get('key', None)
            if(key is None):
                assert 0, 'Need JAX random key for this!'

            keep = random.bernoulli(key, keep_prob, W.shape)
            W = np.where(keep, W / keep_prob, 0)

        # Mask W is needed
        if(mask_id is not None):
            mask = kwargs[mask_id]
            W = W*mask

        return np.dot(inputs, W) + b, state

    return init_fun, apply_fun

################################################################################################################

def BatchNorm(axis=0, epsilon=1e-5, alpha=0.05, beta_init=zeros, gamma_init=ones, name='unnamed'):
    # language=rst
    """
    Batch Normaliziation

    :param axis: Batch axis
    :param epsilon: Constant for numerical stability
    :param alpha: Parameter for exponential moving average of population parameters
    """

    def init_fun(key, input_shape):
        k1, k2 = random.split(key)
        beta, gamma = beta_init(k1, (input_shape[-1],)), gamma_init(k2, (input_shape[-1],))
        running_mean = np.zeros(input_shape)
        running_var = np.ones(input_shape)
        params = (beta, gamma)
        state = (running_mean, running_var)
        return name, input_shape, params, state

    @jit
    def get_bn_params(x, test, running_mean, running_var):
        """ Update the batch norm statistics """
        if(is_testing(test)):
            mean, var = running_mean, running_var
        else:
            mean = np.mean(x, axis=axis)
            var = np.var(x, axis=axis) + epsilon
            running_mean = (1 - alpha)*running_mean + alpha*mean
            running_var = (1 - alpha)*running_var + alpha*var

        return (mean, var), (running_mean, running_var)

    def apply_fun(params, state, inputs, **kwargs):
        beta, gamma = params
        running_mean, running_var = state
        x = inputs

        # Check if we're training or testing
        test = kwargs.get('test', TRAIN)

        # Update the running population parameters
        (mean, var), (running_mean, running_var) = get_bn_params(x, test, running_mean, running_var)

        # Normalize the inputs
        x_hat = (x - mean) / np.sqrt(var)
        z = gamma*x_hat + beta

        updated_state = (running_mean, running_var)
        return z, updated_state

    return init_fun, apply_fun

################################################################################################################

def LowDimInputConvBlock(n_channels=512, init_zeros=True, name='unnamed'):
    # language=rst
    """
    A conv block where we assume the number of input channels and output channels are small
    """
    _apply_fun = None

    def init_fun(key, input_shape):
        H, W, C = input_shape
        if(init_zeros):
            W_init, b_init = zeros, zeros
        else:
            W_init, b_init = glorot_normal(), normal()

        nonlocal _apply_fun
        _init_fun, _apply_fun = sequential(Conv(n_channels, filter_shape=(3, 3), padding=((1, 1), (1, 1)), bias=True, weightnorm=False),
                                           LayerNormSimple(),
                                           Relu(),
                                           Conv(n_channels, filter_shape=(1, 1), padding=((0, 0), (0, 0)), bias=True, weightnorm=False),
                                           LayerNormSimple(),
                                           Relu(),
                                           Conv(C, filter_shape=(3, 3), padding=((1, 1), (1, 1)), bias=True, weightnorm=False, W_init=W_init, b_init=b_init))
        name, output_shape, params, state = _init_fun(key, input_shape)
        return name, output_shape, params, state

    def apply_fun(params, state, inputs, **kwargs):
        return _apply_fun(params, state, inputs, **kwargs)

    return init_fun, apply_fun

def DoubledLowDimInputConvBlock(n_channels=512, init_zeros=True, name='unnamed'):
    # language=rst
    """
    A conv block where we assume the number of input channels and output channels are small
    """
    _apply_fun = None

    def init_fun(key, input_shape):
        H, W, C = input_shape
        if(init_zeros):
            W_init, b_init = zeros, zeros
        else:
            W_init, b_init = glorot_normal(), normal()

        nonlocal _apply_fun
        _init_fun, _apply_fun = sequential(Conv(n_channels, filter_shape=(3, 3), padding=((1, 1), (1, 1)), bias=True, weightnorm=False),
                                           LayerNormSimple(),
                                           Relu(),
                                           Conv(n_channels, filter_shape=(1, 1), padding=((0, 0), (0, 0)), bias=True, weightnorm=False),
                                           LayerNormSimple(),
                                           Relu(),
                                           Conv(2*C, filter_shape=(3, 3), padding=((1, 1), (1, 1)), bias=True, weightnorm=False, W_init=W_init, b_init=b_init),
                                           Split(2, axis=-1),
                                           parallel(Tanh(), Identity()))
        name, output_shape, params, state = _init_fun(key, input_shape)
        return name, output_shape, params, state

    def apply_fun(params, state, inputs, **kwargs):
        return _apply_fun(params, state, inputs, **kwargs)

    return init_fun, apply_fun

################################################################################################################

def SqueezeExcitation(ratio=2, W1_init=glorot_normal(), W2_init=glorot_normal(), name='unnamed'):
    # language=rst
    """
    https://arxiv.org/pdf/1709.01507.pdf

    :param ratio: How to reduce the number of channels for the FC layer
    """
    def init_fun(key, input_shape):
        H, W, C = input_shape
        assert C%ratio == 0
        k1, k2 = random.split(key, 2)
        W1 = W1_init(k1, (C//ratio, C))
        W2 = W2_init(k2, (C, C//ratio))
        output_shape = input_shape
        params = (W1, W2)
        state = ()
        return name, output_shape, params, state

    def apply_fun(params, state, inputs, **kwargs):
        W1, W2 = params

        # Apply the SE transforms
        x = np.mean(inputs, axis=(-2, -3))
        x = np.dot(x, W1.T)
        x = jax.nn.relu(x)
        x = np.dot(x, W2.T)
        x = jax.nn.sigmoid(x)

        # Scale the input
        if(x.ndim == 3):
            out = inputs*x[None, None,:]
        else:
            out = inputs*x[:,None,None,:]
        return out, state

    return init_fun, apply_fun

################################################################################################################

def ConditionedSqueezeExcitation(ratio=4, W_cond_init=glorot_normal(), W1_init=glorot_normal(), W2_init=glorot_normal(), name='unnamed'):
    # language=rst
    """
    Like squeeze excitation, but has an extra input to help form W
    PURPOSE IS TO FIGURE OUT WHICH FEATURE MAPS MATTER GIVEN A CONDITIONER

    :param ratio: How to reduce the number of channels for the FC layer
    """
    def init_fun(key, input_shape):
        (H, W, C), (K,) = input_shape
        k1, k2, k3 = random.split(key, 3)

        # Will be shrinking the conditioner down to the size of the number of channels
        W_cond = W_cond_init(k1, (C, K))

        # Going to be concatenating the conditioner
        C_concat = C + C
        assert C_concat%ratio == 0

        # Create the parameters for the squeeze and excite
        W1 = W1_init(k2, (C_concat//ratio, C_concat))
        W2 = W2_init(k3, (C, C_concat//ratio))

        output_shape = (H, W, C)
        params = (W_cond, W1, W2)
        state = ()
        return name, output_shape, params, state

    def apply_fun(params, state, inputs, **kwargs):
        W_cond, W1, W2 = params
        inputs, cond = inputs

        # Apply the SE transforms
        x = np.mean(inputs, axis=(-2, -3))
        x = np.concatenate([x, np.dot(cond, W_cond.T)], axis=-1)
        x = np.dot(x, W1.T)
        x = jax.nn.relu(x)
        x = np.dot(x, W2.T)
        x = jax.nn.sigmoid(x)

        # Scale the input
        if(x.ndim == 3):
            out = inputs*x[None, None,:]
        else:
            out = inputs*x[:,None,None,:]
        return out, state

    return init_fun, apply_fun

################################################################################################################

def Residual(network, name='unnamed'):
    # language=rst
    """
    Create a residual layer for a given network

    :param network: Input network that is a tuple (init_fun, apply_fun)
    """
    _init_fun, _apply_fun = network

    def init_fun(key, input_shape):
        name, output_shape, params, state = _init_fun(key, input_shape)
        # We're adding the input and output, so need to preserve shape
        assert output_shape == input_shape, 'Output shape is %s and input shape is %s'%(str(output_shape), str(input_shape))
        return name, input_shape, params, state

    def apply_fun(params, state, inputs, **kwargs):
        outputs, updated_state = _apply_fun(params, state, inputs)
        return inputs + outputs, updated_state

    return init_fun, apply_fun

################################################################################################################

def build_autoregressive_masks(dim,
                               hidden_layer_sizes,
                               reverse=False,
                               method='sequential',
                               key=None):
    # language=rst
    """
    Build masks for each weight in a neural network so that an application of the network is autoregressive.
    See MADE paper (https://arxiv.org/pdf/1502.03509.pdf) for details.

    :param dim: The dimension of the input
    :param hidden_layer_sizes: A list of the size of the feature network
    :param reverse: Whether or not to reverse the inputs
    :param method: Either 'sequential' or 'random'.  Controls how indices are assigned to nodes in each layer
    :param key: JAX random key.  Only needed in random mode
    """
    layer_sizes = hidden_layer_sizes + [dim]

    # We can either assign indices randomly or sequentially.  FOR LOW DIMENSIONS USE SEQUENTIAL!!!
    if(method == 'random'):
        assert key is not None
        keys = random.split(key, len(layer_sizes) + 1)
        key_iter = iter(keys)
        input_sel = random.randint(next(key_iter), shape=(dim,), minval=1, maxval=dim+1)
    else:
        # Alternate direction in consecutive layers
        input_sel = np.arange(1, dim + 1)
        if(reverse):
            input_sel = input_sel[::-1]

    # Build the hidden layer masks
    masks = []
    sel = input_sel
    prev_sel = sel
    for size in layer_sizes:

        # Choose the degrees of the next layer
        if(method == 'random'):
            sel = random.randint(next(key_iter), shape=(size,), minval=min(np.min(sel), dim - 1), maxval=dim)
        else:
            sel = np.arange(size)%max(1, dim - 1) + min(1, dim - 1)

        # Create the new mask
        mask = (prev_sel[:,None] <= sel).astype(np.int32)
        prev_sel = sel
        masks.append(mask)

    # Build the mask for the matrix between the input and output
    skip_mask = (input_sel[:,None] < input_sel).astype(np.int32)

    # Build the output layers.  Remember that we need to output mu and sigma.  Just need
    # a triangular matrix for the masks
    out_mask = (prev_sel[:,None] < input_sel).astype(np.int32)

    # Load the masks into a dictionary
    mask_kwargs = dict([('mask_%d'%(j), mask) for j, mask in enumerate(masks)])
    mask_kwargs['skip'] = skip_mask
    mask_kwargs['mu'] = out_mask
    mask_kwargs['alpha'] = out_mask

    return mask_kwargs

def GaussianMADE(dim,
                 hidden_layer_sizes,
                 reverse=False,
                 method='sequential',
                 key=None,
                 name='unnamed',
                 **kwargs):
    # language=rst
    """
    Gaussian MADE https://arxiv.org/pdf/1502.03509.pdf
    Network that enforces autoregressive property.

    :param dim: The dimension of the input
    :param hidden_layer_sizes: A list of the size of the feature network
    :param reverse: Whether or not to reverse the inputs
    :param method: Either 'sequential' or 'random'.  Controls how indices are assigned to nodes in each layer
    :param key: JAX random key.  Only needed in random mode
    """
    layer_sizes = hidden_layer_sizes + [dim]

    ############################## Build the network ##############################

    # Build the weights for the hidden layers
    dense_layers = []
    for j, size in enumerate(layer_sizes):
        dense_layers.append(Dense(size, mask_id='mask_%d'%(j), **kwargs))
        # dense_layers.append(BatchNorm()),
        dense_layers.append(Relu())
    hidden_path = sequential(*dense_layers)

    # Build the mean and log std weights
    mu_out = Dense(dim, mask_id='mu')
    alpha_out = sequential(Dense(dim, mask_id='alpha', **kwargs), Tanh()) # Bound alpha to avoid numerical instability

    # Create the layers of the network
    param_architecture = sequential(hidden_path,
                                    FanOut(2),
                                    parallel(mu_out, alpha_out))

    ############################## Build the masks for the network ##############################

    mask_kwargs = build_autoregressive_masks(dim, hidden_layer_sizes, reverse, method, key)

    # Fill the network application with the mask kwargs so the user doesn't have to
    init_params, network = param_architecture
    network = partial(network, **mask_kwargs)

    def init_fun(key, input_shape):
        x_shape = input_shape
        name, out_shape, params, state = init_params(key, x_shape)
        (mu_shape, alpha_shape) = out_shape
        return name, out_shape, params, state

    def apply_fun(params, state, inputs, **kwargs):
        (mu, alpha), updated_state = network(params, state, inputs, **kwargs)
        return (mu, alpha), updated_state

    return init_fun, apply_fun

################################################################################################################

def get_param(name, names, params):
    # language=rst
    """
    Retrieve a named parameter.  The names pytree should be the same as the params pytree.  We use the
    index of name in the flattened names in order to find the correct parameter in flattened params.

    :param name: Name of the parameter
    :param names: A pytree (nested structure) of names
    :param params: The parameter pytree
    """
    flat_names, treedef = tree_flatten(names)
    mapped_params = treedef.flatten_up_to(params)
    return mapped_params[flat_names.index(name)]

def modify_param(name, names, params, new_param):
    # language=rst
    """
    Change a named parameter.  name and params must have same pytree structure.

    :param name: Name of the parameter
    :param names: A pytree (nested structure) of names
    :param params: The parameter pytree
    :param new_param: The new value of the parameter associated with name.
    """

    flat_names, treedef = tree_flatten(names)
    mapped_params = treedef.flatten_up_to(params)
    old_param = mapped_params[flat_names.index(name)]

    # Make sure that the parameters are the same shape
    _, old_treedef = tree_flatten(old_param)
    _, new_treedef = tree_flatten(new_param)
    assert old_treedef == new_treedef, 'new_param has the wrong structure.  Got %s, expected %s'%(str(new_treedef), str(old_treedef))

    # Replace the parameter
    mapped_params[flat_names.index(name)] = new_param
    return treedef.unflatten(mapped_params)

################################################################################################################
