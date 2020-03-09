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

def get_param( name, names, params ):
    """ Retrieve a named parameter.  The names tree should be the same as the params tree, so we can use
        that to find the index of the target parameter

        Args:
            name   - Name of the parameter
            names  - A pytree (nested structure) of names
            params - The parameter pytree

        Returns:
            param - The parameters corresponding to name
    """
    flat_names, treedef = tree_flatten( names )
    mapped_params = treedef.flatten_up_to( params )
    return mapped_params[flat_names.index( name )]

def modify_param( name, names, params, new_param ):
    """ Change a named parameter

        Args:
            name      - Name of the parameter
            names     - A pytree (nested structure) of names
            params    - The parameter pytree
            new_param - The new value of the parameter associated with name.
                        Must have the same treedef as the old parameter
    """
    flat_names, treedef = tree_flatten( names )
    mapped_params = treedef.flatten_up_to( params )
    old_param = mapped_params[flat_names.index( name )]

    # Make sure that the parameters are the same shape
    _, old_treedef = tree_flatten( old_param )
    _, new_treedef = tree_flatten( new_param )
    assert old_treedef == new_treedef, 'new_param has the wrong structure.  Got %s, expected %s'%( str( new_treedef ), str( old_treedef ) )

    # Replace the parameter
    mapped_params[flat_names.index( name )] = new_param
    return treedef.unflatten( mapped_params )

################################################################################################################

def sequential( *layers ):
    """ Sequential network builder.  Like stax.sequential, but passes static_params and names.
        static_params will be non-trainable parameters (like batch norm parameters or Bayesian
        SVI parameters)

        Args:
            *layers - An unpacked list of (init_fun, apply_fun)
    """

    n_layers = len( layers )
    init_funs, apply_funs = zip( *layers )

    def init_fun( key, input_shape ):
        names, params, static_params = [], [], []
        for init_fun in init_funs:
            key, *keys = random.split( key, 2 )
            name, input_shape, param, static_param = init_fun( keys[0], input_shape )
            names.append( name )
            params.append( param )
            static_params.append( static_param )
        return tuple( names ), input_shape, tuple( params ), tuple( static_params )

    def apply_fun( params, static_params, inputs, **kwargs ):

        # Need to store the ouputs of the functions and the updated static params
        updated_static_params = []

        # Need to pop so that we don't resuse random keys!
        key = kwargs.pop( 'key', None )
        keys = random.split( key, n_layers ) if key is not None else ( None, )*n_layers

        # Evaluate each function and store the updated static parameters
        for fun, param, static_param, key in zip( apply_funs, params, static_params, keys ):
            inputs, updated_static_param = fun( param, static_param, inputs, key=key, **kwargs )
            updated_static_params.append( updated_static_param )

        return inputs, updated_static_params

    return init_fun, apply_fun

def parallel( *layers ):
    """ parallel:stax.parallel = sequential:stax.serial
        Usually use FanOut( K ), parallel( K items ), FanIn<Sum/Concat>() together

        Args:
            *layers - An unpacked list of (init_fun, apply_fun)
    """

    n_layers = len( layers )
    init_funs, apply_funs = zip( *layers )

    def init_fun( key, input_shape ):
        keys = random.split( key, n_layers )
        names, output_shapes, params, static_params = [], [], [], []
        # Split these up so that we can evaluate each of the parallel items together
        for init_fun, key, shape in zip( init_funs, keys, input_shape ):
            name, output_shape, param, static_param = init_fun( key, shape )
            names.append( name )
            output_shapes.append( output_shape )
            params.append( param )
            static_params.append( static_param )
        return names, output_shapes, params, static_params

    def apply_fun( params, static_params, inputs, **kwargs ):

        # Need to pop so that we don't resuse random keys!
        key = kwargs.pop( 'key', None )
        keys = random.split( key, n_layers ) if key is not None else ( None, )*n_layers

        # We need to store each of the outputs and static params
        outputs = []
        updated_static_params = []
        zipped_iterables = zip( apply_funs, params, static_params, inputs, keys )
        for apply_fun, param, static_param, inp, key in zipped_iterables:
            output, updated_static_param = apply_fun( param, static_param, inp, key=key, **kwargs )
            outputs.append( output )
            updated_static_params.append( updated_static_param )

        return outputs, updated_static_params

    return init_fun, apply_fun

################################################################################################################

def stax_wrapper( fun ):
    def ret( *args, name='unnamed', **kwargs ):

        # Some stax layers don't need to be called
        if( isinstance( fun, tuple ) ):
            _init_fun, _apply_fun = fun
        else:
            _init_fun, _apply_fun = fun( *args, **kwargs )

        def init_fun( key, input_shape ):
            output_shape, params = _init_fun( key, input_shape )
            return name, output_shape, params, ()
        def apply_fun( params, static_params, inputs, **kwargs ):
            return _apply_fun( params, inputs, **kwargs ), ()

        return init_fun, apply_fun

    return ret

Tanh = stax_wrapper( stax.Tanh )
Relu = stax_wrapper( stax.Relu )
Exp = stax_wrapper( stax.Exp )
LogSoftmax = stax_wrapper( stax.LogSoftmax )
Softmax = stax_wrapper( stax.Softmax )
Softplus = stax_wrapper( stax.Softplus )
Sigmoid = stax_wrapper( stax.Sigmoid )
Elu = stax_wrapper( stax.Elu )
LeakyRelu = stax_wrapper( stax.LeakyRelu )
Selu = stax_wrapper( stax.Selu )
Gelu = stax_wrapper( stax.Gelu )
Identity = stax_wrapper( stax.Identity )
FanInSum = stax_wrapper( stax.FanInSum )
FanOut = stax_wrapper( stax.FanOut )
FanInConcat = stax_wrapper( stax.FanInConcat )

################################################################################################################

def stax_conv_wrapper( fun ):
    def ret( *args, name='unnamed', **kwargs ):

        # Some stax layers don't need to be called
        if( isinstance( fun, tuple ) ):
            _init_fun, _apply_fun = fun
        else:
            _init_fun, _apply_fun = fun( *args, **kwargs )

        def init_fun( key, input_shape ):
            # JAX conv is weird with batch dims
            assert len( input_shape ) == 3
            input_shape = ( 1, ) + input_shape
            output_shape, params = _init_fun( key, input_shape )
            output_shape = output_shape[1:]
            return name, output_shape, params, ()

        def apply_fun( params, static_params, inputs, **kwargs ):
            if( inputs.ndim == 3 ):
                ans = _apply_fun( params, inputs[None], **kwargs )[0]
            else:
                ans = _apply_fun( params, inputs, **kwargs )

            return ans, ()

        return init_fun, apply_fun

    return ret

Conv = stax_conv_wrapper( stax.Conv )
ConvTranspose = stax_conv_wrapper( stax.ConvTranspose )
MaxPool = stax_conv_wrapper( stax.MaxPool )
SumPool = stax_conv_wrapper( stax.SumPool )
AvgPool = stax_conv_wrapper( stax.AvgPool )

################################################################################################################

def Reshape( shape, name='unnamed' ):
    """ Reshape an input """
    total_dim = np.prod( shape )

    def init_fun( key, input_shape ):
        assert np.prod( input_shape ) == total_dim
        return name, shape, (), ()

    def apply_fun( params, static_params, inputs, **kwargs ):
        if( np.prod( inputs.shape ) != total_dim ):
            assert np.prod( inputs.shape ) % total_dim == 0
            return np.reshape( inputs, ( -1, ) + shape ), ()
        return np.reshape( inputs, shape ), ()

    return init_fun, apply_fun

################################################################################################################

def Dense( out_dim, mask_id=None, keep_prob=1.0, W_init=glorot_normal(), b_init=normal(), name='unnamed' ):
    """ Fully connected layer with dropout and an option to retrieve a mask

        Args:
            out_dim   - The output dimension.  Input is filled automatically during initialization
            mask_id   - A string that indexes into the kwargs to retrieve the mask
            keep_prob - The probability of keeping a weight in the matrix during train time
    """
    use_dropout = keep_prob < 0.999

    def init_fun( key, input_shape ):
        output_shape = input_shape[:-1] + ( out_dim, )
        k1, k2 = random.split( key )
        W, b = W_init( k1, ( input_shape[-1], out_dim ) ), b_init( k2, ( out_dim, ) )
        return name, output_shape, ( W, b ), ()

    def apply_fun( params, static_params, inputs, **kwargs ):

        test = kwargs['test'] if 'test' in kwargs else TRAIN
        W, b = params

        # Dropout
        if( use_dropout and is_testing( test ) == False ):

            key = kwargs.get( 'key', None )
            if( key is None ):
                assert 0, 'Need key for this'

            keep = random.bernoulli( key, keep_prob, W.shape )
            W = np.where( keep, W / keep_prob, 0 )

        # Mask W is needed
        if( mask_id is not None ):
            mask = kwargs[mask_id]
            W = W*mask

        return np.dot( inputs, W ) + b, static_params

    return init_fun, apply_fun

################################################################################################################

def BatchNorm( axis=0, epsilon=1e-5, alpha=0.05, beta_init=zeros, gamma_init=ones, name='unnamed' ):
    """ Batch normaliziation

        Args:
            axis    - Batch axis
            epsilon - Constant for numerical stability
            alpha   - Parameter for exponential moving average of population parameters
    """
    def init_fun( key, input_shape ):
        k1, k2 = random.split( key )
        beta, gamma = beta_init( k1, ( input_shape[-1], ) ), gamma_init( k2, ( input_shape[-1], ) )
        running_mean = np.zeros( input_shape )
        running_var = np.ones( input_shape )
        return name, input_shape, ( beta, gamma ), ( running_mean, running_var )

    def get_bn_params( x, test, running_mean, running_var ):
        if( is_testing( test ) ):
            mean, var = running_mean, running_var
        else:
            mean = np.mean( x, axis=axis )
            var = np.var( x, axis=axis ) + epsilon
            running_mean = ( 1 - alpha )*running_mean + alpha*mean
            running_var = ( 1 - alpha )*running_var + alpha*var

        return ( mean, var ), ( running_mean, running_var )

    def apply_fun( params, static_params, inputs, **kwargs ):
        beta, gamma = params
        running_mean, running_var = static_params
        x = inputs

        # Check if we're training or testing
        test = kwargs['test'] if 'test' in kwargs else TRAIN

        # Update the running population parameters
        ( mean, var ), ( running_mean, running_var ) = get_bn_params( x, test, running_mean, running_var )

        # Normalize the inputs
        x_hat = ( x - mean ) / np.sqrt( var )
        z = gamma*x_hat + beta

        return z, ( running_mean, running_var )

    return init_fun, apply_fun

################################################################################################################

def Residual( network, name='unnamed' ):
    """ Create a residual layer for a given network

        Args:
            network - Input network
    """
    _init_fun, _apply_fun = network

    def init_fun( key, input_shape ):
        _, output_shape, params, static_params = _init_fun( key, input_shape )
        # We're adding the input and output, so need to preserve shape
        assert output_shape == input_shape
        return name, input_shape, params, static_params

    def apply_fun( params, static_params, inputs, **kwargs ):
        outputs, updated_static_params = _apply_fun( params, static_params, inputs )
        return inputs + outputs, updated_static_params

    return init_fun, apply_fun

################################################################################################################

def GaussianMADE( dim,
                  hidden_layer_sizes,
                  reverse=False,
                  method='sequential',
                  key=None,
                  name='unnamed' ):
    """ Gaussian MADE https://arxiv.org/pdf/1502.03509.pdf
        Network that enforces autoregressive property.  Until I think of a cleaner way, the networks that
        comprise the MADE are defined in here.

        Args:
            dim                - The dimension of the input
            hidden_layer_sizes - A list of the size of the feature network
            reverse            - Whether or not to reverse the inputs
            method             - Either 'sequential' or 'random'.  Controls how indices are assigned
                                 to nodes in each layer
            key                - JAX random key.  Only needed in random mode
    """
    layer_sizes = hidden_layer_sizes + [ dim ]

    # Build the weights for the hidden layers
    dense_layers = []
    for j, size in enumerate( layer_sizes ):
        dense_layers.append( Dense( size, mask_id='mask_%d'%( j ) ) )
        dense_layers.append( Relu() )
    hidden_path = sequential( *dense_layers )

    # Build the mean and log std weights
    mu_out = Dense( dim, mask_id='mu' )
    alpha_out = sequential( Dense( dim, mask_id='alpha' ), Tanh() ) # Bound alpha to avoid numerical instability

    # Create the layers of the network
    param_architecture = sequential( hidden_path,
                                 FanOut( 2 ),
                                 parallel( mu_out, alpha_out ) )

    # Create the weights masks to ensure the autoregressive property
    if( method == 'random' ):
        assert key is not None
        keys = random.split( key, len( layer_sizes ) + 1 )
        key_iter = iter( keys )
        input_sel = random.randint( next( key_iter ), shape=( dim, ), minval=1, maxval=dim+1 )
    else:
        # Alternate direction in consecutive layers
        input_sel = np.arange( 1, dim + 1 )
        if( reverse ):
            input_sel = input_sel[::-1]

    # Build the hidden layer masks
    masks = []
    sel = input_sel
    prev_sel = sel
    for size in layer_sizes:

        # Choose the degrees of the next layer
        if( method == 'random' ):
            sel = random.randint( next( key_iter ), shape=( size, ), minval=min( np.min( sel ), dim - 1 ), maxval=dim )
        else:
            sel = np.arange( size )%max( 1, dim - 1 ) + min( 1, dim - 1 )

        # Create the new mask
        mask = ( prev_sel[:,None] <= sel ).astype( np.int32 )
        prev_sel = sel
        masks.append( mask )

    # Build the mask for the matrix between the input and output
    skip_mask = ( input_sel[:,None] < input_sel ).astype( np.int32 )

    # Build the output layers.  Remember that we need to output mu and sigma.  Just need
    # a triangular matrix for the masks
    out_mask = ( prev_sel[:,None] < input_sel ).astype( np.int32 )

    # Load the masks into a dictionary
    mask_kwargs = dict( [ ( 'mask_%d'%( j ), mask ) for j, mask in enumerate( masks ) ] )
    mask_kwargs['skip'] = skip_mask
    mask_kwargs['mu'] = out_mask
    mask_kwargs['alpha'] = out_mask

    # Fill the network application with the mask kwargs so the user doesn't have to
    init_params, network = param_architecture
    network = partial( network, **mask_kwargs )

    def init_fun( key, input_shape ):
        x_shape = input_shape
        name, out_shape, params, static_params = init_params( key, x_shape )
        ( mu_shape, alpha_shape ) = out_shape
        return name, out_shape, params, static_params

    def apply_fun( params, static_params, inputs, **kwargs ):
        ( mu, alpha ), updated_static_params = network( params, static_params, inputs, **kwargs )
        return ( mu, alpha ), updated_static_params

    return init_fun, apply_fun

################################################################################################################
