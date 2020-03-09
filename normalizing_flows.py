import numpy as onp
import jax
from jax import random, jit, vmap, jacobian, grad, value_and_grad
import jax.nn
import jax.numpy as np
from functools import partial, reduce
from jax.experimental import stax
from jax.nn.initializers import glorot_normal, normal, ones, zeros
from jax.ops import index, index_add, index_update
from staxplusplus import *
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_multimap
from util import is_testing, TRAIN, TEST

################################################################################################################

def flow_test( flow, x, key, **kwargs ):
    """ Test a flow layer.  Assumes that x is batched!  ASSUMES THERE IS NO PRIOR AT THE END OF THE FLOW!!!

        Args:
            flow - A normalizing flow
            x    - A batched input
            key  - JAX random key
    """
    init_fun, forward, inverse = flow

    input_shape = x.shape[1:]
    condition_shape = ( input_shape, )
    cond = ( x, )

    names, output_shape, params, static_params = init_fun( key, input_shape, condition_shape )

    # Make sure that the forwards and inverse functions are consistent
    log_px, z, updated_static_params = forward( params, static_params, np.zeros( x.shape[0] ), x, cond, test=TEST, **kwargs )
    log_pfz, fz, updated_static_params = inverse( params, static_params, np.zeros( z.shape[0] ), z, cond, test=TEST, **kwargs )

    x_diff = np.linalg.norm( x - fz )
    log_px_diff = np.linalg.norm( log_px - log_pfz )
    print( 'Transform consistency diffs: x_diff: %5.3f, log_px_diff: %5.3f'%( x_diff, log_px_diff ) )

    # We are assuming theres no prior!!!!!
    log_det = log_px

    # Make sure that the log det terms are correct
    def z_from_x( unflatten, x_flat, cond, **kwargs ):
        x = unflatten( x_flat )
        z = forward( params, static_params, 0, x, cond, test=TEST, **kwargs )[1]
        return ravel_pytree( z )[0]

    def single_elt_logdet( x, cond, **kwargs ):
        x_flat, unflatten = ravel_pytree( x )
        jac = jacobian( partial( z_from_x, unflatten, **kwargs ) )( x_flat, cond )
        return 0.5*np.linalg.slogdet( jac.T@jac )[1]

    actual_log_det = vmap( single_elt_logdet )( x, cond, **kwargs )

    # print( 'actual_log_det', actual_log_det )
    # print( 'log_det', log_det )

    log_det_diff = np.linalg.norm( log_det - actual_log_det )
    print( 'Log det diff: %5.3f'%( log_det_diff ) )

################################################################################################################

def sequential_flow( *layers ):
    """ Sequential invertible network builder.  Same as sequential, but is invertible

        Args:
            *layers - An unpacked list of (name, init_fun, apply_fun)
    """

    n_layers = len( layers )
    init_funs, forward_funs, inverse_funs = zip( *layers )

    def init_fun( key, input_shape, condition_shape ):
        names, params, static_params = [], [], []
        for init_fun in init_funs:
            key, *keys = random.split( key, 2 )
            # Conditioning can only be added in a factor call or at the top level call
            name, input_shape, param, static_param = init_fun( keys[0], input_shape, condition_shape )
            names.append( name )
            params.append( param )
            static_params.append( static_param )
        return tuple( names ), input_shape, tuple( params ), tuple( static_params )

    def evaluate( apply_funs, params, static_params, log_px, inputs, condition, **kwargs ):

        # Need to store the ouputs of the functions and the updated static params
        updated_static_params = []

        # Need to pop so that we don't resuse random keys!
        key = kwargs.pop( 'key', None )
        keys = random.split( key, n_layers ) if key is not None else ( None, )*n_layers

        # Evaluate each function and store the updated static parameters
        for fun, param, static_param, key in zip( apply_funs, params, static_params, keys ):
            log_px, inputs, updated_static_param = fun( param, static_param, log_px, inputs, condition, key=key, **kwargs )
            updated_static_params.append( updated_static_param )

        return log_px, inputs, tuple( updated_static_params )

    def forward( params, static_params, log_px, x, condition, **kwargs ):
        return evaluate( forward_funs, params, static_params, log_px, x, condition, **kwargs )

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):
        return evaluate( inverse_funs[::-1], params[::-1], static_params[::-1], log_pz, z, condition, **kwargs )

    return init_fun, forward, inverse

def factored_flow( *layers ):
    """ Like stax.parallel, but over a probability distribution.  Factors the distribution
        as p(x) = p([x_1,x_2,...x_N]) = p(x_1)p(x_2|x_1)*...*p(x_N|x_N-1,...,x_1).
        The result of each factor is passed to the next factor in order to make inversion possible.
        Basically a less granular version of a MAF.

        Args:
            *layers - An unpacked list of (name, init_fun, apply_fun)
    """

    n_layers = len( layers )
    init_funs, forward_funs, inverse_funs = zip( *layers )

    # Feature extract network
    fe_apply_fun = None

    def init_fun( key, input_shape, condition_shape ):
        keys = random.split( key, n_layers + 1 )

        # Find the shapes of all of the conditionals
        names, output_shapes, params, static_params = [], [], [], []

        # Split these up so that we can evaluate each of the parallel items together
        for init_fun, key, shape in zip( init_funs, keys, input_shape ):
            name, output_shape, param, static_param = init_fun( key, shape, condition_shape )
            names.append( name )
            output_shapes.append( output_shape )
            params.append( param )
            static_params.append( static_param )

            condition_shape = condition_shape + ( output_shape, )

        return tuple( names ), output_shapes, tuple( params ), tuple( static_params )

    def forward( params, static_params, log_px, x, condition, **kwargs ):

        # Need to pop so that we don't resuse random keys!
        key = kwargs.pop( 'key', None )
        n_keys = n_layers if fe_apply_fun is None else n_layers*2
        keys = random.split( key, n_keys ) if key is not None else ( None, )*n_keys
        key_iter = iter( keys )

        # We need to store each of the outputs and static params
        densities, outputs, updated_static_params = [], [], []
        for apply_fun, param, static_param, lpx, inp in zip( forward_funs, params, static_params, log_px, x ):
            lpx, output, updated_static_param = apply_fun( param, static_param, lpx, inp, condition, key=next( key_iter ), **kwargs )
            densities.append( lpx )
            outputs.append( output )
            updated_static_params.append( updated_static_param )

            # Need to do autoregressive type conditioning
            condition = condition + ( output, )

        return densities, outputs, tuple( updated_static_params )

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):

        # Need to pop so that we don't resuse random keys!
        key = kwargs.pop( 'key', None )
        n_keys = n_layers if fe_apply_fun is None else n_layers*2
        keys = random.split( key, n_keys ) if key is not None else ( None, )*n_keys
        key_iter = iter( keys )

        # We need to store each of the outputs and static params
        densities, outputs, updated_static_params = [], [], []
        for apply_fun, param, static_param, lpz, inp in zip( inverse_funs, params, static_params, log_pz, z ):
            lpz, output, updated_static_param = apply_fun( param, static_param, lpz, inp, condition, key=next( key_iter ), **kwargs )
            densities.append( lpz )
            outputs.append( output )
            updated_static_params.append( updated_static_param )

            # Conditioners are inputs during the inverse pass
            condition = condition + ( inp, )

        return densities, outputs, tuple( updated_static_params )

    return init_fun, forward, inverse

################################################################################################################

def UnitGaussianPrior( axis=( -1, ), name='unnamed' ):
    """ Prior that should be placed at the beginning of the flow (farthest layer from data!)

        Args:
    """
    def init_fun( key, input_shape, condition_shape ):
        return name, input_shape, (), ()

    def forward( params, static_params, log_px, x, condition, **kwargs ):
        dim = np.prod( [ x.shape[ax] for ax in axis ] )
        log_px += -0.5*np.sum( x**2, axis=axis ) + -0.5*dim*np.log( 2*np.pi )
        return log_px, x, ()

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):
        dim = np.prod( [ z.shape[ax] for ax in axis ] )
        log_pz += -0.5*np.sum( z**2, axis=axis ) + -0.5*dim*np.log( 2*np.pi )
        return log_pz, z, ()

    return init_fun, forward, inverse

################################################################################################################

def Identity( name=None ):
    """ Identity function """
    def init_fun( key, input_shape, condition_shape ):
        return name, input_shape, (), ()

    def forward( params, static_params, log_px, x, condition, **kwargs ):
        return log_px, x, ()

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):
        return log_pz, z, ()

    return init_fun, forward, inverse

def Reverse( name=None ):
    """ Reverse the order of the input

        Args:
    """
    def init_fun( key, input_shape, condition_shape ):
        return name, input_shape, (), ()

    def forward( params, static_params, log_px, x, condition, **kwargs ):
        return log_px, x[...,::-1], ()

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):
        return log_pz, z[...,::-1], ()

    return init_fun, forward, inverse

def Reshape( shape, name='unnamed' ):
    """ Reshape the input

        Args:
    """

    original_shape = None

    def init_fun( key, input_shape, condition_shape ):
        nonlocal original_shape
        original_shape = input_shape
        return name, shape, (), ()

    def forward( params, static_params, log_px, x, condition, **kwargs ):
        if( x.ndim > len( original_shape ) ):
            z = x.reshape( ( -1, ) + shape )
        else:
            z = x.reshape( shape )

        return log_px, z, ()

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):
        if( z.ndim > len( shape ) ):
            x = z.reshape( ( -1, ) + original_shape )
        else:
            x = z.reshape( original_shape )

        return log_pz, x, ()

    return init_fun, forward, inverse

################################################################################################################

def FactorOut( num, axis=-1, name='unnamed' ):
    """ Factor p(z_{1..N}) = p(z_1)p(z_2|z_1)...p(z_N|z_{1..N-1})

        Args:
            num  - Number of components to split into
            axis - Axis to split
    """
    def init_fun( key, input_shape, condition_shape ):
        ax = axis % len( input_shape )

        # For the moment, ensure we split evenly
        assert input_shape[ax]%num == 0

        split_shape = list( input_shape )
        split_shape[ax] = input_shape[ax]//num
        split_shape = tuple( split_shape )

        return name, [ split_shape ]*num, (), ()

    def forward( params, static_params, log_px, x, condition, **kwargs ):
        z_components = np.split( x, num, axis )

        # Only send the total density through one component.  The density will be recombined later
        log_pxs = [ log_px if i == 0 else np.zeros_like( log_px ) for i, z_i in enumerate( z_components ) ]
        zs = z_components

        return log_pxs, zs, ()

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):
        log_pz = sum( log_pz )
        x = np.concatenate( z, axis )
        return log_pz, x, ()

    return init_fun, forward, inverse

def FanInConcat( num, axis=-1, name='unnamed' ):
    """ Inverse of FactorOut

        Args:
            axis - Axis to concat on
    """
    def init_fun( key, input_shape, condition_shape ):
        # Make sure that each of the inputs are the same size
        assert num == len( input_shape )
        for shape in input_shape:
            assert shape == input_shape[0]
        ax = axis % len( input_shape[0] )
        concat_size = sum( shape[ax] for shape in input_shape )
        out_shape = input_shape[0][:ax] + ( concat_size, ) + input_shape[0][ax+1:]
        return name, out_shape, (), ()

    _, inverse, forward = FactorOut( num, axis=axis )

    return init_fun, forward, inverse

################################################################################################################

def indexer_from_mask( mask ):
    """ Given a 2d mask array, create an array that can index into a vector with the same number of elements
        as nonzero elements in mask and result in an array of the same size as mask, but with the elements
        specified from the vector.

    Args:
        mask - 2d boolean mask array

    Returns:
        index - 2d array with nonzero elements corresponding to a range starting at 1
    """
    index = onp.zeros_like( mask, dtype=int )
    non_zero_indices = np.nonzero( mask )
    index[non_zero_indices] = np.arange( len( non_zero_indices[0] ) ) + 1
    return index

def indexer_and_shape_from_mask( mask ):
    """ Given a 2d mask array, create an array that can index into a vector with the same number of elements
        as nonzero elements in mask and result in an array of the same size as mask, but with the elements
        specified from the vector.  Also return the shape of the resulting array when mask is applied

    Args:
        mask - 2d boolean mask array

    Returns:
        index - 2d array with nonzero elements corresponding to a range starting at 1
        shape - The shape of the resulting array when mask is applied
    """
    index = onp.zeros_like( mask, dtype=int )
    non_zero_indices = np.nonzero( mask )
    index[non_zero_indices] = np.arange( len( non_zero_indices[0] ) ) + 1

    nonzero_x, nonzero_y = non_zero_indices
    n_rows = onp.unique( nonzero_x ).size
    assert nonzero_x.size%n_rows == 0
    n_cols = nonzero_x.size // n_rows
    shape = ( n_rows, n_cols )
    return index, shape

def check_mask( mask ):
    """ Check if the 2d boolean mask is valid

    Args:
        mask - 2d boolean mask array
    """
    if( np.any( mask ) == False ):
        assert 0, 'Empty mask!  Reduce num'
    if( np.sum( mask )%2 == 1 ):
        assert 0, 'Need masks with an even number!  Choose a different num'

def checkerboard_masks( num, shape ):
    """ Finds masks to factor an array with a given shape so that each pixel will be
        present in the resulting masks.  Also return indices that will help reverse
        the factorization.

    Args:
        masks   - A list of 2d boolean mask array whose union is equal to np.ones( shape, dtype=bool )
        indices - A list of index matrices that undo the application of image[mask]
    """
    masks = []
    indices = []
    shapes = []

    for i in range( 2*num ):
        start = 2**i
        step = 2**( i + 1 )

        # Staggered indices
        mask = onp.zeros( shape, dtype=bool )
        mask[start::step,::step] = True
        mask[::step,start::step] = True
        check_mask( mask )
        masks.append( mask )
        index, new_shape = indexer_and_shape_from_mask( mask )
        indices.append( index )
        shapes.append( new_shape )

        if( len( masks ) + 1 == num ):
            break

        # Not staggered indices
        mask = onp.zeros( shape, dtype=bool )
        mask[start::step,start::step] = True
        mask[start::step,start::step] = True
        check_mask( mask )
        masks.append( mask )
        index, new_shape = indexer_and_shape_from_mask( mask )
        indices.append( index )
        shapes.append( new_shape )

        if( len( masks ) + 1 == num ):
            break

    used = sum( masks ).astype( bool )
    mask = ~used
    masks.append( mask )
    index, new_shape = indexer_and_shape_from_mask( mask )
    indices.append( index )
    shapes.append( new_shape )

    return masks, indices, shapes

def recombine( z, index ):
    return np.pad( z.ravel(), ( 1, 0 ) )[index]

def CheckerboardFactor( num, name='unnamed' ):
    """ Factor an image using a checkerboard pattern.  Basically each split will be a lower resolution
        image of the original

    Args:
        num  - Number of components to split into
        axis - Axis to split
    """

    masks, indices, shapes = None, None, None

    def init_fun( key, input_shape, condition_shape ):
        height, width, channel = input_shape

        nonlocal masks, indices, shapes
        masks, indices, shapes = checkerboard_masks( num, ( height, width ) )

        # Swap the order so that our factors go from lowest to highest resolution
        masks, indices, shapes = masks[::-1], indices[::-1], shapes[::-1]

        # Add the channel dim back in
        shapes = [ ( h, w, channel ) for ( h, w ) in shapes ]
        return name, shapes, (), ()

    def forward( params, static_params, log_px, x, condition, **kwargs ):
        if( x.ndim == 4 ):
            # To make things easier, vmap over the batch dimension
            return vmap( partial( forward, params, static_params, **kwargs ), in_axes=( 0, 0, None ) )( log_px, x, condition )

        assert x.ndim == 3

        # Split the pixels into disjoint sets
        zs = []
        for mask, shape in zip( masks, shapes ):
            z = x[mask].reshape( shape )
            zs.append( z )

        # Only send the total density through one component.  The density will be recombined later
        log_pxs = [ log_px if i == 0 else np.zeros_like( log_px ) for i in range( num ) ]

        return log_pxs, zs, ()

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):

        # Add the densities
        log_pz = sum( log_pz )

        # Recombine the pixels into an image
        recombine_vmapped = vmap( recombine, in_axes=( 2, None ), out_axes=2 )

        # If z is batched, then need an extra vmap
        if( z[0].ndim == 4 ):
            recombine_vmapped = vmap( recombine_vmapped, in_axes=( 0, None ), out_axes=0 )

        x = recombine_vmapped( z[0], indices[0] )
        for elt, index in zip( z[1:], indices[1:] ):
            x += recombine_vmapped( elt, index )

        return log_pz, x, ()

    return init_fun, forward, inverse

def CheckerboardCombine( num, name='unnamed' ):
    """ Inverse of CheckerboardFactor

    Args:
        num  - Number of components to split into
        axis - Axis to split
    """

    masks, indices, shapes = None, None, None

    def init_fun( key, input_shape, condition_shape ):
        assert num == len( input_shape )

        # By construction, the height of the last shape is the height of the total image
        height, _, channel = input_shape[-1]

        # Count the total number of pixels
        total_pixels = 0
        for h, w, c in input_shape:
            total_pixels += h*w

        assert total_pixels%height == 0
        width = total_pixels // height

        output_shape = ( height, width, channel )

        # Need to know ahead of time what the masks are and the indices and shapes to undo the masks are
        nonlocal masks, indices, shapes
        masks, indices, shapes = checkerboard_masks( num, ( height, width ) )

        # Swap the order so that our factors go from lowest to highest resolution
        masks, indices, shapes = masks[::-1], indices[::-1], shapes[::-1]

        # Add the channel dim back in
        shapes = [ ( h, w, channel ) for ( h, w ) in shapes ]

        return name, output_shape, (), ()

    def forward( params, static_params, log_px, x, condition, **kwargs ):

        # Add the densities
        log_px = sum( log_px )

        # Recombine the pixels into an image
        recombine_vmapped = vmap( recombine, in_axes=( 2, None ), out_axes=2 )

        # If x is batched, then need an extra vmap
        if( x[0].ndim == 4 ):
            recombine_vmapped = vmap( recombine_vmapped, in_axes=( 0, None ), out_axes=0 )

        z = recombine_vmapped( x[0], indices[0] )
        for elt, index in zip( x[1:], indices[1:] ):
            z += recombine_vmapped( elt, index )

        return log_px, z, ()

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):
        if( z.ndim == 4 ):
            # To make things easier, vmap over the batch dimension
            return vmap( partial( inverse, params, static_params, **kwargs ), in_axes=( 0, 0, None ) )( log_pz, z, condition )

        assert z.ndim == 3

        # Split the pixels into disjoint sets
        xs = []
        for mask, shape in zip( masks, shapes ):
            x = z[mask].reshape( shape )
            xs.append( x )

        # Only send the total density through one component.  The density will be recombined later
        log_pzs = [ log_pz if i == 0 else np.zeros_like( log_pz ) for i in range( num ) ]

        return log_pzs, xs, ()

    return init_fun, forward, inverse

################################################################################################################

def CheckerboardSqueeze( num=2, name='unnamed' ):
    """ See Figure 3 here https://arxiv.org/pdf/1605.08803.pdf.

    Args:
    """
    assert num == 2, 'More not implemented yet'
    masks, indices, shapes = None, None, None

    def init_fun( key, input_shape, condition_shape ):
        height, width, channel = input_shape

        # Find the masks in order to split the image correctly
        nonlocal masks, indices, shapes
        masks, indices, shapes = checkerboard_masks( 2, ( height, width ) )
        shapes = [ ( h, w, channel ) for ( h, w ) in shapes ]

        # We need to get the same shapes
        assert shapes[0] == shapes[1]

        # Get the output shape
        out_height, out_width, _ = shapes[0]
        output_shape = ( out_height, out_width, 2*channel )

        # This is by construction!
        assert out_height == height

        return name, output_shape, (), ()

    def forward( params, static_params, log_px, x, condition, **kwargs ):
        if( x.ndim == 4 ):
            # To make things easier, vmap over the batch dimension
            return vmap( partial( forward, params, static_params, **kwargs ), in_axes=( 0, 0, None ) )( log_px, x, condition )

        assert x.ndim == 3

        # Split the image and concatenate along the channel dimension
        z1 = x[masks[0]].reshape( shapes[0] )
        z2 = x[masks[1]].reshape( shapes[1] )
        z = np.concatenate( [ z1, z2 ], axis=-1 )

        # This is basically a permuation matrix, so the log abs determinant is 0
        return log_px, z, ()

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):
        # Split the image on the channel dimension
        x1, x2 = np.split( z, 2, axis=-1 )

        # Recombine the pixels into an image
        recombine_vmapped = vmap( recombine, in_axes=( 2, None ), out_axes=2 )

        # If z is batched, then need an extra vmap
        if( x1.ndim == 4 ):
            recombine_vmapped = vmap( recombine_vmapped, in_axes=( 0, None ), out_axes=0 )

        x = recombine_vmapped( x1, indices[0] ) + recombine_vmapped( x2, indices[1] )

        return log_pz, x, ()

    return init_fun, forward, inverse

def CheckerboardUnSqueeze( num=2, name='unnamed' ):
    """ Inverse of CheckerboardSqueeze

    Args:
    """
    assert num == 2, 'More not implemented yet'
    masks, indices, shapes = None, None, None

    def init_fun( key, input_shape, condition_shape ):
        # Height remained unchanged
        height, width, channel = input_shape
        assert channel%2 == 0

        # Find the output shape
        out_height = height
        out_width = width*2
        out_channel = channel // 2
        output_shape = ( out_height, out_width, out_channel )

        # Create the masks
        nonlocal masks, indices, shapes
        masks, indices, shapes = checkerboard_masks( 2, ( out_height, out_width ) )
        shapes = [ ( h, w, out_channel ) for ( h, w ) in shapes ]

        return name, output_shape, (), ()

    def forward( params, static_params, log_px, x, condition, **kwargs ):
        # Split the image on the channel dimension
        z1, z2 = np.split( x, 2, axis=-1 )

        # Recombine the pixels into an image
        recombine_vmapped = vmap( recombine, in_axes=( 2, None ), out_axes=2 )

        # If x is batched, then need an extra vmap
        if( z1.ndim == 4 ):
            recombine_vmapped = vmap( recombine_vmapped, in_axes=( 0, None ), out_axes=0 )

        z = recombine_vmapped( z1, indices[0] ) + recombine_vmapped( z2, indices[1] )

        return log_px, z, ()

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):
        if( z.ndim == 4 ):
            # To make things easier, vmap over the batch dimension
            return vmap( partial( inverse, params, static_params, **kwargs ), in_axes=( 0, 0, None ) )( log_pz, z, condition )

        assert z.ndim == 3

        # Split the image and concatenate along the channel dimension
        x1 = z[masks[0]].reshape( shapes[0] )
        x2 = z[masks[1]].reshape( shapes[1] )
        x = np.concatenate( [ x1, x2 ], axis=-1 )

        # This is basically a permuation matrix, so the log abs determinant is 0
        return log_pz, x, ()

    return init_fun, forward, inverse

################################################################################################################

def actnorm_init( x, actnorm_names, names, params, static_params, forward ):
    """ Initialize a network that has actnorms. https://arxiv.org/pdf/1807.03039.pdf section 3.1

        Args:
            x             - The data seed
            actnorm_names - A list of the actnorm parameters to update
            names         - A pytree (nested structure) of names
            params        - The parameter pytree
            staic_params  - The static parameter pytree
            forward       - Forward function

        Returns:
            Seeded parameters
    """

    _, _, updated_static_params = forward( params, static_params, np.zeros( x.shape[0] ), x, (), actnorm_seed=True )
    for an_name in actnorm_names:
        seeded_parm = get_param( an_name, names, updated_static_params )
        params = modify_param( an_name, names, params, seeded_parm )
    return params

def ActNorm( log_s_init=zeros, b_init=zeros, name='unnamed' ):
    """ Act norm normalization.  Act norm requires a data seed in order to properly initialize
        its parameters.  This will be done at runtime.

        Args:
    """

    def init_fun( key, input_shape, condition_shape ):
        k1, k2 = random.split( key )
        log_s = b_init( k1, ( input_shape[-1], ) )
        b = b_init( k2, ( input_shape[-1], ) )

        return name, input_shape, ( log_s, b ), ()

    def forward( params, static_params, log_px, x, condition, **kwargs ):
        log_s, b = params

        # Check to see if we're seeding this function
        actnorm_seed = kwargs.get( 'actnorm_seed', False )
        if( actnorm_seed == True ):
            # The initial parameters should normalize the input
            mean = np.mean( x, axis=0 )
            std = np.std( x, axis=0 ) + 1e-5
            log_s = -np.log( std )
            b = -mean/std
            static_params = ( log_s, b )
        else:
            static_params = ()

        z = np.exp( log_s )*x + b
        log_det = log_s.sum()

        return log_px + log_det, z, static_params

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):
        log_s, b = params
        x = ( z - b )*np.exp( -log_s )

        log_det = log_s.sum()
        return log_pz + log_det, x, ()

    return init_fun, forward, inverse

def BatchNorm( epsilon=1e-5, alpha=0.05, beta_init=zeros, gamma_init=zeros, name='unnamed' ):
    """ Invertible batch norm

        Args:
            axis    - Batch axis
            epsilon - Constant for numerical stability
            alpha   - Parameter for exponential moving average of population parameters
    """

    def init_fun( key, input_shape, condition_shape ):
        k1, k2 = random.split( key )
        beta, log_gamma = beta_init( k1, input_shape ), gamma_init( k2, input_shape )
        running_mean = np.zeros( input_shape )
        running_var = np.ones( input_shape )
        return name, input_shape, ( beta, log_gamma ), ( running_mean, running_var )

    def get_bn_params( x, test, running_mean, running_var ):
        if( is_testing( test ) ):
            mean, var = running_mean, running_var
        else:
            mean = np.mean( x, axis=0 )
            var = np.var( x, axis=0 ) + epsilon
            running_mean = ( 1 - alpha )*running_mean + alpha*mean
            running_var = ( 1 - alpha )*running_var + alpha*var

        return ( mean, var ), ( running_mean, running_var )

    def forward( params, static_params, log_px, x, condition, **kwargs ):
        beta, log_gamma = params
        running_mean, running_var = static_params

        # Check if we're training or testing
        test = kwargs['test'] if 'test' in kwargs else TRAIN

        # Update the running population parameters
        ( mean, var ), ( running_mean, running_var ) = get_bn_params( x, test, running_mean, running_var )

        # Normalize the inputs
        x_hat = ( x - mean ) / np.sqrt( var )
        z = np.exp( log_gamma )*x_hat + beta

        log_det = log_gamma.sum()*np.ones( ( z.shape[0], ) )
        log_det += -0.5*np.log( var ).sum()

        return log_px + log_det, z, ( running_mean, running_var )

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):
        beta, log_gamma = params
        running_mean, running_var = static_params

        # Check if we're training or testing
        test = kwargs['test'] if 'test' in kwargs else TRAIN

        # Update the running population parameters
        ( mean, var ), ( running_mean, running_var ) = get_bn_params( z, test, running_mean, running_var )

        # Normalize the inputs
        z_hat = ( z - beta )*np.exp( -log_gamma )
        x = z_hat*np.sqrt( var ) + mean

        log_det = log_gamma.sum()*np.ones( ( z.shape[0], ) )
        log_det += -0.5*np.log( var ).sum()

        return log_pz + log_det, x, ( running_mean, running_var )

    return init_fun, forward, inverse

################################################################################################################

def upper_triangular_indices( N ):
    values = np.arange( N )
    padded_values = np.hstack( [ values, 0 ] )

    idx = onp.ogrid[:N,N:0:-1]
    idx = sum( idx ) - 1

    mask = np.arange( N ) >= np.arange( N )[:,None]
    return ( idx + np.cumsum( values + 1 )[:,None][::-1] - N + 1 )*mask

def n_elts_upper_triangular( N ):
    return N*( N + 1 ) // 2 - 1

def upper_triangular_from_values( vals, N ):
    assert n_elts_upper_triangular( N ) == vals.shape[-1]
    zero_padded_vals = np.pad( vals, ( 1, 0 ) )
    return zero_padded_vals[upper_triangular_indices( N )]

tri_solve = jax.scipy.linalg.solve_triangular
L_solve = jit( partial( tri_solve, lower=True, unit_diagonal=True ) )
U_solve = jit( partial( tri_solve, lower=False, unit_diagonal=True ) )

def AffineLDU( L_init=normal(), d_init=ones, U_init=normal(), name='unnamed', return_mat=False ):
    """ LDU parametrized square dense matrix

        Args:
    """

    triangular_indices = None

    def init_fun( key, input_shape, condition_shape ):
        # This is a square matrix!
        dim = input_shape[-1]

        # Create the fancy indices that we'll use to turn our vectors into triangular matrices
        nonlocal triangular_indices
        triangular_indices = np.pad( upper_triangular_indices( dim - 1 ), ( ( 0, 1 ), ( 1, 0 ) ) )
        flat_dim = n_elts_upper_triangular( dim )

        k1, k2, k3 = random.split( key, 3 )
        L_flat, d, U_flat = L_init( k1, ( flat_dim, ) ), d_init( k2, ( dim, ) ), U_init( k3, ( flat_dim, ) )
        return name, input_shape, ( L_flat, d, U_flat ), ()

    def get_LDU( params ):
        L_flat, d, U_flat = params

        L = np.pad( L_flat, ( 1, 0 ) )[triangular_indices]
        L = L + np.eye( d.shape[-1] )
        L = L.T

        U = np.pad( U_flat, ( 1, 0 ) )[triangular_indices]
        U = U + np.eye( d.shape[-1] )

        return L, d, U

    def forward( params, static_params, log_px, x, condition, **kwargs ):
        # Go from x to x
        if( x.ndim == 2 ):
            return vmap( partial( forward, params, static_params, **kwargs ), in_axes=( 0, 0, None ) )( log_px, x, condition )

        L, d, U = get_LDU( params )

        z = np.einsum( 'ij,j->i', U, x )
        z = z*d
        z = np.einsum( 'ij,j->i', L, z )

        log_det = np.sum( np.log( np.abs( d ) ), axis=-1 )

        return log_px + log_det, z, ()

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):
        if( z.ndim == 2 ):
            return vmap( partial( inverse, params, static_params, **kwargs ), in_axes=( 0, 0, None ) )( log_pz, z, condition )

        L, d, U = get_LDU( params )

        x = L_solve( L, z )
        x = x/d
        x = U_solve( U, x )

        log_det = np.sum( np.log( np.abs( d ) ), axis=-1 )

        return log_pz + log_det, x, ()

    # Have the option to directly get the matrix
    if( return_mat ):
        return init_fun, forward, inverse, get_LDU

    return init_fun, forward, inverse

def Affine( *args, mode='LDU', **kwargs ):
    if( mode == 'LDU' ):
        return AffineLDU( *args, **kwargs )
    else:
        assert 0, 'Invalid choice of affine backend'

def pinv_logdet( A ):
    """ Compute the det of a tall matrix

        Args:
            A - 2d array
    """
    ATA = A.T@A
    return -0.5*np.linalg.slogdet( ATA )[1]

def pinv( A ):
    """ Compute the pseudo inverse a tall matrix

        Args:
            A - 2d array
    """
    ATA = A.T@A
    return np.linalg.inv( ATA )@A.T

################################################################################################################

def OnebyOneConv( name=None ):
    """ Invertible 1x1 convolution.  Basically just a matrix multiplication over the channel dimension

        Args:
    """
    affine_forward, affine_inverse = None, None

    def init_fun( key, input_shape, condition_shape ):
        height, width, channel = input_shape

        nonlocal affine_forward, affine_inverse
        affine_init_fun, affine_forward, affine_inverse = AffineLDU()
        _, _, params, static_params = affine_init_fun( key, ( channel, ), condition_shape )
        return name, input_shape, params, static_params

    def forward( params, static_params, log_px, x, condition, **kwargs ):
        if( x.ndim == 4 ):
            # vmap over the batch dim
            z = vmap( partial( forward, params, static_params, **kwargs ), in_axes=( 0, 0, None ) )( log_px, x, condition )
            return z

        # need to vmap over the height and width axes
        assert x.ndim == 3
        over_width = vmap( partial( affine_forward, params, static_params, **kwargs ), in_axes=( None, 0, None ) )
        over_height_width = vmap( over_width, in_axes=( None, 0, None ) )

        # Not sure what to do about the updated static params in this case
        log_det, z, _ = over_height_width( 0, x, condition )
        return log_px + log_det.sum( axis=( -2, -1 ) ), z, static_params

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):
        if( z.ndim == 4 ):
            # vmap over the batch dim
            x = vmap( partial( inverse, params, static_params, **kwargs ), in_axes=( 0, 0, None ) )( log_pz, z, condition )
            return x

        # need to vmap over the height and width axes
        assert z.ndim == 3
        over_width = vmap( partial( affine_inverse, params, static_params, **kwargs ), in_axes=( None, 0, None ) )
        over_height_width = vmap( over_width, in_axes=( None, 0, None ) )

        # Not sure what to do about the updated static params in this case
        log_det, x, _ = over_height_width( 0, z, condition )
        return log_pz + log_det.sum( axis=( -2, -1 ) ), x, static_params

    return init_fun, forward, inverse

################################################################################################################

def LeakyReLU( alpha=0.01, name='unnamed' ):
    """ Invertible leaky relu

        Args:
            alpha - Slope of for negative inputs
    """

    def init_fun( key, input_shape, condition_shape ):
        return name, input_shape, (), ()

    def forward( params, static_params, log_px, x, condition, **kwargs ):
        z = np.where( x > 0, x, alpha*x )

        log_dx_dz = np.where( x > 0, 0, np.log( alpha ) )
        log_det = log_dx_dz.sum( axis=-1 )
        return log_px + log_det, z, ()

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):
        x = np.where( z > 0, z, z/alpha )

        log_dx_dz = np.where( z > 0, 0, np.log( alpha ) )
        log_det = log_dx_dz.sum( axis=-1 )

        return log_pz + log_det, x, ()

    return init_fun, forward, inverse

################################################################################################################

def Sigmoid( name=None ):
    """ Invertible sigmoid.  The logit function is its inverse.  Remember to apply sigmoid before logit so that
        the input ranges are as expected!

        Args:
    """
    def init_fun( key, input_shape, condition_shape ):
        return name, input_shape, (), ()

    def forward( params, static_params, log_px, x, condition, **kwargs ):
        z = jax.nn.sigmoid( x )
        log_det = -( jax.nn.softplus( x ) + jax.nn.softplus( -x ) ).sum( axis=-1 )
        if( log_det.ndim > 1 ):
            # Then we have an image and have to sum more
            log_det = log_det.sum( axis=( -2, -1 ) )
        return log_px + log_det, z, ()

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):
        z = np.maximum( 1e-6, z )
        z = np.minimum( 1-1e-6, z )
        x = jax.scipy.special.logit( z )
        log_det = -( jax.nn.softplus( x ) + jax.nn.softplus( -x ) ).sum( axis=-1 )
        if( log_det.ndim > 1 ):
            # Then we have an image and have to sum more
            log_det = log_det.sum( axis=( -2, -1 ) )
        return log_pz + log_det, x, ()

    return init_fun, forward, inverse

def Logit( name=None ):
    """ Invertible logit

        Args:
    """
    def init_fun( key, input_shape, condition_shape ):
        return name, input_shape, (), ()

    def forward( params, static_params, log_px, x, condition, **kwargs ):
        x = np.maximum( 1e-6, x )
        x = np.minimum( 1-1e-6, x )
        z = jax.scipy.special.logit( x )
        log_det = ( jax.nn.softplus( z ) + jax.nn.softplus( -z ) ).sum( axis=-1 )
        if( log_det.ndim > 1 ):
            # Then we have an image and have to sum more
            log_det = log_det.sum( axis=( -2, -1 ) )
        return log_px + log_det, z, ()

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):
        x = jax.nn.sigmoid( z )
        log_det = ( jax.nn.softplus( z ) + jax.nn.softplus( -z ) ).sum( axis=-1 )
        if( log_det.ndim > 1 ):
            # Then we have an image and have to sum more
            log_det = log_det.sum( axis=( -2, -1 ) )
        return log_pz + log_det, x, ()

    return init_fun, forward, inverse

################################################################################################################

def MAF( hidden_layer_sizes,
         reverse=False,
         method='sequential',
         key=None,
         name=None ):
    """ Masked Autoregressive Flow https://arxiv.org/pdf/1705.07057.pdf
        Invertible network that enforces autoregressive property.  Until I think of a cleaner way, the networks
        that make the MADE are defined there in order to make sure we keep the autoregressive property.

        Args:
            hidden_layer_sizes - A list of the size of the feature network.  MADE creates the network
            reverse            - Whether or not to reverse the inputs
            method             - Either 'sequential' or 'random'.  Controls how indices are assigned
                                 to nodes in each layer
            key                - JAX random key.  Only needed in random mode
    """
    made_apply_fun = None

    def init_fun( key, input_shape, condition_shape ):
        # Ugly, but saves us from initializing when calling function
        # Another design choice could be to define forward and inverse inside init_fun
        nonlocal made_apply_fun
        made_init_fun, made_apply_fun = GaussianMADE( input_shape[-1], hidden_layer_sizes, reverse=reverse, method=method, key=key, name=name )

        _, ( mu_shape, alpha_shape ), params, static_params = made_init_fun( key, input_shape )
        return name, input_shape, params, static_params

    def forward( params, static_params, log_px, x, condition, **kwargs ):
        ( mu, alpha ), updated_static_params = made_apply_fun( params, static_params, x, **kwargs )
        z = ( x - mu )*np.exp( -alpha )
        log_det = -alpha.sum( axis=-1 )
        return log_px + log_det, z, updated_static_params

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):
        # TODO: Speed this up with lax.while and fixed point iteration (and add vjps for each)
        x = np.zeros_like( z )
        u = z

        # Helper for item assignment
        idx_axis_0 = np.arange( x.shape[0] )

        # For each MADE block, need to build output a dimension at a time
        for i in range( 1, 1 + z.shape[-1] ):
            ( mu, alpha ), _ = made_apply_fun( params, static_params, u, **kwargs )
            w = mu + u*np.exp( alpha )
            x = index_update( x, [ idx_axis_0, np.ones_like( idx_axis_0 )*( i - 1 ) ], w[:,( i - 1 )] )

        # Do the backwards pass again to get the determinant term
        # Not sure if this is the correct way to update the static params though
        ( mu, alpha ), updated_static_params = made_apply_fun( params, static_params, x, **kwargs )
        log_det = -alpha.sum( axis=-1 )

        return log_pz + log_det, x, updated_static_params

    return init_fun, forward, inverse

################################################################################################################

def AffineCoupling( transform_fun, axis=-1, name='unnamed' ):
    """ Apply an arbitrary transform to half of the input vector

        Args:
            transform - A transformation that will act on half of the input vector.
                        Must return 2 vectors!!!
            axis      - Axis to split on
    """
    apply_fun = None
    reduce_axes = None

    def init_fun( key, input_shape, condition_shape ):
        ax = axis % len( input_shape )
        assert input_shape[-1]%2 == 0
        half_split_dim = input_shape[ax]//2

        # We need to keep track of the input shape in order to know how to reduce the log det
        nonlocal reduce_axes
        reduce_axes = tuple( range( -1, -( len( input_shape ) + 1 ), -1 ) )

        # Find the split shape
        split_input_shape = input_shape[:ax] + ( half_split_dim, ) + input_shape[ax + 1:]

        # Ugly, but saves us from initializing when calling function
        nonlocal apply_fun
        _init_fun, apply_fun = transform_fun( out_shape=split_input_shape )
        _, ( log_s_shape, t_shape ), params, static_params = _init_fun( key, split_input_shape )

        return name, input_shape, params, static_params

    def forward( params, static_params, log_px, x, condition, **kwargs ):

        xa, xb = np.split( x, 2, axis=axis )
        ( log_s, t ), updated_static_params = apply_fun( params, static_params, xb, **kwargs )

        za = xa*np.exp( log_s ) + t
        z = np.concatenate( [ za, xb ], axis=axis )

        log_det = np.sum( log_s, axis=reduce_axes )

        return log_px + log_det, z, updated_static_params

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):

        za, zb = np.split( z, 2, axis=axis )
        ( log_s, t ), updated_static_params = apply_fun( params, static_params, zb, **kwargs )

        xa = ( za - t )*np.exp( -log_s )
        x = np.concatenate( [ xa, zb ], axis=axis )

        log_det = np.sum( log_s, axis=reduce_axes )

        return log_pz + log_det, x, updated_static_params

    return init_fun, forward, inverse

def ConditionedAffineCoupling( transform_fun, axis=-1, name='unnamed' ):
    """ Apply an arbitrary transform to half of the input vector with conditioning

        Args:
            transform - A transformation that will act on half of the input vector.
                        Must return 2 vectors!!!
            axis      - Axis to split on
    """
    apply_fun = None
    reduce_axes = None

    def init_fun( key, input_shape, condition_shape ):
        ax = axis % len( input_shape )
        assert input_shape[-1]%2 == 0
        half_split_dim = input_shape[ax]//2

        # We need to keep track of the input shape in order to know how to reduce the log det
        nonlocal reduce_axes
        reduce_axes = tuple( range( -1, -( len( input_shape ) + 1 ), -1 ) )

        # Find the split shape
        split_input_shape = input_shape[:ax] + ( half_split_dim, ) + input_shape[ax + 1:]
        transform_input_shape = ( split_input_shape, ) + condition_shape

        # Ugly, but saves us from initializing when calling function
        nonlocal apply_fun
        _init_fun, apply_fun = transform_fun( out_shape=split_input_shape )
        _, ( log_s_shape, t_shape ), params, static_params = _init_fun( key, transform_input_shape )

        return name, input_shape, params, static_params

    def forward( params, static_params, log_px, x, condition, **kwargs ):

        xa, xb = np.split( x, 2, axis=axis )
        ( log_s, t ), updated_static_params = apply_fun( params, static_params, ( xb, *condition ), **kwargs )

        za = xa*np.exp( log_s ) + t
        z = np.concatenate( [ za, xb ], axis=axis )

        log_det = np.sum( log_s, axis=reduce_axes )

        return log_px + log_det, z, updated_static_params

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):

        za, zb = np.split( z, 2, axis=axis )
        ( log_s, t ), updated_static_params = apply_fun( params, static_params, ( zb, *condition ), **kwargs )

        xa = ( za - t )*np.exp( -log_s )
        x = np.concatenate( [ xa, zb ], axis=axis )

        log_det = np.sum( log_s, axis=reduce_axes )

        return log_pz + log_det, x, updated_static_params

    return init_fun, forward, inverse

################################################################################################################

def GLOW( transform_fun, name='unnamed' ):
    return sequential_flow( ActNorm( name='%s_act_norm'%name ),
                        OnebyOneConv( name='%s_one_by_one_conv'%name ),
                        AffineCoupling( transform_fun, name='%s_affine_coupling'%name ) )

def ConditionedGLOW( transform_fun, name='unnamed' ):
    return sequential_flow( ActNorm( name='%s_act_norm'%name ),
                        OnebyOneConv( name='%s_one_by_one_conv'%name ),
                        ConditionedAffineCoupling( transform_fun, name='%s_affine_coupling'%name ) )

################################################################################################################

fft_channel_vmap = vmap( np.fft.fftn, in_axes=( 2, ), out_axes=2 )
ifft_channel_vmap = vmap( np.fft.ifftn, in_axes=( 2, ), out_axes=2 )
fft_double_channel_vmap = vmap( fft_channel_vmap, in_axes=( 2, ), out_axes=2 )

inv_height_vmap = vmap( np.linalg.inv )
inv_height_width_vmap = vmap( inv_height_vmap )

slogdet_height_width_vmap = vmap( vmap( lambda x: np.linalg.slogdet( x )[1] ) )

def CircularConv( filter_size, W_init=glorot_normal(), name=None ):
    """ Convolve assuming full padding and we preserve the number of channels

        Args:
            filter_size - ( height, width ) of kernel
    """

    def init_fun( key, input_shape, condition_shape ):
        height, width, channel = input_shape
        W = W_init( key, filter_size + ( channel, channel ) )
        return name, input_shape, W, ()

    def forward( params, static_params, log_px, x, condition, **kwargs ):
        if( x.ndim == 4 ):
            return vmap( partial( forward, params, static_params, **kwargs ), in_axes=( 0, 0, None ) )( log_px, x, condition )

        kernel = params

        # http://developer.download.nvidia.com/compute/cuda/2_2/sdk/website/projects/convolutionFFT2D/doc/convolutionFFT2D.pdf
        x_h, x_w, x_c = x.shape
        kernel_h, kernel_w, kernel_c_out, kernel_c_in = kernel.shape

        # See how much we need to roll the kernel
        kernel_x = ( kernel_h - 1 ) // 2
        kernel_y = ( kernel_w - 1 ) // 2

        # Pad the kernel to match the fft size and roll it so that its center is at (0,0)
        kernel_padded = np.pad( kernel[::-1,::-1,:,:], ( ( 0, x_h - kernel_h ), ( 0, x_w - kernel_w ), ( 0, 0 ), ( 0, 0 ) ) )
        kernel_padded = np.roll( kernel_padded, ( -kernel_x, -kernel_y ), axis=( 0, 1 ) )

        # Apply the FFT to get the convolution
        image_fft = fft_channel_vmap( x )
        kernel_fft = fft_double_channel_vmap( kernel_padded )
        z_fft = np.einsum( 'abij,abj->abi', kernel_fft, image_fft )
        z = ifft_channel_vmap( z_fft ).real

        # The log determinant is the log det of the frequencies over the channel dims
        log_det = slogdet_height_width_vmap( kernel_fft ).sum()

        return log_px + log_det, z, ()

    def inverse( params, static_params, log_pz, z, condition, **kwargs ):
        if( z.ndim == 4 ):
            return vmap( partial( inverse, params, static_params, **kwargs ), in_axes=( 0, 0, None ) )( log_pz, z, condition )

        kernel = params

        z_h, z_w, z_c = z.shape
        kernel_h, kernel_w, kernel_c_out, kernel_c_in = kernel.shape

        # See how much we need to roll the kernel
        kernel_x = ( kernel_h - 1 ) // 2
        kernel_y = ( kernel_w - 1 ) // 2

        # Pad the kernel to match the fft size and roll it so that its center is at (0,0)
        kernel_padded = np.pad( kernel[::-1,::-1,:,:], ( ( 0, z_h - kernel_h ), ( 0, z_w - kernel_w ), ( 0, 0 ), ( 0, 0 ) ) )
        kernel_padded = np.roll( kernel_padded, ( -kernel_x, -kernel_y ), axis=( 0, 1 ) )

        # Apply the FFT to get the convolution
        image_fft = fft_channel_vmap( z )
        kernel_fft = fft_double_channel_vmap( kernel_padded )

        # For deconv, we need to invert the kernel over the channel dims
        kernel_fft_inv = inv_height_width_vmap( kernel_fft )

        x_fft = np.einsum( 'abij,abj->abi', kernel_fft_inv, image_fft )
        x = ifft_channel_vmap( x_fft ).real

        # The log determinant is the log det of the frequencies over the channel dims
        log_det = slogdet_height_width_vmap( kernel_fft ).sum()

        return log_pz + log_det, x, ()

    return init_fun, forward, inverse
