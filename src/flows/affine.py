import jax
import jax.numpy as jnp
import jax.nn.initializers as jaxinit
import src.util as util
from jax import random, vmap
from functools import partial
import src.flows.base as base

@base.auto_batch
def AffineLDU(L_init=jaxinit.normal(), d_init=jaxinit.ones, U_init=jaxinit.normal(), name='affine_ldu'):
    # language=rst
    """
    LDU parametrized square dense matrix
    """
    triangular_indices = None

    def get_LDU(params):
        L_flat, d, U_flat = params['L_flat'], params['d'], params['U_flat']

        L = jnp.pad(L_flat, (1, 0))[triangular_indices]
        L = L + jnp.eye(d.shape[-1])
        L = L.T

        U = jnp.pad(U_flat, (1, 0))[triangular_indices]
        U = U + jnp.eye(d.shape[-1])

        return L, d, U

    def forward(params, state, inputs, **kwargs):
        L, d, U = get_LDU(params)
        x = inputs['x']

        z = jnp.einsum('ij,j->i', U, x)
        z = z*d
        z = jnp.einsum('ij,j->i', L, z)

        log_det = jnp.sum(jnp.log(jnp.abs(d)), axis=-1)

        inputs['x'] = z
        inputs['log_det'] = log_det

        return inputs, state

    def inverse(params, state, inputs, **kwargs):
        L, d, U = get_LDU(params)
        z = inputs['x']

        x = util.L_solve(L, z)
        x = x/d
        x = util.U_solve(U, x)

        log_det = jnp.sum(jnp.log(jnp.abs(d)), axis=-1)

        inputs['x'] = x
        inputs['log_det'] = log_det

        return inputs, state

    def init_fun(key, input_shapes):
        # This is a square matrix!
        dim = input_shapes['x'][-1]

        # Create the fancy indices that we'll use to turn our vectors into triangular matrices
        nonlocal triangular_indices
        triangular_indices = jnp.pad(util.upper_triangular_indices(dim - 1), ((0, 1), (1, 0)))
        flat_dim = util.n_elts_upper_triangular(dim)

        k1, k2, k3 = random.split(key, 3)
        L_flat, d, U_flat = L_init(k1, (flat_dim,)), d_init(k2, (dim,)), U_init(k3, (flat_dim,))

        params = {'L_flat': L_flat, 
                  'd'     : d,
                  'U_flat': U_flat}
        state = {}
        output_shapes = {}
        output_shapes.update(input_shapes)
        output_shapes['log_det'] = (1,)

        return base.Flow(name, input_shapes, output_shapes, params, state, forward, inverse)

    return init_fun, base.data_independent_init(init_fun)

@base.auto_batch
def AffineSVD(n_householders, U_init=jaxinit.glorot_normal(), log_s_init=jaxinit.normal(), VT_init=jaxinit.glorot_normal(), name='affine_svd'):
    # language=rst
    """
    Affine matrix with SVD parametrization.  Uses a product of householders to parametrize the orthogonal matrices.

    :param n_householders: Number of householders to use in U and V parametrization.  When n_householders = dim(x),
                           we can represent any orthogonal matrix with det=-1 (I think?)
    """

    def forward(params, state, inputs, **kwargs):
        U, log_s, VT = params['U'], params['log_s'], params['VT']
        x = inputs['x']
        z = util.householder_prod(x, VT)
        z = z*jnp.exp(log_s)
        z = util.householder_prod(z, U)
        log_det = log_s.sum()
        inputs['x'] = z
        inputs['log_det'] = log_det
        return inputs, state

    def inverse(params, state, inputs, **kwargs):
        U, log_s, VT = params['U'], params['log_s'], params['VT']
        z = inputs['x']
        x = util.householder_prod_transpose(z, U)
        x = x*jnp.exp(-log_s)
        x = util.householder_prod_transpose(x, VT)
        log_det = log_s.sum()
        inputs['x'] = x
        inputs['log_det'] = log_det
        return inputs, state

    def init_fun(key, input_shapes):
        keys = random.split(key, 3)
        x_shape = input_shapes['x']
        U = U_init(keys[0], (n_householders, x_shape[-1]))
        log_s = log_s_init(keys[1], (x_shape[-1],))
        VT = VT_init(keys[2], (n_householders, x_shape[-1]))
        params = {'U'    : U, 
                  'log_s': log_s,
                  'VT'   : VT}
        state = {}
        output_shapes = {}
        output_shapes.update(input_shapes)
        output_shapes['log_det'] = (1,)
        return base.Flow(name, input_shapes, output_shapes, params, state, forward, inverse)

    return init_fun, base.data_independent_init(init_fun)

@base.auto_batch
def AffineDense(W_init=jaxinit.glorot_normal(), name='affine_dense'):
    # language=rst
    """
    Basic affine transformation
    """

    def forward(params, state, inputs, **kwargs):
        x = inputs['x']
        W = params['W']
        W = W/jnp.sqrt(jnp.sum(W**2, axis=0) + 1e-5)
        z = jnp.dot(x, W.T)
        log_det = jnp.linalg.slogdet(W)[1]
        inputs['x'] = z
        inputs['log_det'] = log_det
        return inputs, state

    def inverse(params, state, inputs, **kwargs):
        z = inputs['x']
        W = params['W']
        W = W/jnp.sqrt(jnp.sum(W**2, axis=0) + 1e-5)
        W_inv = jnp.linalg.inv(W)
        x = jnp.dot(z, W_inv.T)
        log_det = jnp.linalg.slogdet(W)[1]
        inputs['x'] = x
        inputs['log_det'] = log_det
        return inputs, state

    def init_fun(key, input_shapes):
        x_shape = input_shapes['x']
        W = W_init(key, (x_shape[-1], x_shape[-1]))
        U, s, VT = jnp.linalg.svd(W, full_matrices=False)
        W = jnp.dot(U, VT)
        params = {'W': W}
        state = {}
        output_shapes = {}
        output_shapes.update(input_shapes)
        output_shapes['log_det'] = (1,)

        return base.Flow(name, input_shapes, output_shapes, params, state, forward, inverse)



    return init_fun, base.data_independent_init(init_fun)

def Affine(*args, mode='dense', **kwargs):
    # language=rst
    """
    Affine matrix with choice of parametrization

    :param mode: Name of parametrization choice
    """
    if(mode == 'LDU'):
        return AffineLDU(*args, **kwargs)
    elif(mode == 'SVD'):
        return AffineSVD(*args, **kwargs)
    elif(mode == 'dense'):
        return AffineDense(*args, **kwargs)
    else:
        assert 0, 'Invalid choice of affine backend'

################################################################################################################

@base.auto_batch
def OnebyOneConvLDU(name='one_by_one_conv_ldu'):
    # language=rst
    """
    Invertible 1x1 convolution.  Implemented as matrix multiplication over the channel dimension
    """
    affine_forward, affine_inverse = None, None

    def forward(params, state, inputs, **kwargs):
        x = inputs['x']

        # need to vmap over the height and width axes
        assert x.ndim == 3
        over_width = vmap(partial(affine_forward, params, state, **kwargs))
        over_height_width = vmap(over_width)

        # Not sure what to do about the updated state in this case
        over_height_width_outputs, _ = over_height_width({'x': x})
        inputs['x'] = over_height_width_outputs['x']
        inputs['log_det'] = over_height_width_outputs['log_det'].sum(axis=(-2, -1))
        return inputs, state

    def inverse(params, state, inputs, **kwargs):
        z = inputs['x']

        # need to vmap over the height and width axes
        assert z.ndim == 3
        over_width = vmap(partial(affine_inverse, params, state, **kwargs))
        over_height_width = vmap(over_width)

        # Not sure what to do about the updated state in this case
        over_height_width_outputs, _ = over_height_width({'x': z})
        inputs['x'] = over_height_width_outputs['x']
        inputs['log_det'] = over_height_width_outputs['log_det'].sum(axis=(-2, -1))
        return inputs, state

    def init_fun(key, input_shapes):
        height, width, channel = input_shapes['x']

        nonlocal affine_forward, affine_inverse
        affine_init_fun, _ = AffineLDU()
        LDU_flow = affine_init_fun(key, {'x' : (channel,)})
        affine_forward = LDU_flow.forward
        affine_inverse = LDU_flow.inverse
        params = LDU_flow.params
        state  = LDU_flow.state
        output_shapes = {}
        output_shapes.update(input_shapes)
        output_shapes['log_det'] = (1,)
        return base.Flow(name, input_shapes, output_shapes, params, state, forward, inverse)

    return init_fun, base.data_independent_init(init_fun)

@base.auto_batch
def OnebyOneConv(W_init=jaxinit.glorot_normal(), name='one_by_one_conv'):
    # language=rst
    """
    Invertible 1x1 convolution.  Implemented as matrix multiplication over the channel dimension.
    THIS IS VERY UNSTABLE FOR SOME REASON
    """

    def forward(params, state, inputs, **kwargs):
        x = inputs['x']
        W, = params['W']
        log_det = jnp.linalg.slogdet(W)[1]
        height, width, channel = x.shape[-3], x.shape[-2], x.shape[-1]
        assert channel == W.shape[0]

        if(x.ndim == 4):
            z = jnp.einsum('ij,bhwj->bhwi', W, x)
        else:
            z = jnp.einsum('ij,hwj->hwi', W, x)

        log_det *= height*width

        inputs['log_det'] = log_det
        inputs['x'] = z

        return inputs, state

    def inverse(params, state, inputs, **kwargs):
        z = inputs['x']
        W, = params['W']
        W_inv = jnp.linalg.inv(W)
        log_det = jnp.linalg.slogdet(W)[1]
        height, width, channel = z.shape[-3], z.shape[-2], z.shape[-1]
        assert channel == W.shape[0]

        if(z.ndim == 4):
            # x = vmap(vmap(vmap(partial(jnp.linalg.solve, W))))(z)
            x = jnp.einsum('ij,bhwj->bhwi', W_inv, z)
        else:
            # x = vmap(vmap(partial(jnp.linalg.solve, W)))(z)
            x = jnp.einsum('ij,hwj->hwi', W_inv, z)

        log_det *= height*width

        inputs['log_det'] = log_det
        inputs['x'] = x

        return inputs, state

    def init_fun(key, input_shapes):
        height, width, channel = input_shapes['x']

        W = W_init(key, (channel, channel))
        # U, s, VT = jnp.linalg.svd(W, full_matrices=False)
        # W = jnp.dot(U, VT)

        params = {'W': (W,)}
        state = {}
        output_shapes = {}
        output_shapes.update(input_shapes)
        output_shapes['log_det'] = (1,)
        return base.Flow(name, input_shapes, output_shapes, params, state, forward, inverse)

    return init_fun, base.data_independent_init(init_fun)

@base.auto_batch
def OnebyOneConvLAX(W_init=jaxinit.glorot_normal(), name='unnamed'):
    # language=rst
    """
    Invertible 1x1 convolution.
    """
    filter_shape = (1, 1)
    padding = 'SAME'
    strides = (1, 1)
    one = (1,) * len(filter_shape)
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')

    def forward(params, state, inputs, **kwargs):
        x = inputs['x']
        W, = params['W']
        height, width, channel = x.shape[-3], x.shape[-2], x.shape[-1]


        W = W/jnp.sqrt(jnp.sum(W**2, axis=0) + 1e-5)
        log_det = jnp.linalg.slogdet(W)[1]*height*width
        assert channel == W.shape[0]

        batched = True
        if(x.ndim == 3):
            batched = False
            x = x[None]

        z = jax.lax.conv_general_dilated(x, W[None,None,...], strides, padding, (1, 1), (1, 1), dimension_numbers=dimension_numbers)

        if(batched == False):
            z = z[0]

        inputs['log_det'] = log_det
        inputs['x'] = z

        return inputs, state

    def inverse(params, state, inputs, **kwargs):
        z = inputs['x']
        W, = params['W']
        height, width, channel = z.shape[-3], z.shape[-2], z.shape[-1]

        W = W/jnp.sqrt(jnp.sum(W**2, axis=0) + 1e-5)
        W_inv = jnp.linalg.inv(W)
        log_det = jnp.linalg.slogdet(W)[1]*height*width
        assert channel == W.shape[0]

        batched = True
        if(z.ndim == 3):
            batched = False
            z = z[None]

        x = jax.lax.conv_general_dilated(z, W_inv[None,None,...], strides, padding, (1, 1), (1, 1), dimension_numbers=dimension_numbers)

        if(batched == False):
            x = x[0]

        inputs['log_det'] = log_det
        inputs['x'] = x

        return inputs, state

    def init_fun(key, input_shapes):
        height, width, channel = input_shapes['x']

        W = W_init(key, (channel, channel))
        U, s, VT = jnp.linalg.svd(W, full_matrices=False)
        W = jnp.dot(U, VT)

        # JAX conv is weird with batch dims
        assert len(input_shapes['x']) == 3
        input_shape = (1,) + input_shapes['x']
        filter_shape_iter = iter(filter_shape)
        lhs_spec, rhs_spec, out_spec = dimension_numbers
        kernel_shape = [channel if c == 'O' else input_shape[lhs_spec.index('C')] if c == 'I' else next(filter_shape_iter) for c in rhs_spec]
        output_shape = jax.lax.conv_general_shape_tuple(input_shape, kernel_shape, strides, padding, dimension_numbers)
        output_shape = output_shape[1:]

        assert output_shape == (height, width, channel)

        params = {'W': (W,)}
        state = {}
        output_shapes = {'x': output_shape,
                         'log_det': (1,)}

        return base.Flow(name, input_shapes, output_shapes, params, state, forward, inverse)

    return init_fun, base.data_independent_init(init_fun)

################################################################################################################

@base.auto_batch
def LocalDense(filter_shape=(2, 2), dilation=(1, 1), W_init=jaxinit.glorot_normal(), name='unnamed'):
    # language=rst
    """
    Dense matrix that gets multiplied by partitioned sections of an image.
    Works by applying dilated_squeeze, 1x1 conv, then undilated_squeeze.

    """
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')

    def forward(params, state, inputs, **kwargs):
        x = inputs['x']
        W, = params['W']

        # Normalize
        # W = W/jnp.sqrt(jnp.sum(W**2, axis=0) + 1e-5)

        # See if this is batched
        batched = True
        if(x.ndim == 3):
            batched = False
            x = x[None]

        dil_sq = vmap(util.dilated_squeeze, in_axes=(0, None, None))
        dil_unsq = vmap(util.dilated_unsqueeze, in_axes=(0, None, None))

        # dilate and squeeze the input
        x = dil_sq(x, filter_shape, dilation)
        h, w, c = x.shape[-3], x.shape[-2], x.shape[-1]

        # 1x1 convolution
        z = jax.lax.conv_general_dilated(x, W[None,None,...], (1, 1), 'SAME', (1, 1), (1, 1), dimension_numbers=dimension_numbers)

        # Undo the dilate and squeeze
        z = dil_unsq(z, filter_shape, dilation)
        z = z if batched else z[0]

        # Compute the log determinant
        log_det = jnp.linalg.slogdet(W)[1]*h*w

        inputs['log_det'] = log_det
        inputs['x'] = z

        return inputs, state

    def inverse(params, state, inputs, **kwargs):
        z = inputs['x']
        W, = params['W']

        # Normalize
        # W = W/jnp.sqrt(jnp.sum(W**2, axis=0) + 1e-5)

        # Invert
        W_inv = jnp.linalg.inv(W)

        # See if this is batched
        batched = True
        if(z.ndim == 3):
            batched = False
            z = z[None]
        dil_sq = vmap(util.dilated_squeeze, in_axes=(0, None, None))
        dil_unsq = vmap(util.dilated_unsqueeze, in_axes=(0, None, None))

        # dilate and squeeze the input
        z = dil_sq(z, filter_shape, dilation)
        h, w, c = z.shape[-3], z.shape[-2], z.shape[-1]

        x = jax.lax.conv_general_dilated(z, W_inv[None,None,...], (1, 1), 'SAME', (1, 1), (1, 1), dimension_numbers=dimension_numbers)

        # Undo the dilate and squeeze
        x = dil_unsq(x, filter_shape, dilation)
        x = x if batched else x[0]

        # Compute the log determinant
        log_det = jnp.linalg.slogdet(W)[1]*h*w

        inputs['log_det'] = log_det
        inputs['x'] = x

        return inputs, state

    def init_fun(key, input_shapes):
        x_shape = input_shapes['x']
        h, w, c = x_shape
        fh, fw = filter_shape
        dh, dw = dilation

        # Find the shape of the dilated_squeeze output
        H_sq, W_sq, C_sq = (h//fh, w//fw, c*fh*fw)

        # Create the matrix we're going to use in the conv.
        W = W_init(key, (C_sq, C_sq))
        # U, s, VT = jnp.linalg.svd(W, full_matrices=False)
        # W = jnp.dot(U, VT)
        W = jnp.eye(C_sq)

        # Ensure the convolution shapes will work
        input_shape = (1, H_sq, W_sq, C_sq)
        filter_shape_iter = iter((1, 1))
        lhs_spec, rhs_spec, out_spec = dimension_numbers
        kernel_shape = [C_sq if c == 'O' else input_shape[lhs_spec.index('C')] if c == 'I' else next(filter_shape_iter) for c in rhs_spec]
        output_shape = jax.lax.conv_general_shape_tuple(input_shape, kernel_shape, (1, 1), 'SAME', dimension_numbers)
        output_shape = output_shape[1:]
        assert output_shape == (H_sq, W_sq, C_sq)

        params = {'W': (W,)}
        state = {}
        output_shapes = {'x': (h,w,c),
                         'log_det': (1,)}

        return base.Flow(name, input_shapes, output_shapes, params, state, forward, inverse)

    return init_fun, base.data_independent_init(init_fun)

################################################################################################################

__all__ = ['AffineLDU',
           'AffineSVD',
           'AffineDense',
           'Affine',
           'OnebyOneConvLDU',
           'OnebyOneConv',
           'OnebyOneConvLAX',
           'LocalDense']
