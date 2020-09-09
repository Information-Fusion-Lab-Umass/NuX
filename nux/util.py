import os
import numpy as np
import jax.numpy as jnp
from jax import jit, random
from functools import partial
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from jax.tree_util import tree_flatten, tree_unflatten
import jax
import pickle
import haiku as hk
import pathlib

################################################################################################################

@jit
def xTAx(A, x):
    return jnp.einsum('i,ij,j', x, A, x)

################################################################################################################

def linear_warmup_lr_schedule(i, warmup=1000, lr_decay=1.0, lr=1e-4):
    return jnp.where(i < warmup,
                     lr*i/warmup,
                     lr*(lr_decay**(i - warmup)))

################################################################################################################

TEST = jnp.ones((0, 0))
TRAIN = jnp.ones((0,))

def is_testing(x):
    return x.ndim == 2

################################################################################################################

def key_tree_like(key, pytree):
    # Figure out what the tree structure is
    flat_tree, treedef = jax.tree_util.tree_flatten(pytree)

    # Generate a tree of keys with the same structure as pytree
    n_keys = len(flat_tree)
    keys = random.split(key, n_keys)
    key_tree = jax.tree_util.tree_unflatten(treedef, keys)
    return key_tree

# @partial(jit, static_argnums=(0,))
def tree_multimap_multiout(f, tree, *rest):
    # Like tree_multimap but expects f(leaves) to return a tuple.
    # This function will return trees for each tuple element.
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in rest]
    new_leaves = [f(*xs) for xs in zip(*all_leaves)]
    return [treedef.unflatten(leaf) for leaf in zip(*new_leaves)]

@jit
def tree_shapes(pytree):
    return jax.tree_util.tree_map(lambda x:x.shape, pytree)

@jit
def tree_ndims(pytree):
    return jax.tree_util.tree_map(lambda x:x.ndim, pytree)

################################################################################################################

@jit
def whiten(x):
    U, s, VT = jnp.linalg.svd(x, full_matrices=False)
    return jnp.dot(U, VT)

################################################################################################################

def weight_norm(x):
    return x*jax.lax.rsqrt(jnp.sum(x**2, axis=0) + 1e-5)

class SimpleMLP(hk.Module):

    def __init__(self, out_shape, hidden_layer_sizes, is_additive, weight_norm=True, name=None):
        super().__init__(name=name)
        assert len(out_shape) == 1
        self.out_dim = out_shape[0]
        self.hidden_layer_sizes = hidden_layer_sizes
        self.is_additive = is_additive
        self.weight_norm = weight_norm

    def __call__(self, x, **kwargs):

        w_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal')

        for i, output_size in enumerate(self.hidden_layer_sizes):
            input_size = x.shape[-1]
            w = hk.get_parameter(f"w_{i}", [output_size, input_size], init=w_init)
            if(self.weight_norm):
                w = weight_norm(w)

            b = hk.get_parameter(f"b_{i}", [output_size], init=jnp.zeros)
            x = w@x + b

            x = jax.nn.swish(x)/1.1

        w_mu = hk.get_parameter("w_mu", [self.out_dim, x.shape[-1]], init=w_init)
        if(self.weight_norm):
            w_mu = weight_norm(w_mu)
        b_mu = hk.get_parameter("b_mu", [self.out_dim], init=jnp.zeros)
        mu = w_mu@x + b_mu

        if(self.is_additive):
            return mu

        w_alpha = hk.get_parameter("w_alpha", [self.out_dim, x.shape[-1]], init=w_init)
        if(self.weight_norm):
            w_alpha = weight_norm(w_alpha)
        b_alpha = hk.get_parameter("b_alpha", [self.out_dim], init=jnp.zeros)
        alpha = w_alpha@x + b_alpha

        alpha = jnp.tanh(alpha)
        return mu, alpha

class WeightNormConv(hk.Module):

    def __init__(self,
                 out_channels,
                 kernel_shape,
                 stride=(1, 1),
                 padding='SAME',
                 lhs_dilation=(1, 1),
                 rhs_dilation=(1, 1),
                 w_init=None,
                 b_init=jnp.zeros,
                 name=None):
        super().__init__(name=name)
        self.out_channels      = out_channels
        self.kernel_shape      = kernel_shape
        self.stride            = stride
        self.padding           = padding
        self.lhs_dilation      = lhs_dilation
        self.rhs_dilation   = rhs_dilation
        self.w_init            = w_init
        self.b_init            = b_init
        self.dimension_numbers = ('NHWC', 'HWIO', 'NHWC')

    def __call__(self, x, **kwargs):
        in_channels = x.shape[-1]

        w_shape = self.kernel_shape + (in_channels, self.out_channels)
        w = hk.get_parameter("w", w_shape, x.dtype, init=self.w_init)
        w = jax.vmap(jax.vmap(weight_norm))(w)

        out = jax.lax.conv_general_dilated(x,
                                           w,
                                           window_strides=self.stride,
                                           padding=self.padding,
                                           lhs_dilation=self.lhs_dilation,
                                           rhs_dilation=self.rhs_dilation,
                                           dimension_numbers=self.dimension_numbers)

        b = hk.get_parameter("b", (self.out_channels,), x.dtype, init=self.b_init)
        b = jnp.broadcast_to(b, out.shape)
        out = out + b
        return out

class SimpleConv(hk.Module):

    def __init__(self, out_shape, n_hidden_channels, is_additive, name=None):
        super().__init__(name=name)
        _, _, out_channels = out_shape
        self.out_channels = out_channels
        self.n_hidden_channels = n_hidden_channels
        self.is_additive = is_additive

        self.last_channels = out_channels if is_additive else 2*out_channels

    def __call__(self, x, **kwargs):
        H, W, C = x.shape

        x = WeightNormConv(out_channels=self.n_hidden_channels,
                           kernel_shape=(3, 3),
                           stride=(1, 1),
                           w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal'))(x[None])[0]

        x = jax.nn.relu(x)

        x = WeightNormConv(out_channels=self.n_hidden_channels,
                           kernel_shape=(1, 1),
                           stride=(1, 1),
                           w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal'))(x[None])[0]

        x = jax.nn.relu(x)

        x = WeightNormConv(out_channels=self.last_channels,
                           kernel_shape=(3, 3),
                           stride=(1, 1),
                           w_init=hk.initializers.Constant(0),
                           b_init=hk.initializers.Constant(0))(x[None])[0]

        if(self.is_additive):
            return x

        mu, alpha = jnp.split(x, 2, axis=-1)
        alpha = jnp.tanh(alpha)
        return mu, alpha

################################################################################################################

def dilated_squeeze(x, filter_shape, dilation):
    H, W, C = x.shape

    fh, fw = filter_shape
    dh, dw = dilation

    assert H%(dh*fh) == 0
    assert W%(dw*fw) == 0

    # Rearrange for dilation
    x = x.reshape((H//dh, dh, W//dw, dw, C))
    x = x.transpose((1, 0, 3, 2, 4)) # (dh, H//dh, dw, W//dw, C)

    # Squeeze
    x = x.reshape((H//fh, fh, W//fw, fw, C))
    x = x.transpose((0, 2, 1, 3, 4)) # (H//fh, W//fw, fh, fw, C)
    x = x.reshape((H//fh, W//fw, C*fh*fw))
    return x

def dilated_unsqueeze(x, filter_shape, dilation):

    fh, fw = filter_shape
    dh, dw = dilation

    H_in, W_in, C_in = x.shape
    assert C_in%(fh*fw) == 0

    H, W, C = H_in*fh, W_in*fw, C_in//(fh*fw)

    assert H%(dh*fh) == 0
    assert W%(dw*fw) == 0

    # Un-squeeze
    x = x.reshape((H_in, W_in, fh, fw, C))
    x = x.transpose((0, 2, 1, 3, 4))

    # Un-dilate
    x = x.reshape((dh, H//dh, dw, W//dw, C))
    x = x.transpose((1, 0, 3, 2, 4))
    x = x.reshape((H, W, C))

    return x

################################################################################################################

def scaled_logsumexp(x, log_b, axis=0):
    """ logsumexp with scaling
    """
    x_max = jnp.amax(log_b + x, axis=axis, keepdims=True)
    y = jnp.sum(jnp.exp(log_b + x - x_max), axis=axis)
    sign_y = jnp.sign(y)
    abs_y = jnp.log(jnp.abs(y))
    return abs_y + jnp.squeeze(x_max, axis=axis)

################################################################################################################

@partial(jit, static_argnums=(0,))
def replicate(shape, pytree):
    replicate_fun = lambda x: jnp.broadcast_to(x, shape + x.shape)
    return tree_map(replicate_fun, pytree)

@jit
def unreplicate(pytree):
    return tree_map(lambda x:x[0], pytree)

################################################################################################################
# Thanks! https://github.com/google/jax/issues/2116#issuecomment-580322624
from jax.tree_util import pytree
import pickle
from pathlib import Path
from typing import Union

suffix = '.pickle'

def save_pytree(data: pytree, path: Union[str, Path], overwrite: bool = False):
    path = Path(path)
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(data, file)

def load_pytree(path: Union[str, Path]) -> pytree:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    if path.suffix != suffix:
        raise ValueError(f'Not a {suffix} file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def save_np_array_to_file(np_array, path):
    np.savetxt(path, np_array, delimiter=",")

################################################################################################################

@jit
def gaussian_chol_cov_logpdf(x, mean, cov_chol):
    dx = x - mean
    y = jax.lax_linalg.triangular_solve(cov_chol, dx, lower=True, transpose_a=True)
    log_px = -0.5*jnp.sum(y**2) - jnp.log(jnp.diag(cov_chol)).sum() - 0.5*x.shape[0]*jnp.log(2*jnp.pi)
    return log_px

@jit
def gaussian_full_cov_logpdf(x, mean, cov):
    dx = x - mean
    cov_inv = jnp.linalg.inv(cov)
    log_px = -0.5*jnp.sum(jnp.dot(dx, cov_inv.T)*dx, axis=-1)
    return log_px - 0.5*jnp.linalg.slogdet(cov)[1] - 0.5*x.shape[-1]*jnp.log(2*jnp.pi)

@jit
def gaussian_diag_cov_logpdf(x, mean, log_diag_cov):
    dx = x - mean
    log_px = -0.5*jnp.sum(dx*jnp.exp(-log_diag_cov)*dx, axis=-1)
    return log_px - 0.5*jnp.sum(log_diag_cov) - 0.5*x.shape[-1]*jnp.log(2*jnp.pi)

@jit
def unit_gaussian_logpdf(x):
    if(x.ndim > 1):
        return jax.vmap(unit_gaussian_logpdf)(x)
    return -0.5*jnp.dot(x, x) - 0.5*x.shape[-1]*jnp.log(2*jnp.pi)

################################################################################################################

@jit
def lower_cho_solve(chol, x):
    return jax.scipy.linalg.cho_solve((chol, True), x)

def upper_triangular_indices(N):
    values = jnp.arange(N)
    padded_values = jnp.hstack([values, 0])

    idx = np.ogrid[:N,N:0:-1]
    idx = sum(idx) - 1

    mask = jnp.arange(N) >= jnp.arange(N)[:,None]
    return (idx + jnp.cumsum(values + 1)[:,None][::-1] - N + 1)*mask

def n_elts_upper_triangular(N):
    return N*(N + 1) // 2 - 1

def upper_triangular_from_values(vals, N):
    assert n_elts_upper_triangular(N) == vals.shape[-1]
    zero_padded_vals = jnp.pad(vals, (1, 0))
    return zero_padded_vals[upper_triangular_indices(N)]

tri_solve = jax.scipy.linalg.solve_triangular
L_solve = jit(partial(tri_solve, lower=True, unit_diagonal=True))
U_solve = jit(partial(tri_solve, lower=False, unit_diagonal=True))

################################################################################################################

@jit
def householder(x, v):
    return x - 2*jnp.einsum('i,j,j', v, v, x)/jnp.sum(v**2)

@jit
def householder_prod_body(carry, inputs):
    x = carry
    v = inputs
    return householder(x, v), 0

@jit
def householder_prod(x, vs):
    return jax.lax.scan(householder_prod_body, x, vs)[0]

@jit
def householder_prod_transpose(x, vs):
    return jax.lax.scan(householder_prod_body, x, vs[::-1])[0]

@jit
def householder_apply(U, log_s, VT, z):
    # Compute Az
    x = householder_prod(z, VT)
    x = x*jnp.exp(log_s)
    x = jnp.pad(x, (0, U.shape[1] - z.shape[0]))
    x = householder_prod(x, U)
    return x

@jit
def householder_pinv_apply(U, log_s, VT, x):
    # Compute A^+@x and also return U_perp^T@x
    UTx = householder_prod_transpose(x, U)
    z, UperpTx = jnp.split(UTx, jnp.array([log_s.shape[0]]))
    z = z*jnp.exp(-log_s)
    z = householder_prod_transpose(z, VT)
    return z, UperpTx

@jit
def householder_to_dense(U, log_s, VT):
    return jax.vmap(partial(householder_apply, U, log_s, VT))(jnp.eye(VT.shape[0])).T

@jit
def householder_pinv_to_dense(U, log_s, VT):
    return jax.vmap(partial(householder_pinv_apply, U, log_s, VT))(jnp.eye(U.shape[0]))[0].T

################################################################################################################

# def LowDimInputConvBlock(n_channels=512, init_zeros=True, name='unnamed'):
#     # language=rst
#     """
#     A conv block where we assume the number of input channels and output channels are small
#     """
#     _apply_fun = None

#     def init_fun(key, input_shape):
#         H, W, C = input_shape
#         if(init_zeros):
#             W_init, b_init = zeros, zeros
#         else:
#             W_init, b_init = glorot_normal(), normal()

#         nonlocal _apply_fun
#         _init_fun, _apply_fun = sequential(Conv(n_channels, filter_shape=(3, 3), padding=((1, 1), (1, 1)), bias=True, weightnorm=False),
#                                            LayerNormSimple(),
#                                            Relu(),
#                                            Conv(n_channels, filter_shape=(1, 1), padding=((0, 0), (0, 0)), bias=True, weightnorm=False),
#                                            LayerNormSimple(),
#                                            Relu(),
#                                            Conv(C, filter_shape=(3, 3), padding=((1, 1), (1, 1)), bias=True, weightnorm=False, W_init=W_init, b_init=b_init))
#         name, output_shape, params, state = _init_fun(key, input_shape)
#         return name, output_shape, params, state

#     def apply_fun(params, state, inputs, **kwargs):
#         return _apply_fun(params, state, inputs, **kwargs)

#     return init_fun, apply_fun

# def DoubledLowDimInputConvBlock(n_channels=512, init_zeros=True, name='unnamed'):
#     # language=rst
#     """
#     A conv block where we assume the number of input channels and output channels are small
#     """
#     _apply_fun = None

#     def init_fun(key, input_shape):
#         H, W, C = input_shape
#         if(init_zeros):
#             W_init, b_init = zeros, zeros
#         else:
#             W_init, b_init = glorot_normal(), normal()

#         nonlocal _apply_fun
#         _init_fun, _apply_fun = sequential(Conv(n_channels, filter_shape=(3, 3), padding=((1, 1), (1, 1)), bias=True, weightnorm=False),
#                                            LayerNormSimple(),
#                                            Relu(),
#                                            Conv(n_channels, filter_shape=(1, 1), padding=((0, 0), (0, 0)), bias=True, weightnorm=False),
#                                            LayerNormSimple(),
#                                            Relu(),
#                                            Conv(2*C, filter_shape=(3, 3), padding=((1, 1), (1, 1)), bias=True, weightnorm=False, W_init=W_init, b_init=b_init),
#                                            Split(2, axis=-1),
#                                            parallel(Tanh(), Identity()))
#         name, output_shape, params, state = _init_fun(key, input_shape)
#         return name, output_shape, params, state

#     def apply_fun(params, state, inputs, **kwargs):
#         return _apply_fun(params, state, inputs, **kwargs)

#     return init_fun, apply_fun

# ################################################################################################################

# def SqueezeExcitation(ratio=2, W1_init=glorot_normal(), W2_init=glorot_normal(), name='unnamed'):
#     # language=rst
#     """
#     https://arxiv.org/pdf/1709.01507.pdf

#     :param ratio: How to reduce the number of channels for the FC layer
#     """
#     def init_fun(key, input_shape):
#         H, W, C = input_shape
#         assert C%ratio == 0
#         k1, k2 = random.split(key, 2)
#         W1 = W1_init(k1, (C//ratio, C))
#         W2 = W2_init(k2, (C, C//ratio))
#         output_shape = input_shape
#         params = (W1, W2)
#         state = ()
#         return name, output_shape, params, state

#     def apply_fun(params, state, inputs, **kwargs):
#         W1, W2 = params

#         # Apply the SE transforms
#         x = np.mean(inputs, axis=(-2, -3))
#         x = np.dot(x, W1.T)
#         x = jax.nn.relu(x)
#         x = np.dot(x, W2.T)
#         x = jax.nn.sigmoid(x)

#         # Scale the input
#         if(x.ndim == 3):
#             out = inputs*x[None, None,:]
#         else:
#             out = inputs*x[:,None,None,:]
#         return out, state

#     return init_fun, apply_fun

# ################################################################################################################

# def ConditionedSqueezeExcitation(ratio=4, W_cond_init=glorot_normal(), W1_init=glorot_normal(), W2_init=glorot_normal(), name='unnamed'):
#     # language=rst
#     """
#     Like squeeze excitation, but has an extra input to help form W
#     PURPOSE IS TO FIGURE OUT WHICH FEATURE MAPS MATTER GIVEN A CONDITIONER

#     :param ratio: How to reduce the number of channels for the FC layer
#     """
#     def init_fun(key, input_shape):
#         (H, W, C), (K,) = input_shape
#         k1, k2, k3 = random.split(key, 3)

#         # Will be shrinking the conditioner down to the size of the number of channels
#         W_cond = W_cond_init(k1, (C, K))

#         # Going to be concatenating the conditioner
#         C_concat = C + C
#         assert C_concat%ratio == 0

#         # Create the parameters for the squeeze and excite
#         W1 = W1_init(k2, (C_concat//ratio, C_concat))
#         W2 = W2_init(k3, (C, C_concat//ratio))

#         output_shape = (H, W, C)
#         params = (W_cond, W1, W2)
#         state = ()
#         return name, output_shape, params, state

#     def apply_fun(params, state, inputs, **kwargs):
#         W_cond, W1, W2 = params
#         inputs, cond = inputs

#         # Apply the SE transforms
#         x = np.mean(inputs, axis=(-2, -3))
#         x = np.concatenate([x, np.dot(cond, W_cond.T)], axis=-1)
#         x = np.dot(x, W1.T)
#         x = jax.nn.relu(x)
#         x = np.dot(x, W2.T)
#         x = jax.nn.sigmoid(x)

#         # Scale the input
#         if(x.ndim == 3):
#             out = inputs*x[None, None,:]
#         else:
#             out = inputs*x[:,None,None,:]
#         return out, state

#     return init_fun, apply_fun
