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
import nux.spectral_norm as sn

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
