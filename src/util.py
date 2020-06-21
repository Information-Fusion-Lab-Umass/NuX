import os
import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from jax.tree_util import tree_flatten, tree_unflatten
import jax
import pickle
import haiku as hk
import pathlib

TEST = jnp.ones((0, 0))
TRAIN = jnp.ones((0,))

def is_testing(x):
    return x.ndim == 2

# TEST = True
# TRAIN = False

# def is_testing(x):
#     return x == TEST

################################################################################################################

@jit
def tree_shapes(pytree):
    return jax.tree_util.tree_map(lambda x:x.shape, pytree)

@jit
def tree_ndims(pytree):
    return jax.tree_util.tree_map(lambda x:x.ndim, pytree)

################################################################################################################

class SimpleMLP(hk.Module):

    def __init__(self, out_shape, hidden_layer_sizes, is_additive, name=None):
        super().__init__(name=name)
        assert len(out_shape) == 1
        self.out_dim = out_shape[0]
        self.hidden_layer_sizes = hidden_layer_sizes
        self.is_additive = is_additive

    def __call__(self, x, **kwargs):
        for dim in self.hidden_layer_sizes:
            x = hk.Linear(dim)(x)
        mu = hk.Linear(self.out_dim)(x)
        if(self.is_additive):
            return mu
        alpha = hk.Linear(self.out_dim)(x)
        alpha = jnp.tanh(alpha)
        return mu, alpha

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

        x = hk.Conv2D(output_channels=self.n_hidden_channels, kernel_shape=(3, 3), stride=(1, 1))(x[None])[0]
        x = jax.nn.relu(x)
        x = hk.Conv2D(output_channels=self.n_hidden_channels, kernel_shape=(1, 1), stride=(1, 1))(x[None])[0]
        x = jax.nn.relu(x)
        x = hk.Conv2D(output_channels=self.last_channels, kernel_shape=(3, 3), stride=(1, 1))(x[None])[0]

        if(self.is_additive):
            return x

        mu, alpha = jnp.split(x, 2, axis=-1)
        alpha = jnp.tanh(alpha)
        return mu, alpha

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

@jit
def tall_affine_posterior_diag_cov(x, b, A, log_diag_cov, sigma):
    """ Sample from N(z|mu(x), Sigma(x)) and compute h(x).
        Args:
    """
    # In case we want to change the noise model.  This equation corresponds
    # to how we are changing noise in the inverse section
    log_diag_cov = log_diag_cov + 2*jnp.log(sigma)

    diag_cov = jnp.exp(log_diag_cov)
    xminusb = x - b

    # Find the pseudo inverse and the projection
    ATSA = A.T/diag_cov@A
    ATSA_inv = jnp.linalg.inv(ATSA)
    z = jnp.dot(xminusb, (ATSA_inv@A.T/diag_cov).T)
    x_proj = jnp.dot(z, A.T)/diag_cov

    # Get the terms that don't depend on z
    dim_x, dim_z = A.shape
    log_hx = -0.5*jnp.sum(xminusb*(xminusb/diag_cov - x_proj), axis=-1)
    log_hx -= 0.5*jnp.linalg.slogdet(ATSA)[1]
    log_hx -= 0.5*log_diag_cov.sum()
    log_hx -= 0.5*(dim_x - dim_z)*jnp.log(2*jnp.pi)

    # For sampling
    sigma_ATA_chol = jnp.linalg.cholesky(ATSA_inv)

    return z, log_hx, sigma_ATA_chol

@partial(jit, static_argnums=(3,))
def upsample_posterior(x, b, log_diag_cov, repeats):
    """ Posterior of N(x|Az + b, Sigma) where A is an upsample matrix"""
    assert x.shape == b.shape
    assert x.shape == log_diag_cov.shape
    assert x.ndim == 3
    xmb = x - b
    one_over_diag_cov = jnp.exp(-log_diag_cov)

    # Compute the diagonal of the riemannian metric.  This is the diagonal of A^T Sigma^{-1} A
    hr, wr, cr = repeats; assert cr == 1 # Haven't tested cr != 1
    Hx, Wx, C = x.shape
    H, W = Hx//hr, Wx//wr
    rm_diag = one_over_diag_cov.reshape((H, hr, W, wr, C)).transpose((0, 2, 4, 1, 3)).reshape((H, W, C, hr*wr)).sum(axis=-1)

    # Compute the mean of z
    z_mean = upsample_pseudo_inverse(xmb*one_over_diag_cov, (2, 2, 1))/rm_diag*(hr*wr)
    x_proj = upsample(repeats, z_mean)*one_over_diag_cov
    dim_x = jnp.prod(x.shape)
    dim_z = jnp.prod(z_mean.shape)

    # Compute the manifold error term
    log_hx = -0.5*jnp.sum(xmb*(xmb*one_over_diag_cov - x_proj))
    log_hx -= 0.5*jnp.sum(jnp.log(rm_diag))
    log_hx -= 0.5*log_diag_cov.sum()
    log_hx -= 0.5*(dim_x - dim_z)*jnp.log(2*jnp.pi)

    return z_mean, log_hx, rm_diag

################################################################################################################

def indexer_and_shape_from_mask(mask):
    # language=rst
    """
    Given a 2d mask array, create an array that can index into a vector with the same number of elements
    as nonzero elements in mask and result in an array of the same size as mask, but with the elements
    specified from the vector. Also return the shape of the resulting array when mask is applied.

    :param mask: 2d boolean mask array
    """
    index = jnp.zeros_like(mask, dtype=int)
    non_zero_indices = jnp.nonzero(mask)
    index[non_zero_indices] = jnp.arange(len(non_zero_indices[0])) + 1

    nonzero_x, nonzero_y = non_zero_indices
    n_rows = jnp.unique(nonzero_x).size
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
    if(jnp.any(mask) == False):
        assert 0, 'Empty mask!  Reduce num'
    if(jnp.sum(mask)%2 == 1):
        assert 0, 'Need masks with an even number!  Choose a different num'

def checkerboard_masks(num, shape):
    # language=rst
    """
    Finds masks to factor an array with a given shape so that each pixel will be
    present in the resulting masks.  Also return indices that will help reverse
    the factorization.

    :param masks: A list of 2d boolean mask array whose union is equal to jnp.ones(shape, dtype=bool)
    :param indices: A list of index matrices that undo the application of image[mask]
    """
    masks = []
    indices = []
    shapes = []

    for i in range(2*num):
        start = 2**i
        step = 2**(i + 1)

        # Staggered indices
        mask = jnp.zeros(shape, dtype=bool)
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
        mask = jnp.zeros(shape, dtype=bool)
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

def upsample(repeats, z):
    x = z
    is_batched = int(x.ndim == 2 or x.ndim == 4)
    for i, r in enumerate(repeats):
        x = jnp.repeat(x, r, axis=i + is_batched)
    return x

def downsample(repeats, x):
    return x[[slice(0, None, r) for r in repeats]]

def upsample_pseudo_inverse(x, repeats):
    # language=rst
    """
    Compute the pseudo inverse of an upsample
    """
    hr, wr, cr = repeats
    assert cr == 1
    Hx, Wx, C = x.shape
    assert Hx%hr == 0 and Wx%wr == 0
    H, W = Hx//hr, Wx//wr

    return x.reshape((H, hr, W, wr, C)).transpose((0, 2, 4, 1, 3)).reshape((H, W, C, hr*wr)).mean(axis=-1)

def upsample_idx(repeats, idx):
    repeats = (repeats[0], repeats[1])
    for i, r in enumerate(repeats):
        idx = jnp.repeat(idx, r, axis=i)

    idx = jnp.array(idx)

    k = 1
    for i in range(idx.shape[0]):
        for j in range(idx.shape[1]):
            if(idx[i,j] >= 1):
                idx[i,j] = k
                k += 1
    return idx

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

def save_np_array_to_file(np_array, path):
    np.savetxt(path, np_array, delimiter=",")

def save_pytree_to_file(pytree, path):
    """ Save a pytree to file in pickle format"""
    dir_structure, file_name = os.path.split(path)
    assert file_name.endswith('.npz')

    # Create the path if it doesn't exist
    pathlib.Path(dir_structure).mkdir(parents=True, exist_ok=True)

    # Save the raw numpy parameters
    flat_pytree, _ = ravel_pytree(pytree)
    numpy_tree = np.array(flat_pytree)

    # Save the array to an npz file
    np.savez_compressed(path, flat_tree=numpy_tree)

def load_pytree_from_file(pytree, path):
    assert os.path.exists(path), '%s does not exist!'%path

    # Load the pytree structure
    _, unflatten = ravel_pytree(pytree)

    with np.load(path) as data:
        numpy_tree = data['flat_tree']

    return unflatten(numpy_tree)

################################################################################################################

def gaussian_chol_cov_logpdf(x, mean, cov_chol):
    pass

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

################################################################################################################

@jit
def upper_cho_solve(chol, x):
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
