import numpy as onp
import jax.numpy as np
from jax import grad, jit, vmap, random, jacobian
from jax.scipy.special import logsumexp
import itertools
from functools import partial
import jax
from jax.experimental import stax
from functools import reduce
from jax.flatten_util import ravel_pytree

TEST = np.ones((0, 0))
TRAIN = np.ones((0,))

@jit
def is_testing(x):
    return x.ndim == 2

################################################################################################################

def gaussian_logpdf(x, mean, cov):
    dx = x - mean
    cov_inv = np.linalg.inv(cov)
    log_px = -0.5*np.sum(np.dot(dx, cov_inv.T)*dx, axis=-1)
    return log_px - 0.5*np.linalg.slogdet(cov)[1] - 0.5*x.shape[-1]*np.log(2*np.pi)

def gaussian_diag_cov_logpdf(x, mean, log_diag_cov):
    dx = x - mean
    log_px = -0.5*np.sum(dx*np.exp(-log_diag_cov)*dx, axis=-1)
    return log_px - 0.5*np.sum(log_diag_cov) - 0.5*x.shape[-1]*np.log(2*np.pi)

################################################################################################################

@jit
def upper_cho_solve(chol, x):
    return jax.scipy.linalg.cho_solve((chol, True), x)

def upper_triangular_indices(N):
    values = np.arange(N)
    padded_values = np.hstack([values, 0])

    idx = onp.ogrid[:N,N:0:-1]
    idx = sum(idx) - 1

    mask = np.arange(N) >= np.arange(N)[:,None]
    return (idx + np.cumsum(values + 1)[:,None][::-1] - N + 1)*mask

def n_elts_upper_triangular(N):
    return N*(N + 1) // 2 - 1

def upper_triangular_from_values(vals, N):
    assert n_elts_upper_triangular(N) == vals.shape[-1]
    zero_padded_vals = np.pad(vals, (1, 0))
    return zero_padded_vals[upper_triangular_indices(N)]

tri_solve = jax.scipy.linalg.solve_triangular
L_solve = jit(partial(tri_solve, lower=True, unit_diagonal=True))
U_solve = jit(partial(tri_solve, lower=False, unit_diagonal=True))

################################################################################################################

@jit
def householder(x, v):
    return x - 2*np.einsum('i,j,j', v, v, x)/np.sum(v**2)

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

################################################################################################################

def slogsumexp(x, signs, axis=None):
    """ logsumexp with signs (b argument for logsumexp isn't implemented in JAX!)
    """
    x_max = np.amax(x, axis=axis, keepdims=True)
    y = np.sum(signs*np.exp(x - x_max), axis=axis)
    sign_y = np.sign(y)
    abs_y = np.log(np.abs(y))
    return sign_y, abs_y + np.squeeze(x_max, axis=axis)

################################################################################################################

def hvp(scalar_loss, z, v):
    return jax.jvp(grad(scalar_loss), [z], [v])

@partial(jit, static_argnums=(1,))
def cond_fun(max_iters, inputs):
    x, r, p, r_sq, index = inputs
    ans = (r_sq > 1e-3).astype(np.int32) * (index < max_iters).astype(np.int32)
    return ans == 1

@partial(jit, static_argnums=(0,))
def conjugate_gradient_while_body(A_eval, inputs):
    x, r, p, r_sq, index = inputs

    A_eval_p = A_eval(p)

    alpha = r_sq / np.dot(p.ravel(), A_eval_p.ravel())
    x += alpha*p
    r -= alpha*A_eval_p
    r_sq_new = np.dot(r.ravel(), r.ravel())

    beta = r_sq_new / r_sq
    p = r + beta*p
    return x, r, p, r_sq_new, index + 1

@partial(jit, static_argnums=(1, 2))
def CTC_eval(kernel, pad, stride, image):
    Cz = fft_conv(kernel, image, pad, stride)
    return fft_conv_transpose(kernel, Cz, pad, stride)

@partial(jit, static_argnums=(1, 2, 3))
def CTC_solve(kernel, pad, stride, max_iters, b, initial_guess):
    filled_CTC_eval = jit(partial(CTC_eval, kernel, pad, stride))
    filled_while_body = jit(partial(conjugate_gradient_while_body, filled_CTC_eval))
    filled_cond = jit(partial(cond_fun, max_iters))

    x = initial_guess
    r = b - filled_CTC_eval(x)
    p = r
    r_sq = np.dot(r.ravel(), r.ravel())

    reconstruction, _, _, r_sq, iters = lax.while_loop(filled_cond, filled_while_body, (x, r, p, r_sq, 0.0))

    return reconstruction, r_sq, iters

# b = fft_conv_transpose(kernel, convolved_image, pad, stride)
# reconstruction, r_sq, iters = CTC_solve(kernel, pad, stride, 100, b, np.zeros_like(b))

################################################################################################################

def conv_transpose_output_shape(H, W, filter_shape, stride, pad):
    out_shape0 = (H - 1)*stride[0] - filter_shape[0] + pad[0][0] + pad[0][1] + 2
    out_shape1 = (W - 1)*stride[1] - filter_shape[1] + pad[1][0] + pad[1][1] + 2
    return out_shape0, out_shape1

################################################################################################################

@partial(jit, static_argnums=(1, 2))
def dilate(x, n_cols, n_rows):

    # Insert columns of zeros
    if(n_cols == 0):
        with_cols = x
    else:
        zero_cols = np.zeros(x.shape + (n_cols,))
        with_cols = np.concatenate([x[...,None], zero_cols], axis=-1).reshape(x.shape[0], -1)[:,:-n_cols]

    # Insert rows of zeros
    if(n_rows == 0):
        return with_cols
    zero_rows = np.zeros((with_cols.shape[0], n_rows*with_cols.shape[1]))
    return np.hstack((with_cols, zero_rows)).reshape(-1, with_cols.shape[1])[:-n_rows]

@partial(jit, static_argnums=(2,))
def _fft_2d_conv(kernel, image, pad):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    top_pad, bottom_pad, left_pad, right_pad = pad

    # Need to pad the kernel and image so that they are the same size.
    # Also need to flip the kernel because a ML convolution uses different conventions
    kernel_vertical_pad = (0, image_height + bottom_pad + top_pad - 1)
    kernel_horizontal_pad = (0, image_width + right_pad + left_pad - 1)
    kernel_padded = np.pad(kernel[::-1,::-1], (kernel_vertical_pad, kernel_horizontal_pad))

    # Pad the image slightly differently so that we get the same effect as padding the image only
    image_vertical_pad = (top_pad, kernel_height + bottom_pad - 1)
    image_horizontal_pad = (left_pad, kernel_width + right_pad - 1)
    image_padded = np.pad(image, (image_vertical_pad, image_horizontal_pad))

    # Apply the FFT to get the convolution
    kernel_fft = np.fft.fftn(kernel_padded)
    image_fft = np.fft.fftn(image_padded)
    ans = np.real(np.fft.ifftn(kernel_fft*image_fft))

    return ans

@partial(jit, static_argnums=(2, 3))
def fft_2d_conv(kernel, image, pad, stride):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    top_pad, bottom_pad, left_pad, right_pad = pad
    vertical_stride, horizontal_stride = stride

    ans = _fft_2d_conv(kernel, image, pad)

    # Trim the output and take the stride into account
    ans = ans[kernel_height-1:-kernel_height+1:vertical_stride,kernel_width-1:-kernel_width+1:horizontal_stride]
    return ans

@partial(jit, static_argnums=(2, 3))
def fft_image_conv(kernel, image, pad, stride):
    # Must have the same channel dimension!
    channel_image, _, _ = image.shape
    channel_kernel, _, _ = kernel.shape
    assert channel_image == channel_kernel

    # Perform the convolution along each channel dimension
    conv = vmap(fft_2d_conv, in_axes=(0, 0, None, None), out_axes=0)(kernel, image, pad, stride)

    # Sum across channel dimension
    return conv.sum(axis=0)

@partial(jit, static_argnums=(2, 3))
def fft_conv(kernel, image, pad, stride):
    if(image.ndim == 4):
        # Then we have a batch dimension
        return vmap(fft_conv, in_axes=(None, 0, None, None), out_axes=0)(kernel, image, pad, stride)

    in_channel, out_channel, kernel_height, kernel_width = kernel.shape
    in_channel, image_height, image_width = image.shape

    return vmap(fft_image_conv, in_axes=(1, None, None, None), out_axes=0)(kernel, image, pad, stride)

@partial(jit, static_argnums=(2, 3))
def fft_2d_conv_transpose(kernel, image, pad, stride):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    top_pad, bottom_pad, left_pad, right_pad = pad
    vertical_stride, horizontal_stride = stride

    # Need to flip the kernel and dilate the input to account for strides
    kernel_transpose = kernel[::-1,::-1]
    dilated_image = dilate(image, horizontal_stride - 1, vertical_stride - 1)

    # Perform the convolution with 1x1 strides
    ans = _fft_2d_conv(kernel_transpose, dilated_image, pad)

    # Account for padding
    ans = ans[2*top_pad:ans.shape[0] - 2*bottom_pad,2*left_pad:ans.shape[1] - 2*right_pad]

    return ans

@partial(jit, static_argnums=(2, 3))
def fft_image_conv_transpose(kernel, image, pad, stride):
    # Must have the same channel dimension!
    channel_image, _, _ = image.shape
    channel_kernel, _, _ = kernel.shape
    assert channel_image == channel_kernel

    # Perform the convolution along each channel dimension
    conv = vmap(fft_2d_conv_transpose, in_axes=(0, 0, None, None), out_axes=0)(kernel, image, pad, stride)

    # Sum across channel dimension
    return conv.sum(axis=0)

@partial(jit, static_argnums=(2, 3))
def fft_conv_transpose(kernel, image, pad, stride):
    if(image.ndim == 4):
        # Then we have a batch dimension
        return vmap(fft_conv_transpose, in_axes=(None, 0, None, None), out_axes=0)(kernel, image, pad, stride)

    swapped_kernel = kernel.transpose((1, 0, 2, 3))
    in_channel, out_channel, kernel_height, kernel_width = swapped_kernel.shape
    in_channel, image_height, image_width = image.shape

    return vmap(fft_image_conv_transpose, in_axes=(1, None, None, None), out_axes=0)(swapped_kernel, image, pad, stride)

################################################################################################################

def latex_mat(H, W, pad=0, letter='x'):
    index = 0
    mat_str = ''
    for i in range(-pad, H + pad):
        for j in range(-pad, W + pad):

            if(i >= 0 and i < H and j >= 0 and j < W):
                mat_str += '%s_{%d}'%(letter, index)
                index += 1
            else:
                mat_str += '-'

            if(j < W + pad - 1):
                mat_str += '&'

        if(i < H + pad - 1):
            mat_str += '\\\\'

    mat_str = '\\begin{bmatrix}%s\\end{bmatrix}'%(mat_str)
    return mat_str

def latex_image(H, W, pad=0):
    return latex_mat(H, W, pad=pad, letter='x')

def latex_kernel(H, W):
    return latex_mat(H, W, pad=0, letter='k')

# from IPython.display import display, Math, Latex
# display(Math('%s*%s'%(image_str, kernel_str)))

def latex_hstack(mats):
    rows = list(zip(*[mat.split('\\\\') for mat in mats]))
    combined_rows = ['&'.join(row) for row in rows]
    return '\\\\'.join(combined_rows)

def latex_vstack(mats):
    return '\\\\'.join(mats)

def unflatten_latex(flat_mat, shape):
    assert np.prod(shape) == len(flat_mat)
    H, W = shape
    letter_iter = iter(flat_mat)
    mat_str = ''

    rows = []
    for i in range(H):
        row = []
        for j in range(W):
            row.append(next(letter_iter))

        rows.append(latex_hstack(row))
    return latex_vstack(rows)

def latex_block(image_width, kernel_row_letters, left_pad=0, right_pad=0, horizontal_stride=1, empty='-'):
    # This function has to do with an individual row of the kernel and image

    # Assume perfect padding and stride of 1, then crop and select rows
    first_row = kernel_row_letters[-1:] + [empty]*(image_width - 1)
    first_col = kernel_row_letters[::-1] + [empty]*(len(first_row) - 1)
    N = len(kernel_row_letters)

    # Create a toeplitz matrix
    vec = first_row[-1:0:-1] + first_col
    indices = sum(onp.ogrid[:len(first_col),len(first_row)-1:-1:-1])

    # Keep track of which rows to crop
    max_pad = len(kernel_row_letters) - 1
    assert left_pad <= max_pad, 'Too much left padding!  Max is %d'%(max_pad)
    assert right_pad <= max_pad, 'Too much right padding!  Max is %d'%(max_pad)

    remove_head = max_pad - left_pad
    remove_tail = max_pad - right_pad
    remove_pad = remove_head + remove_tail
    width_with_pad = indices.shape[0] - remove_pad

    # Keep track of how many rows to remove due to striding
    assert horizontal_stride != 0
    assert horizontal_stride < width_with_pad, 'Stride is too large.  Max is %d'%(width_with_pad - 1)
    remove_stride = width_with_pad - onp.ceil(width_with_pad / horizontal_stride).astype(int)
    width_with_pad_stride = width_with_pad - remove_stride

    # Compute the output shape.  Need to take into account stride and padding!
    output_shape = (width_with_pad_stride, indices.shape[1])

    flat_mat = []
    for i, index_row in enumerate(indices):
        # Skip rows due to padding
        if(i < remove_head or i >= len(indices) - remove_tail):
            continue

        # Skip rows due to stride
        if(i%horizontal_stride != 0):
            continue

        for idx in index_row:
            flat_mat.append(vec[idx])

    full_mat = unflatten_latex(flat_mat, output_shape)
    return full_mat

def latex_conv_matrix(image_shape, kernel_shape, pad=(0, 0, 0, 0), stride=(1, 1)):
    # Write out a convolution as a matrix product in latex

    image_height, image_width = image_shape
    kernel_height, kernel_width = kernel_shape
    kernel_letters = ['k_{%d}'%(i) for i in range(kernel_height*kernel_width)]

    # padding is (top, bottom, left, right)
    filled_block_call = lambda row: latex_block(image_width,
                                                 row,
                                                 pad[2],
                                                 pad[3],
                                                 stride[1])
    # Get the unique blocks in the matrix
    blocks = []
    for i in range(kernel_height):
        kernel_row_letters = [kernel_letters[i*kernel_width + j] for j in range(kernel_width)]
        blocks.append(filled_block_call(kernel_row_letters))


    # Construct the full matrix correctly
    # THIS IS THE EXACT SAME LOGIC AS GETTING THE UNIQUE BLOCKS!!!!
    # Also make sure the empty blocks are the correct size
    n_rows = len(blocks[0].split('\\\\'))
    n_cols = len(blocks[0].split('\\\\')[0].split('&'))
    empty_row = '&'.join(['-' for _ in range(n_cols)])
    empty_mat = '\\\\'.join([empty_row for _ in range(n_rows)])

    return latex_block(image_height, blocks, pad[0], pad[1], stride[0], empty=empty_mat)

def numpy_block(image_width, kernel_row, left_pad=0, right_pad=0, horizontal_stride=1):

    # Define the full Toeplitz matrix assuming perfect padding and a stride of 1
    row_zero_shape = (image_width - 1,) + kernel_row.shape[1:]
    first_row = np.concatenate([kernel_row[:1,...], np.zeros(row_zero_shape)], axis=0)
    col_zero_shape = (first_row.shape[0] - 1,) + kernel_row.shape[1:]
    first_col = np.concatenate([kernel_row[::-1,...], np.zeros(col_zero_shape)], axis=0)

    # Create the components for the full Toeplitz matrix
    toeplitz_vec = np.concatenate([first_row[-1:0:-1,...], first_col], axis=0)
    indices = sum(onp.ogrid[:first_col.shape[0],first_row.shape[0]-1:-1:-1])

    # Keep track of which rows to crop
    max_pad = kernel_row.shape[0] - 1
    assert left_pad <= max_pad, 'Too much left padding!  Max is %d'%(max_pad)
    assert right_pad <= max_pad, 'Too much right padding!  Max is %d'%(max_pad)

    remove_head = max_pad - left_pad
    remove_tail = max_pad - right_pad
    remove_pad = remove_head + remove_tail
    width_with_pad = indices.shape[0] - remove_pad

    # Keep track of how many rows to remove due to striding
    assert horizontal_stride != 0
    assert horizontal_stride < width_with_pad, 'Stride is too large.  Max is %d'%(width_with_pad - 1)
    remove_stride = width_with_pad - onp.ceil(width_with_pad / horizontal_stride).astype(int)
    width_with_pad_stride = width_with_pad - remove_stride

    trimmed_indices = indices[remove_head:indices.shape[0]-remove_tail:horizontal_stride]
    return toeplitz_vec[trimmed_indices]

def numpy_conv_matrix(kernel, image_shape, pad=(0, 0, 0, 0), stride=(1, 1)):

    top_pad, bottom_pad, left_pad, right_pad = pad
    vertical_stride, horizontal_stride = stride
    image_height, image_width = image_shape
    kernel_height, kernel_width = kernel.shape

    # Get the unique blocks in the matrix
    blocks = []
    for i in range(kernel_height):
        block = numpy_block(image_width, kernel[i], pad[2], pad[3], stride[1])
        blocks.append(block)

    blocks = np.array(blocks)

    # Run the same algorithm over blocks.  The output shape will be (out_height, in_height, out_width, in_width)
    ans = numpy_block(image_height, blocks, pad[0], pad[1], stride[0])

    # Reshape to (out_height, out_width, in_height, in_width)
    ans = ans.transpose((0, 2, 1, 3))

    # Compute the expected output shape
    expected_height = (image_height - kernel_height + top_pad + bottom_pad)/vertical_stride + 1
    expected_width = (image_width - kernel_width + left_pad + right_pad)/horizontal_stride + 1

    assert ans.shape[0] == expected_height
    assert ans.shape[1] == expected_width

    # Flatten the output
    return ans.reshape((ans.shape[0]*ans.shape[1], ans.shape[2]*ans.shape[3]))

################################################################################################################

@jit
def symmetric_toeplitz(row):
    # Create an index matrix that is symmetric toeplitz
    N = row.shape[0]
    a, b = onp.ogrid[:N,N - 1:-1:-1]
    index = np.triu(a + b) + np.triu(a + b, k=1).T
    index = N - index - 1

    # Index into the first row to create the rest of the matrix
    return row[index]

def symmetric_block_toeplitz_indices(size_of_blocks, number_of_blocks):
    # Get the indices for each block
    index = symmetric_toeplitz_indices(size_of_blocks)

    # Create the first blocked row
    offset = np.repeat(np.arange(number_of_blocks), size_of_blocks)*size_of_blocks
    first_blocked_row = np.tile(index, number_of_blocks) + offset

    # Create the full matrix by keeping the Toeplitz property
    full_indices = np.vstack([np.roll(first_blocked_row, size_of_blocks*i, axis=1) for i in range(number_of_blocks)])

    # Make the matrix symmetric
    indices = np.triu(full_indices) + np.triu(full_indices, k=1).T
    return indices

def matrix_conv(w, z):
    w_row = np.pad(w[:1], (0, z.shape[-1] - 1))
    w_col = np.pad(w, (0, z.shape[-1] - 1))
    W = scipy.linalg.toeplitz(w_col, w_row)
    if(z.ndim == 1):
        z = z[None]
    return W, np.einsum('ij,bj->bi', W, z).squeeze()

def CTC_first_row(w, z_shape):
    # same as onp.convolve(w[::-1], np.pad(w, (0, z.shape[-1] - 1)), mode='valid')
    w_row = np.pad(w[:1], (0, z_shape[-1] - 1))
    w_col = np.pad(w, (0, z_shape[-1] - 1))
    W = scipy.linalg.toeplitz(w_col, w_row)
    return W.T.dot(w_col)

@jit
def conv_1d(w, z):
    # x = C@z
    if(z.ndim == 2):
        return vmap(conv_1d, in_axes=(None, 0))(w, z)
    assert z.ndim == 1
    w_padded = np.pad(w, (0, z.shape[-1] - 1))
    z_padded = np.pad(z, (0, w.shape[-1] - 1))
    return np.real(np.fft.ifftn(np.fft.fftn(w_padded)*np.fft.fftn(z_padded)))

@jit
def conv_1d_transpose(w, x):
    # z' = C.T@x
    len_z = x.shape[-1] - w.shape[-1] + 1
    padded_conv = conv_1d(w[::-1], x)
    if(padded_conv.ndim == 1):
        return padded_conv[w.shape[-1] - 1:w.shape[-1] + len_z - 1]
    return padded_conv[:,w.shape[-1] - 1:w.shape[-1] + len_z - 1]

@jit
def deconv_1d(w, x):
    len_z = x.shape[-1] - w.shape[-1] + 1
    w_padded = np.pad(w, (0, len_z - 1))
    return np.real(np.fft.ifftn(np.fft.fftn(x)/np.fft.fftn(w_padded)))[:len_z]

@jit
def fft_CTC_first_row(w, z):
    # First row of C^T@C
    row = conv_1d(w[::-1], w)[w.shape[-1] - 1:]
    return np.pad(row, (0, z.shape[-1] - w.shape[-1]))

def toeplitz_prod(first_row, z):
    # first_row is the first row of the toeplitz matrix
    circulant_row = np.hstack([first_row[0], first_row[1:-1], first_row[1:][::-1]])
    return circular_conv(np.pad(z, (0, circulant_row.shape[-1] - z.shape[-1])), circulant_row)[:z.shape[-1]]

def conv_projection(w, x):
    # Solve z for x = w*z
    len_z = x.shape[-1] - w.shape[-1] + 1
    z_prime = conv_transpose(w, x)
    first_row = fft_CTC_first_row(w)
    matvec = scipy.sparse.linalg.LinearOperator((len_z, len_z), matvec=partial(toeplitz_prod, first_row))
    z_proj, success = scipy.sparse.linalg.cg(matvec, z_prime)
    return z_proj, success

def symmetric_toeplitz_solve(c_first_row, x):
    def toeplitz_prod(first_row, z):
        # first_row is the first row of the toeplitz matrix
        circulant_row = np.hstack([first_row[0], first_row[1:-1], first_row[1:][::-1]])
        return circular_conv(np.pad(z, (0, circulant_row.shape[-1] - z.shape[-1])), circulant_row)[:z.shape[-1]]

    matvec = scipy.sparse.linalg.LinearOperator((x.shape[-1], x.shape[-1]), matvec=partial(toeplitz_prod, first_row))
    z_proj, success = scipy.sparse.linalg.cg(matvec, x)
    return z_proj
