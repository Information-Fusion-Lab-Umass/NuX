import jax.numpy as jnp
import jax.nn.initializers as jaxinit
from jax import vmap, jit
from functools import partial
import nux.flows.base as base
import nux.util as util

fft_channel_vmap = vmap(jnp.fft.fftn, in_axes=(2,), out_axes=2)
ifft_channel_vmap = vmap(jnp.fft.ifftn, in_axes=(2,), out_axes=2)
fft_double_channel_vmap = vmap(fft_channel_vmap, in_axes=(2,), out_axes=2)

inv_height_vmap = vmap(jnp.linalg.inv)
inv_height_width_vmap = vmap(inv_height_vmap)

@jit
def complex_slogdet(x):
    D = jnp.block([[x.real, -x.imag], [x.imag, x.real]])
    return 0.25*jnp.linalg.slogdet(D@D.T)[1]
slogdet_height_width_vmap = jit(vmap(vmap(complex_slogdet)))

@base.auto_batch
def CircularConv(filter_size, kernel_init=jaxinit.glorot_normal(), name='circular_conv'):
    # language=rst
    """
    Invertible circular convolution

    :param filter_size: (height, width) of kernel
    """
    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        x = inputs['x']
        kernel = params['kernel']

        # http://developer.download.nvidia.com/compute/cuda/2_2/sdk/website/projects/convolutionFFT2D/doc/convolutionFFT2D.pdf
        x_h, x_w, x_c = x.shape
        kernel_h, kernel_w, kernel_c_out, kernel_c_in = kernel.shape

        # See how much we need to roll the kernel
        kernel_x = (kernel_h - 1) // 2
        kernel_y = (kernel_w - 1) // 2

        # Pad the kernel to match the fft size and roll it so that its center is at (0,0)
        kernel_padded = jnp.pad(kernel[::-1,::-1,:,:], ((0, x_h - kernel_h), (0, x_w - kernel_w), (0, 0), (0, 0)))
        kernel_padded = jnp.roll(kernel_padded, (-kernel_x, -kernel_y), axis=(0, 1))

        # Apply the FFT to get the convolution
        image_fft = fft_channel_vmap(x)
        kernel_fft = fft_double_channel_vmap(kernel_padded)

        if(reverse == True):
            z_fft = jnp.einsum('abij,abj->abi', kernel_fft, image_fft)
            z = ifft_channel_vmap(z_fft).real
        else:
            # For deconv, we need to invert the kernel over the channel dims
            kernel_fft_inv = inv_height_width_vmap(kernel_fft)

            x_fft = jnp.einsum('abij,abj->abi', kernel_fft_inv, image_fft)
            z = ifft_channel_vmap(x_fft).real

        # The log determinant is the log det of the frequencies over the channel dims
        log_det = -slogdet_height_width_vmap(kernel_fft).sum()

        outputs = {'x': z, 'log_det': log_det}
        return outputs, state

    def create_params_and_state(key, input_shapes):
        x_shape = input_shapes['x']
        height, width, channel = x_shape
        kernel = kernel_init(key, filter_size + (channel, channel))
        kernel = vmap(vmap(util.whiten))(kernel)

        assert kernel.shape == filter_size + (channel, channel)

        assert filter_size[0] <= height, 'filter_size: %s, x_shape: %s'%(filter_size, x_shape)
        assert filter_size[1] <= width, 'filter_size: %s, x_shape: %s'%(filter_size, x_shape)
        params = {'kernel': kernel}
        state = {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

################################################################################################################

__all__ = ['CircularConv']
