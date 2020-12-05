import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap, jit
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Sequence
from nux.internal.layer import Layer
import nux.util as util

__all__ = ["CircularConv"]

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

class CircularConv(Layer):

  def __init__(self,
               filter_shape: Sequence[int],
               name: str="circular_conv"
  ):
    """ Circular convolution.  Equivalent to a regular convolution with circular padding.
        https://papers.nips.cc/paper/2019/file/b1f62fa99de9f27a048344d55c5ef7a6-Paper.pdf
    Args:
      filter_shape: Height and width for the convolutional filter, (Kx, Ky).  The full
                    kernel will have shape (Kx, Ky, C, C)
      name        : Optional name for this module.
    """
    super().__init__(name=name)
    assert len(filter_shape) == 2
    self.filter_shape = filter_shape

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    x = inputs["x"]
    outputs = {}

    # http://developer.download.nvidia.com/compute/cuda/2_2/sdk/website/projects/convolutionFFT2D/doc/convolutionFFT2D.pdf
    x_h, x_w, x_c = x.shape[-3:]

    def orthogonal_init(shape, dtype):
      W = random.normal(rng, shape=shape, dtype=dtype)*0.1
      return vmap(vmap(util.whiten))(W)
    W = hk.get_parameter("W", shape=self.filter_shape + (x_c, x_c), dtype=x.dtype, init=orthogonal_init)
    b = hk.get_parameter("b", shape=(x_c,), dtype=x.dtype, init=jnp.zeros)
    W_h, W_w, W_c_out, W_c_in = W.shape

    # See how much we need to roll the filter
    W_x = (W_h - 1) // 2
    W_y = (W_w - 1) // 2

    # Pad the filter to match the fft size and roll it so that its center is at (0,0)
    W_padded = jnp.pad(W[::-1,::-1,:,:], ((0, x_h - W_h), (0, x_w - W_w), (0, 0), (0, 0)))
    W_padded = jnp.roll(W_padded, (-W_x, -W_y), axis=(0, 1))

    @self.auto_batch
    def apply(x):
      # Apply the FFT to get the convolution
      if sample == False:
        image_fft = fft_channel_vmap(x)
      else:
        image_fft = fft_channel_vmap(x - b)
      W_fft = fft_double_channel_vmap(W_padded)

      if sample == True:
        z_fft = jnp.einsum('abij,abj->abi', W_fft, image_fft)
        z = ifft_channel_vmap(z_fft).real + b
      else:
        # For deconv, we need to invert the W over the channel dims
        W_fft_inv = inv_height_width_vmap(W_fft)

        x_fft = jnp.einsum('abij,abj->abi', W_fft_inv, image_fft)
        z = ifft_channel_vmap(x_fft).real

      # The log determinant is the log det of the frequencies over the channel dims
      log_det = -slogdet_height_width_vmap(W_fft).sum()
      return z, log_det

    z, log_det = apply(x)

    outputs = {'x': z, 'log_det': log_det}
    return outputs
