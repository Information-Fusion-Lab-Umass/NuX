import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import nux.util as util
from nux.flows.base import Flow

__all__ = ["CircularConv",
           "OneByOneConv"]

fft_channel_vmap = jax.vmap(jnp.fft.fftn, in_axes=(2,), out_axes=2)
ifft_channel_vmap = jax.vmap(jnp.fft.ifftn, in_axes=(2,), out_axes=2)
fft_double_channel_vmap = jax.vmap(fft_channel_vmap, in_axes=(2,), out_axes=2)

inv_height_vmap = jax.vmap(jnp.linalg.inv)
inv_height_width_vmap = jax.vmap(inv_height_vmap)

def complex_slogdet(x):
    D = jnp.block([[x.real, -x.imag], [x.imag, x.real]])
    return 0.25*jnp.linalg.slogdet(D@D.T)[1]
slogdet_height_width_vmap = jax.vmap(jax.vmap(complex_slogdet))

class CircularConv(Flow):

  def __init__(self,
               filter_shape: Sequence[int]=(3, 3)
  ):
    """ Circular convolution.  Equivalent to a regular convolution with circular padding.
        https://papers.nips.cc/paper/2019/file/b1f62fa99de9f27a048344d55c5ef7a6-Paper.pdf
    Args:
      filter_shape: Height and width for the convolutional filter, (Kx, Ky).  The full
                    kernel will have shape (Kx, Ky, C, C)
      name        : Optional name for this module.
    """
    assert len(filter_shape) == 2
    self.filter_shape = filter_shape

  def get_params(self):
    return {"w": self.w}

  def __call__(self, x, params=None, inverse=False, rng_key=None, **kwargs):

    # http://developer.download.nvidia.com/compute/cuda/2_2/sdk/website/projects/convolutionFFT2D/doc/convolutionFFT2D.pdf
    x_h, x_w, x_c = x.shape[-3:]

    if params is None:

      def orthogonal_init(shape, dtype):
        w = random.normal(rng_key, shape=shape, dtype=dtype)
        return jax.vmap(jax.vmap(util.whiten))(w)
      self.w = orthogonal_init(self.filter_shape + (x_c, x_c), x.dtype)
    else:
      self.w = params["w"]

    Kx, Ky, _, _ = self.w.shape

    # See how much we need to roll the filter
    W_x = (Kx - 1) // 2
    W_y = (Ky - 1) // 2

    # Pad the filter to match the fft size and roll it so that its center is at (0,0)
    W_padded = jnp.pad(self.w[::-1,::-1,:,:], ((0, x_h - Kx), (0, x_w - Ky), (0, 0), (0, 0)))
    W_padded = jnp.roll(W_padded, (-W_x, -W_y), axis=(0, 1))

    def apply(x):
      # Apply the FFT to get the convolution
      if inverse == False:
        image_fft = fft_channel_vmap(x)
      else:
        image_fft = fft_channel_vmap(x)
      W_fft = fft_double_channel_vmap(W_padded)

      if inverse == True:
        z_fft = jnp.einsum('abij,abj->abi', W_fft, image_fft)
        z = ifft_channel_vmap(z_fft).real
      else:
        # For deconv, we need to invert the W over the channel dims
        W_fft_inv = inv_height_width_vmap(W_fft)

        x_fft = jnp.einsum('abij,abj->abi', W_fft_inv, image_fft)
        z = ifft_channel_vmap(x_fft).real

      # The log determinant is the log det of the frequencies over the channel dims
      log_det = -slogdet_height_width_vmap(W_fft).sum()
      return z, log_det

    z, log_det = jax.vmap(apply)(x)

    return z, log_det

################################################################################################################

class OneByOneConv(Flow):

  def __init__(self):
    """ 1x1 convolution.  Uses a dense parametrization because the channel dimension will probably
        never be that big.  Costs O(C^3).  Used in GLOW https://arxiv.org/pdf/1807.03039.pdf
    """
    pass

  def get_params(self):
    return {"w": self.w}

  def __call__(self, x, params=None, inverse=False, rng_key=None, **kwargs):
    H, W, C = x.shape[-3:]

    if params is None:
      def orthogonal_init(shape, dtype):
        w = random.normal(rng_key, shape=shape, dtype=dtype)
        return util.whiten(w)
      w = random.normal(rng_key, shape=(C, C))
      self.w = util.whiten(w)
    else:
      self.w = params["w"]

    # Using lax.conv instead of matrix multiplication over the channel dimension
    # is faster and also more numerically stable for some reason.
    def conv(w, x):
      return jax.lax.conv_general_dilated(x,
                                          w[None,None,...],
                                          (1, 1),
                                          'SAME',
                                          (1, 1),
                                          (1, 1),
                                          dimension_numbers=('NHWC', 'HWIO', 'NHWC'))

    # Run the flow
    if inverse == False:
      z = conv(self.w, x)
    else:
      w_inv = jnp.linalg.inv(self.w)
      z = conv(w_inv, x)

    log_det = jnp.linalg.slogdet(self.w)[1]*H*W
    log_det *= jnp.ones(x.shape[:1])

    return z, log_det

################################################################################################################
