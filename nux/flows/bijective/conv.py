import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap, jit
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Sequence
from nux.internal.layer import InvertibleLayer
import nux.util as util

__all__ = ["CircularConv",
           "OneByOneConv"]

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

class CircularConv(InvertibleLayer):

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

################################################################################################################

class OneByOneConv(InvertibleLayer):

  def __init__(self,
               weight_norm: bool=True,
               name: str="one_by_one_conv"
  ):
    """ 1x1 convolution.  Uses a dense parametrization because the channel dimension will probably
        never be that big.  Costs O(C^3).  Used in GLOW https://arxiv.org/pdf/1807.03039.pdf
    Args:
      weight_norm: Should weight norm be applied to the layer?
      name       : Optional name for this module.
    """
    super().__init__(name=name)
    self.weight_norm = weight_norm

    def orthogonal_init(shape, dtype):
      key = hk.next_rng_key()
      W = random.normal(key, shape=shape, dtype=dtype)
      return util.whiten(W)
    self.W_init = orthogonal_init

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    outputs = {}
    x = inputs["x"]
    height, width, channel = x.shape[-3:]

    # Using lax.conv instead of matrix multiplication over the channel dimension
    # is faster and also more numerically stable for some reason.
    @partial(self.auto_batch, in_axes=(None, 0), expected_depth=1)
    def conv(W, x):
      return jax.lax.conv_general_dilated(x,
                                          W[None,None,...],
                                          (1, 1),
                                          'SAME',
                                          (1, 1),
                                          (1, 1),
                                          dimension_numbers=('NHWC', 'HWIO', 'NHWC'))

    dtype = x.dtype
    W = hk.get_parameter("W", shape=(channel, channel), dtype=dtype, init=self.W_init)

    # Initialize with weight norm https://arxiv.org/pdf/1602.07868.pdf
    # This seems to improve performance.
    if self.weight_norm and x.ndim > 3:
      W *= jax.lax.rsqrt(jnp.sum(W**2, axis=0))

      def g_init(shape, dtype):
        t = conv(W, x)
        g = 1/(jnp.std(t, axis=(0, 1, 2)) + 1e-5)
        return g

      def b_init(shape, dtype):
        t = conv(W, x)
        return -jnp.mean(t, axis=(0, 1, 2))/(jnp.std(t, axis=(0, 1, 2)) + 1e-5)

      g = hk.get_parameter("g", (channel,), dtype, init=g_init)
      b = hk.get_parameter("b", (channel,), dtype, init=b_init)

      W *= g

    else:
      b = hk.get_parameter("b", shape=(channel,), dtype=dtype, init=jnp.zeros)

    # Run the flow
    if sample == False:
      z = conv(W, x)
      outputs["x"] = z + b
    else:
      W_inv = jnp.linalg.inv(W)
      outputs["x"] = conv(W_inv, x - b)

    outputs["log_det"] = jnp.linalg.slogdet(W)[1]*height*width*jnp.ones(self.batch_shape)

    return outputs
