import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap, jit
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Sequence
from nux.flows.base import *
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

class CircularConv(AutoBatchedLayer):

  def __init__(self, filter_shape: Sequence[int], name: str="circular_conv", **kwargs):
    super().__init__(name=name, **kwargs)
    assert len(filter_shape) == 2
    self.filter_shape = filter_shape

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:

    shape, dtype = inputs["x"].shape, inputs["x"].dtype
    init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal')
    W = hk.get_parameter("W", shape=self.filter_shape + (shape[-1], shape[-1]), dtype=dtype, init=init)
    b = hk.get_parameter("b", shape=shape, dtype=dtype, init=jnp.zeros)

    x = inputs["x"]
    outputs = {}

    # http://developer.download.nvidia.com/compute/cuda/2_2/sdk/website/projects/convolutionFFT2D/doc/convolutionFFT2D.pdf
    x_h, x_w, x_c = x.shape
    W_h, W_w, W_c_out, W_c_in = W.shape

    # See how much we need to roll the filter
    W_x = (W_h - 1) // 2
    W_y = (W_w - 1) // 2

    # Pad the filter to match the fft size and roll it so that its center is at (0,0)
    W_padded = jnp.pad(W[::-1,::-1,:,:], ((0, x_h - W_h), (0, x_w - W_w), (0, 0), (0, 0)))
    W_padded = jnp.roll(W_padded, (-W_x, -W_y), axis=(0, 1))

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

    outputs = {'x': z, 'log_det': log_det}
    return outputs
