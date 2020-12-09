import jax
import jax.numpy as jnp

def dilated_squeeze(x, filter_shape=(2, 2), dilation=(1, 1)):
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

def dilated_unsqueeze(x, filter_shape=(2, 2), dilation=(1, 1)):

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

def pixel_squeeze(x, grid_size=4):
  # Standard squeezing stacks channel dimensions.  This doesn't do that.
  # This rearranges an image so that 2x2 patches are put on the last axis
  # Can account for missing pixels with grid_size
  H, W, C = x.shape
  x = x.reshape(H//2, 2, W//2, 2, C)
  x = x.transpose((0, 2, 4, 1, 3))
  x = x.reshape(H//2, W//2, -1, grid_size)
  return x

def pixel_unsqueeze(x):
  H, W, C, _ = x.shape
  x = x.reshape(H, W, C, 2, 2)
  x = x.transpose((0, 3, 1, 4, 2))
  x = x.reshape(H*2, W*2, C)
  return x

def upsample(x):
  x = jnp.repeat(x, 2, axis=0)
  x = jnp.repeat(x, 2, axis=1)
  return x

def half_squeeze(x):
  H, W, C = x.shape
  x = x.reshape((H, W//2, 2, C))
  x = x.transpose((0, 1, 3, 2))
  return x.reshape((H, W//2, 2*C))

def half_unsqueeze(x):
  H, W_half, C_two = x.shape
  W = W_half*2
  C = C_two//2
  x = x.reshape((H, W_half, C, 2))
  x = x.transpose((0, 1, 3, 2))
  x = x.reshape((H, W, C))
  return x
