import jax
from jax import random
import jax.numpy as jnp
from functools import partial
import nux.util as util
from typing import Optional, Mapping, Callable, Sequence
import nux

""" Adapted from the SurVAE repo: https://github.com/didriknielsen/survae_flows/blob/master/survae/transforms/surjections/maxpool2d.py """

__all__ = ["MaxPool"]

################################################################################################################

def extract_max_elts(x):
  H, W, C = x.shape
  assert H%2 == 0 and W%2 == 0

  # Squeeze so that the grid elements are aligned on the last axis
  x_squeeze = util.pixel_squeeze(x)

  # Sort each grid
  x_sq_argsorted = x_squeeze.argsort(axis=-1)

  # Find the max index of each grid
  max_idx = x_sq_argsorted[...,-1:]

  # Get all of the elements that aren't the max.
  non_max_idx = x_sq_argsorted[...,:-1]

  # Sort the non-max indices so that we can pass the decoder consistent information.
  non_max_idx = non_max_idx.sort(axis=-1)

  # Take the max elements
  max_elts = jnp.take_along_axis(x_squeeze, max_idx, axis=-1).squeeze(axis=-1)
  assert max_elts.shape == (H//2, H//2, C)

  # Take the remaining elements
  non_max_elts = jnp.take_along_axis(x_squeeze, non_max_idx, axis=-1)

  # Subtract elts from the max so that we are left with positive elements
  non_max_elts = max_elts[...,None] - non_max_elts
  non_max_elts = non_max_elts.reshape((H//2, W//2, 3*C))

  return max_elts, non_max_elts, max_idx.squeeze(axis=-1), non_max_idx

################################################################################################################

def generate_grid_indices(shape, rng):
  total_dim = util.list_prod(shape)

  # Generate the indices for each pixel
  idx = jnp.arange(4).tile((total_dim, 1))

  # Shuffle the indices.  random.permutation doesn't accept an axis argument for some reason.
  rngs = random.split(rng, total_dim)
  idx = jax.vmap(random.permutation)(rngs, idx)

  # Separate into the max and non-max indices
  max_idx = idx[...,0].reshape(shape)
  non_max_idx = idx[...,1:]
  non_max_idx = non_max_idx.sort(axis=-1)
  non_max_idx = non_max_idx.reshape(shape + (3,))

  return max_idx, non_max_idx

def index_to_coordinate_array(idx, offset=4, repeat=1):
  # Turn an array of index values into a tuple of coordinate arrays
  H, W, C = idx.shape[:3]

  # The input indices will be spread out by some offset
  flat_coordinates = idx.ravel() + offset*jnp.arange(H*W*C).repeat(repeat)

  return jnp.unravel_index(flat_coordinates, (H, W, C, offset))

################################################################################################################

def contruct_from_max_elts(max_elts, non_max_elts, max_idx, non_max_idx):
  H, W, three_C = non_max_elts.shape
  assert three_C%3 == 0 and max_elts.shape == (H, W, three_C//3)
  C = three_C//3

  # The non max elements are passed in as values greater than 0.  Translate them
  # into other non-max elements here
  non_max_elts = max_elts[...,None] - non_max_elts.reshape((H, W, C, 3))

  # Turn the indices of the max elements to coordinate arrays
  max_coord = index_to_coordinate_array(max_idx, offset=4, repeat=1)
  non_max_coord = index_to_coordinate_array(non_max_idx, offset=4, repeat=3)

  # Construct the new array
  x_squeeze = jnp.zeros((H, W, C, 4))
  x_squeeze = x_squeeze.at[max_coord].set(max_elts.ravel())
  x_squeeze = x_squeeze.at[non_max_coord].set(non_max_elts.ravel())

  # Unsqueeze the image
  return util.pixel_unsqueeze(x_squeeze)

################################################################################################################

class MaxPool():

  def __init__(self, decoder: Callable=None):
    """ Max pool as described in https://arxiv.org/pdf/2007.02731.pdf
        This isn't the usual max pool where we pool with overlapping patches.
        Instead, this pools over non-overlapping patches of pixels.
    Args:
      decoder       : The flow to use to learn the non-max elements.
      name          : Optional name for this module.
    """
    self.decoder = decoder

  def get_params(self):
    return {"decoder": self.decoder.get_params()}

  def __call__(self, x, params=None, inverse=False, rng_key=None, **kwargs):
    if params is None:
      self.d_params = None
    else:
      self.d_params = params["decoder"]

    if inverse == False:

      # Get the max and non-max elements
      max_elts, non_max_elts, max_idx, non_max_idx = jax.vmap(extract_max_elts)(x)

      # See how likely these non-max elements are.  Condition on values and indices
      # so that the decoder has context on what to generate.
      cond = jnp.concatenate([max_elts[...,None], non_max_idx], axis=-1)
      cond = cond.reshape(cond.shape[:-2] + (-1,))
      _, log_qzgx = self.decoder(non_max_elts, aux=cond, params=self.d_params, inverse=False, rng_key=rng_key)

      # We are assuming a uniform distribution for the order of the indices
      log_qkgx = -jnp.log(4)*max_elts.size
    else:
      max_elts = x
      max_elts_shape = x.shape[1:]
      max_elts_size = util.list_prod(max_elts_shape)
      rng1, rng2 = random.split(rng_key, 2)

      # Sample the max indices from q(k|x)
      n_keys = x.shape[0]
      rngs = random.split(rng1, n_keys)
      max_idx, non_max_idx = jax.vmap(partial(generate_grid_indices, max_elts_shape))(rngs)
      log_qkgx = -jnp.log(4)*max_elts_size

      # Sample the non-max indices
      H, W, C = max_idx.shape[-3:]
      cond = jnp.concatenate([max_elts[...,None], non_max_idx], axis=-1)
      cond = cond.reshape(cond.shape[:-2] + (-1,))
      decoder_inputs = {"x": jnp.zeros(self.batch_shape + (H, W, 3*C)), "condition": cond}
      non_max_elts, log_qzgx = self.decoder(jnp.zeros((1, H, W, 3*C), aux=cond, params=params, inverse=True, rng_key=rng2))

      # Combine the max elements with the non-max elements
      z = jax.vmap(contruct_from_max_elts)(max_elts, non_max_elts, max_idx, non_max_idx)

    llc = log_qzgx + log_qkgx
    return z, llc