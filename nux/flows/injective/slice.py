import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import nux.util as util
import einops
from nux.flows.base import Flow

__all__ = ["Slice"]

class Slice(Flow):

  def __init__(self, manifold_dim):
    self.manifold_dim = manifold_dim

  def get_params(self):
    return {}

  def __call__(self, x, inverse=False, **kwargs):
    if inverse == False:
      self.x_shape = x.shape[1:]
      z = x[...,:self.manifold_dim]
    else:
      pad_shape = x.shape[:1] + self.x_shape
      pad_shape = pad_shape[:-1] + (pad_shape[-1] - self.manifold_dim,)
      z = jnp.concatenate([x, jnp.zeros(pad_shape)], axis=-1)
    return z, jnp.zeros(x.shape[:1])
