import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import nux.util as util
import einops
from nux.flows.base import Flow

__all__ = ["Reverse",
           "Squeeze"]

class Reverse(Flow):

  def __init__(self):
    """ Reverse the last axis of an input.  Useful to put in between coupling layers.
    """
    pass

  def get_params(self):
    return {}

  def __call__(self, x, inverse=False, **kwargs):
    z = x[...,::-1]
    return z, jnp.zeros(x.shape[:1])

class Squeeze(Flow):

  def __init__(self):
    """ Space to depth
    """
    pass

  def get_params(self):
    return {}

  def __call__(self, x, inverse=False, **kwargs):
    if inverse == False:
      z = einops.rearrange(x, "b (h d1) (w d2) c -> b h w (c d1 d2)", d1=2, d2=2)
    else:
      z = einops.rearrange(x, "b h w (c d1 d2) -> b (h d1) (w d2) c", d1=2, d2=2)
    return z, jnp.zeros(x.shape[:1])
