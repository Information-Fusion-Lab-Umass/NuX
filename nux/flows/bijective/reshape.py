import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Sequence
from nux.internal.layer import InvertibleLayer
import nux.util as util

__all__ = ["Squeeze",
           "UnSqueeze",
           "Flatten",
           "Reshape",
           "Reverse"]

class Squeeze(InvertibleLayer):

  def __init__(self,
               filter_shape: Sequence[int]=(2, 2),
               dilation: Sequence[int]=(1, 1),
               name: str="squeeze"
  ):
    """ Squeeze operation as described in RealNVP https://arxiv.org/pdf/1605.08803.pdf
        Stacks consecutive 2x2 patches of pixels (and their channel dims) on top of each other.
        The dilation argument can allow these 2x2 patches to overlap.
    Args:
      filter_shape: Size of patch
      dilation    : How far each element of a patch is from each other
      name        : Optional name for this module.
    """
    super().__init__(name=name)
    self.filter_shape = filter_shape
    self.dilation     = dilation

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]

    @self.auto_batch
    def apply_fun(x):
      sq = util.dilated_squeeze if sample == False else util.dilated_unsqueeze
      return sq(x, self.filter_shape, self.dilation)

    z = apply_fun(x)
    outputs = {"x": z, "log_det": jnp.zeros(self.batch_shape)}

    return outputs

class UnSqueeze(InvertibleLayer):

  def __init__(self,
               filter_shape: Sequence[int]=(2, 2),
               dilation: Sequence[int]=(1, 1),
               name: str="unsqueeze"
  ):
    """ Undo the squeeze operation
    Args:
      filter_shape: Size of patch
      dilation    : How far each element of a patch is from each other
      name        : Optional name for this module.
    """
    super().__init__(name=name)
    self.filter_shape = filter_shape
    self.dilation     = dilation

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]

    @self.auto_batch
    def apply_fun(x):
      sq = util.dilated_squeeze if sample == True else util.dilated_unsqueeze
      return sq(x, self.filter_shape, self.dilation)

    z = apply_fun(x)
    outputs = {"x": z, "log_det": jnp.zeros(self.batch_shape)}

    return outputs

class Flatten(InvertibleLayer):

  def __init__(self,
               name: str="flatten",
               **kwargs
  ):
    """ Flatten an input.  NuX will internally keep track of the input
        shape for inversion.
    Args:
      name: Optional name for this module.
    """
    super().__init__(name=name, **kwargs)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]

    if sample == False:
      unbatched_shape = self.unbatched_input_shapes["x"]
      flat_dim = util.list_prod(unbatched_shape)
      flat_shape = self.batch_shape + (flat_dim,)
      z = x.reshape(flat_shape)
    else:
      original_shape = self.batch_shape + self.unbatched_input_shapes["x"]
      z = x.reshape(original_shape)

    outputs = {"x": z, "log_det": jnp.zeros(self.batch_shape)}
    return outputs

class Reshape(InvertibleLayer):

  def __init__(self,
               output_shape: Sequence[int],
               name: str="flatten",
               **kwargs
  ):
    """ Reshape an input.  NuX will internally keep track of the input
        shape for inversion.
    Args:
      output_shape: Target shape
      name        : Optional name for this module.
    """
    self.output_shape = output_shape

    super().__init__(name=name, **kwargs)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]

    if sample == False:
      out_dim = util.list_prod(self.output_shape)
      expected_out_dim = util.list_prod(self.unbatched_input_shapes["x"])
      assert out_dim == expected_out_dim, f"Dimension mismatch"

      z = x.reshape(self.batch_shape + self.output_shape)
    else:
      original_shape = self.batch_shape + self.unbatched_input_shapes["x"]
      z = x.reshape(original_shape)

    outputs = {"x": z, "log_det": jnp.zeros(self.batch_shape)}
    return outputs

class Reverse(InvertibleLayer):

  def __init__(self,
               name: str="reverse",
               **kwargs
  ):
    """ Reverse the last axis of an input.  Useful to put in between coupling layers.
    Args:
      name        : Optional name for this module.
    """
    super().__init__(name=name, **kwargs)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    z = x[...,::-1]
    outputs = {"x": z, "log_det": jnp.zeros(self.batch_shape)}
    return outputs
