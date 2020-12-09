import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence
from nux.internal.layer import Layer
import nux.util as util
import nux.networks as net

__all__ = ["Coupling"]

from nux.flows.bijective.coupling_base import CouplingBase

class Coupling(CouplingBase):
  def __init__(self,
               create_network: Optional[Callable]=None,
               kind: Optional[str]="affine",
               axis: Optional[int]=-1,
               split_kind: str="channel",
               masked: bool=False,
               use_condition: bool=False,
               apply_to_both_halves: Optional[bool]=True,
               network_kwargs: Optional[Mapping]=None,
               name: str="coupling",
               **kwargs
  ):
    """ Affine/additive coupling.  Transform an input, x = [xa,xb] using
        za = (xa - t(xb))/s(xb), zb = (xb - t)/s, z = [za, ab]
        Used in RealNVP https://arxiv.org/pdf/1605.08803.pdf
    Args:
      create_network: Function to create the conditioner network.  Should accept a tuple
                      specifying the output shape.  See coupling_base.py
      kind          : "affine" or "additive".  If additive, s(.) = 1
      axis          : Axis to apply the transformation to
      split_kind    : If we input an image, we can split by "channel" or using a "checkerboard" split
      use_condition : Should we use inputs["condition"] to form t([xb, condition]), s([xb, condition])?
      network_kwargs: Dictionary with settings for the default network (see get_default_network in util.py)
      name          : Optional name for this module.
    """
    super().__init__(create_network=create_network,
                     axis=axis,
                     split_kind=split_kind,
                     masked=masked,
                     use_condition=use_condition,
                     name=name,
                     apply_to_both_halves=apply_to_both_halves,
                     network_kwargs=network_kwargs,
                     **kwargs)
    self.kind = kind

  def get_out_shape(self, x):
    x_shape = x.shape[len(self.batch_shape):]
    out_dim = x_shape[-1] if self.kind == "additive" else 2*x_shape[-1]
    return x_shape[:-1] + (out_dim,)

  def transform(self, x, params=None, sample=False, mask=None):
    # Remember that self.get_unbatched_shapes(sample)["x"] is NOT the shape of x here!
    # The x we see here is only half of the actual x!

    # Get the parameters of the transformation
    scale_init = hk.initializers.RandomNormal(stddev=0.01)
    if params is None:
      x_shape = x.shape[len(self.batch_shape):]
      if self.kind == "affine":
        log_s = hk.get_parameter("log_s", shape=x_shape, dtype=x.dtype, init=scale_init)
      t = hk.get_parameter("t", shape=x_shape, dtype=x.dtype, init=scale_init)

    else:
      if self.kind == "affine":
        scale_scale = hk.get_parameter("scale_scale", shape=(), dtype=x.dtype, init=scale_init)
      shift_scale = hk.get_parameter("shift_scale", shape=(), dtype=x.dtype, init=scale_init)

      # Split the output and bound the scaling term
      if self.kind == "affine":
        t, log_s = jnp.split(params, 2, axis=self.axis)
        log_s = util.constrain_log_scale(log_s)
      else:
        t = params

      # Scale the parameters so that we can initialize this function to the identity
      t = shift_scale*t
      if self.kind == "affine":
        log_s = scale_scale*log_s

    # Evaluate the transformation
    if sample == False:
      z = (x - t)*jnp.exp(-log_s) if self.kind == "affine" else x - t
    else:
      z = x*jnp.exp(log_s) + t if self.kind == "affine" else x + t

    # If we're doing mask coupling, need to correctly mask log_s before
    # computing the log determinant and also mask the output
    if mask is not None:
      z *= mask
      log_s *= mask

    # Compute the log determinant
    if self.kind == "affine":
      sum_axes = util.last_axes(x.shape[len(self.batch_shape):])
      log_det = -log_s.sum(axis=sum_axes)
    else:
      log_det = jnp.zeros(self.batch_shape)

    return z, log_det
