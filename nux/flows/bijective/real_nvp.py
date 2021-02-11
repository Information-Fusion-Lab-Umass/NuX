import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence
from nux.internal.layer import InvertibleLayer
import nux.util as util
import nux.networks as net

__all__ = ["RealNVP"]

from nux.flows.bijective.coupling_base import Elementwise

class RealNVP(Elementwise):
  def __init__(self,
               create_network: Optional[Callable]=None,
               kind: Optional[str]="affine",
               axis: Optional[int]=-1,
               coupling: bool=True,
               split_kind: str="channel",
               masked: bool=False,
               use_condition: bool=False,
               condition_method: str="nin",
               apply_to_both_halves: Optional[bool]=True,
               network_kwargs: Optional[Mapping]=None,
               safe_diag: bool=True,
               name: str="real_nvp",
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
                     coupling=coupling,
                     split_kind=split_kind,
                     masked=masked,
                     use_condition=use_condition,
                     condition_method=condition_method,
                     name=name,
                     apply_to_both_halves=apply_to_both_halves,
                     network_kwargs=network_kwargs,
                     **kwargs)
    self.kind = kind
    self.safe_diag = safe_diag

  def get_out_shape(self, x):
    x_shape = x.shape[len(self.batch_shape):]
    out_dim = x_shape[-1] if self.kind == "additive" else 2*x_shape[-1]
    return x_shape[:-1] + (out_dim,)

  def transform(self, x, params=None, sample=False, rng=None, **kwargs):
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

    if self.kind == "affine":
      if self.safe_diag:
        s = util.proximal_relu(log_s) + 1e-6
        log_s = jnp.log(s)
      else:
        s = jnp.exp(log_s)

    # Evaluate the transformation
    if sample == False:
      z = (x - t)/s if self.kind == "affine" else x - t
    else:
      z = x*s + t if self.kind == "affine" else x + t

    if self.kind == "affine":
      elementwise_log_det = jnp.broadcast_to(-log_s, x.shape)
    else:
      elementwise_log_det = jnp.zeros_like(x)

    return z, elementwise_log_det
