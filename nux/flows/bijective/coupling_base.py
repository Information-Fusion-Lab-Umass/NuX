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
from abc import ABC, abstractmethod

__all__ = ["CouplingBase"]

class CouplingBase(Layer, ABC):

  def __init__(self,
               *,
               create_network: Optional[Callable],
               axis: Optional[int],
               split_kind: str,
               masked: bool,
               use_condition: bool,
               network_kwargs: Optional[Mapping],
               apply_to_both_halves: Optional[bool],
               name: str,
               **kwargs
  ):
    """ Coupling transformation.  Transform an input, x = [xa,xb] using
        za = f(xa; NN(xb)), zb = f(xb; theta), z = [za, ab]
        Remember that BOTH halves of x/z are transformed in this function.
    Args:
      create_network: Function to create the conditioner network.  Should accept a tuple
                      specifying the output shape.  See coupling_base.py
      kind          : "affine" or "additive".  If additive, s(.) = 1
      axis          : Axis to apply the transformation to
      split_kind    : If we input an image, we can split by "channel" or using a "checkerboard" split
      use_condition : Should we concatenate inputs["condition"] to xb in NN([xb, condition])?
      network_kwargs: Dictionary with settings for the default network (see get_default_network in util.py)
      name          : Optional name for this module.
    """
    super().__init__(name=name, **kwargs)
    self.axis                 = axis
    self.create_network       = create_network
    self.network_kwargs       = network_kwargs
    self.use_condition        = use_condition
    self.apply_to_both_halves = apply_to_both_halves
    self.split_kind           = split_kind
    self.masked               = masked
    if masked:
      assert self.apply_to_both_halves == False, "Not supporting both halves if using masked"
    assert split_kind in ["checkerboard", "channel"]

  def get_network(self, out_shape):
    # The user can specify a custom network
    if self.create_network is not None:
      return self.create_network(out_shape)

    return util.get_default_network(out_shape, network_kwargs=self.network_kwargs)

  @abstractmethod
  def get_out_shape(self, x):
    pass

  @abstractmethod
  def transform(self, x, params=None, sample=False, mask=None):
    pass

  def resize_condition(self, x_shape, condition):

    # Ensure that condition is the same size as xb so that they can be concatenated
    H, W = x.shape[-3:-1]
    Hc, Wc = condition.shape[-3:-1]

    if Hc > H and Wc > W:
      condition = self.auto_batch(partial(hk.max_pool, strides=2, window_shape=2, padding="VALID"), expected_depth=1)(condition)
      Hc, Wc = condition.shape[-3:-1]

    assert Hc == H and Wc == W
    return condition

  def split_call(self,
                 inputs: Mapping[str, jnp.ndarray],
                 rng: jnp.ndarray=None,
                 sample: Optional[bool]=False,
                 **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    """ Perform coupling by splitting the input
    """
    x = inputs["x"]
    unbatched_dim = len(self.get_unbatched_shapes(sample)["x"])
    if self.use_condition:
      assert "condition" in inputs
      condition = inputs["condition"]

    if self.split_kind == "checkerboard":
      if unbatched_dim == 1:
        assert 0, "Only supporting masked checkerboard coupling for 1d inputs"
      x = util.half_squeeze(x)

    # Figure out the output shape
    x_shape = x.shape[-unbatched_dim:]
    ax = self.axis%len(x_shape)
    split_index = x_shape[ax]//2
    xa, xb = jnp.split(x, indices_or_sections=jnp.array([split_index]), axis=self.axis)

    # Initialize the network
    out_shape = self.get_out_shape(xa)
    network = self.get_network(out_shape)

    if sample == False:
      # zb = f(xb; theta)
      if self.apply_to_both_halves:
        zb, log_detb = self.transform(xb, sample=False)
      else:
        zb, log_detb = xb, 0.0

      # za = f(xa; NN(xb))
      network_in = jnp.concatenate([xb, condition], axis=self.axis) if self.use_condition else xb
      network_out = self.auto_batch(network, expected_depth=1, in_axes=(0, None))(network_in, rng)
      za, log_deta = self.transform(xa, params=network_out, sample=False)
    else:
      # xb = f^{-1}(zb; theta).  (x and z are swapped so that the code is a bit cleaner)
      if self.apply_to_both_halves:
        zb, log_detb = self.transform(xb, sample=True)
      else:
        zb, log_detb = xb, 0.0

      # xa = f^{-1}(za; NN(xb)).
      network_in = jnp.concatenate([zb, condition], axis=self.axis) if self.use_condition else zb
      network_out = self.auto_batch(network, expected_depth=1, in_axes=(0, None))(network_in, rng)
      za, log_deta = self.transform(xa, params=network_out, sample=True)

    # Recombine
    log_det = log_deta + log_detb
    z = jnp.concatenate([za, zb], axis=self.axis)

    if self.split_kind == "checkerboard":
      z = util.half_unsqueeze(z)

    outputs = {"x": z, "log_det": log_det}
    return outputs

  def masked_call(self,
                 inputs: Mapping[str, jnp.ndarray],
                 rng: jnp.ndarray=None,
                 sample: Optional[bool]=False,
                 **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    """ Perform coupling by masking the input
    """
    # Generate the mask
    def mask_init(shape, dtype):
      if len(shape) == 3:
        H, W, C = shape
        X, Y, Z = jnp.meshgrid(jnp.arange(H), jnp.arange(W), jnp.arange(C))
        if self.split_kind == "checkerboard":
          mask = (X + Y + Z)%2
        elif self.split_kind == "channel":
          mask = (X, Y, Z)[self.axis] > shape[self.axis]//2
      else:
        dim, = shape
        if self.split_kind == "checkerboard":
          mask = jnp.arange(dim)%2
        elif self.split_kind == "channel":
          mask = jnp.arange(dim) > dim//2
      return mask.astype(dtype)

    x_shape = self.unbatched_input_shapes["x"]
    mask = hk.get_state("mask", shape=x_shape, dtype=bool, init=mask_init)

    x = inputs["x"]
    if self.use_condition:
      assert "condition" in inputs
      condition = inputs["condition"]

    # Mask the input
    x_not_mask = x*(~mask)

    # Initialize the network
    out_shape = self.get_out_shape(x)
    network = self.get_network(out_shape)

    network_in = jnp.concatenate([x_not_mask, condition], axis=self.axis) if self.use_condition else x_not_mask
    network_out = self.auto_batch(network, expected_depth=1, in_axes=(0, None))(network_in, rng)

    if sample == False:
      z, log_det = self.transform(x, params=network_out, sample=False, mask=mask)
    else:
      z, log_det = self.transform(x, params=network_out, sample=True, mask=mask)

    # Apply the other half of the mask to the output
    z_mask = z*mask
    z = x_not_mask + z_mask

    outputs = {"x": z, "log_det": log_det}
    return outputs

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    if self.masked:
      return self.masked_call(inputs, rng, sample=sample, **kwargs)
    return self.split_call(inputs, rng, sample=sample, **kwargs)
