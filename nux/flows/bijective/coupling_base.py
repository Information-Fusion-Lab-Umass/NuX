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
from abc import ABC, abstractmethod
import warnings

__all__ = ["Elementwise",
           "condition_by_coupling"]

class Elementwise(InvertibleLayer, ABC):

  def __init__(self,
               *,
               create_network: Optional[Callable],
               axis: Optional[int],
               coupling: bool,
               split_kind: str,
               masked: bool,
               use_condition: bool,
               condition_method: str,
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
    self.condition_method     = condition_method
    self.apply_to_both_halves = apply_to_both_halves
    self.coupling                = coupling
    self.split_kind           = split_kind
    self.masked               = masked
    assert split_kind in ["checkerboard", "channel"]
    assert condition_method in ["concat", "nin"]

  def get_network(self, out_shape):
    if hasattr(self, "_network"):
      return self._network

    # The user can specify a custom network
    if self.create_network is not None:
      self._network = self.create_network(out_shape)

    else:
      self._network = util.get_default_network(out_shape, network_kwargs=self.network_kwargs)

    return self._network

  @abstractmethod
  def get_out_shape(self, x):
    pass

  @abstractmethod
  def transform(self, x, params=None, sample=False, rng=None, **kwargs):
    # Returns z, elementwise_log_det
    pass

  def _transform(self, x, params=None, sample=False, mask=None, rng=None, **kwargs):
    z, ew_log_det = self.transform(x, params=params, sample=sample, rng=rng, **kwargs)
    assert z.shape == ew_log_det.shape

    # If we're doing mask coupling, need to correctly mask log_s before
    # computing the log determinant and also mask the output
    if mask is not None:
      z *= mask
      ew_log_det *= mask

    # Sum over each dimension
    sum_axes = util.last_axes(x.shape[len(self.batch_shape):])
    log_det = ew_log_det.sum(sum_axes)
    return z, log_det

  def apply_conditioner_network(self, key, x, condition, **kwargs):
    if self.use_condition:
      if self.condition_method == "nin":
        if len(self.unbatched_input_shapes["x"]) == 1:
          warnings.warn("Using 'nin' conditioning method on 1d inputs is currently not supported.")
        network_in = {"x": x, "aux": condition}

      elif self.condition_method == "concat":
        x_in = jnp.concatenate([x, condition], axis=self.axis) if condition is not None else x
        network_in = {"x": x_in}

      else:
        assert 0, "Invalid condition method"
    else:
      network_in = {"x": x}

    network_out = self.network(network_in, key, **kwargs)
    x_out = network_out["x"]
    return x_out

  def standard_call(self,
                    inputs: Mapping[str, jnp.ndarray],
                    rng: jnp.ndarray=None,
                    sample: Optional[bool]=False,
                    **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    """ Perform coupling by splitting the input
    """
    k1, k2, k3 = random.split(rng, 3)
    x = inputs["x"]
    unbatched_dim = len(self.get_unbatched_shapes(sample)["x"])
    if self.use_condition:
      assert "condition" in inputs
      condition = inputs["condition"]
    else:
      condition = None

    # Initialize the network
    out_shape = self.get_out_shape(x)
    self.network = self.get_network(out_shape)

    # Pass in whatever other kwargs we might have
    extra_kwargs = inputs.copy()
    extra_kwargs.update(kwargs)
    extra_kwargs.pop("x")

    if sample == False:
      # z = f(x; condition, theta)
      network_out = self.apply_conditioner_network(k2, condition, None, **kwargs) if condition is not None else None
      z, log_det = self._transform(x, params=network_out, sample=False, rng=k3, **extra_kwargs)
    else:
      # xb = f^{-1}(zb; theta).  (x and z are swapped so that the code is a bit cleaner)
      network_out = self.apply_conditioner_network(k2, condition, None, **kwargs) if condition is not None else None
      z, log_det = self._transform(x, params=network_out, sample=True, rng=k3, **extra_kwargs)

    outputs = {"x": z, "log_det": log_det}
    return outputs

  def split_call(self,
                 inputs: Mapping[str, jnp.ndarray],
                 rng: jnp.ndarray=None,
                 sample: Optional[bool]=False,
                 **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    """ Perform coupling by splitting the input
    """
    k1, k2, k3 = random.split(rng, 3)
    x = inputs["x"]
    unbatched_dim = len(self.get_unbatched_shapes(sample)["x"])
    if self.use_condition:
      assert "condition" in inputs
      condition = inputs["condition"]
    else:
      condition = None

    if self.split_kind == "checkerboard":
      if unbatched_dim == 1:
        assert 0, "Only supporting masked checkerboard coupling for 1d inputs"
      x = self.auto_batch(util.half_squeeze)(x)

    # Figure out the output shape
    x_shape = x.shape[-unbatched_dim:]
    ax = self.axis%len(x_shape)
    split_index = x_shape[ax]//2
    xa, xb = jnp.split(x, indices_or_sections=jnp.array([split_index]), axis=self.axis)

    # Initialize the network
    out_shape = self.get_out_shape(xa)
    self.network = self.get_network(out_shape)

    # Pass in whatever other kwargs we might have
    extra_kwargs = inputs.copy()
    extra_kwargs.update(kwargs)
    extra_kwargs.pop("x")

    if sample == False:
      # zb = f(xb; theta)
      if self.apply_to_both_halves:
        zb, log_detb = self._transform(xb, sample=False, rng=k1, **extra_kwargs)
      else:
        zb, log_detb = xb, 0.0

      # za = f(xa; NN(xb))
      network_out = self.apply_conditioner_network(k2, xb, condition, **kwargs)
      za, log_deta = self._transform(xa, params=network_out, sample=False, rng=k3, **extra_kwargs)
    else:
      # xb = f^{-1}(zb; theta).  (x and z are swapped so that the code is a bit cleaner)
      if self.apply_to_both_halves:
        zb, log_detb = self._transform(xb, sample=True, rng=k1, **extra_kwargs)
      else:
        zb, log_detb = xb, 0.0

      # xa = f^{-1}(za; NN(xb)).
      network_out = self.apply_conditioner_network(k2, zb, condition, **kwargs)
      za, log_deta = self._transform(xa, params=network_out, sample=True, rng=k3, **extra_kwargs)

    # Recombine
    z = jnp.concatenate([za, zb], axis=self.axis)
    log_det = log_deta + log_detb

    if self.split_kind == "checkerboard":
      z = self.auto_batch(util.half_unsqueeze)(z)

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
    k1, k2, k3 = random.split(rng, 3)
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
    nmask = ~mask

    x = inputs["x"]
    if self.use_condition:
      assert "condition" in inputs
      condition = inputs["condition"]
    else:
      condition = None

    # Mask the input
    x_mask = x*mask
    x_nmask = x*nmask

    # Initialize the network
    out_shape = self.get_out_shape(x_mask)
    self.network = self.get_network(out_shape)

    if sample == False:
      # zb = f(xb; theta)
      if self.apply_to_both_halves:
        z_nmask, log_det_b = self._transform(x_nmask, sample=False, mask=nmask, rng=k1)
      else:
        z_nmask, log_det_b = x_nmask, 0.0

      # za = f(xa; NN(xb))
      network_out = self.apply_conditioner_network(k2, x_nmask, condition, **kwargs)
      z_mask, log_det_a = self._transform(x, params=network_out, sample=False, mask=mask, rng=k3)
    else:
      # xb = f^{-1}(zb; theta).  (x and z are swapped so that the code is a bit cleaner)
      if self.apply_to_both_halves:
        z_nmask, log_det_b = self._transform(x_nmask, sample=True, mask=nmask, rng=k1)
      else:
        z_nmask, log_det_b = x_nmask, 0.0

      # xa = f^{-1}(za; NN(xb)).
      network_out = self.apply_conditioner_network(k2, z_nmask, condition, **kwargs)
      x_in = z_nmask + x_mask
      z_mask, log_det_a = self._transform(x_in, params=network_out, sample=True, mask=mask, rng=k3)

    # Apply the other half of the mask to the output
    z = z_nmask + z_mask
    log_det = log_det_a + log_det_b

    outputs = {"x": z, "log_det": log_det}
    return outputs

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    if self.coupling == False:
      return self.standard_call(inputs, rng, sample=sample, **kwargs)
    if self.masked:
      return self.masked_call(inputs, rng, sample=sample, **kwargs)
    return self.split_call(inputs, rng, sample=sample, **kwargs)

################################################################################################################

class condition_by_coupling(InvertibleLayer):

  def __init__(self,
               flow,
               name: str="condition_by_coupling"
  ):
    """ If we want to use a "condition" input in conjunction with an input,
        this provides a way to do that using whats already implemented for coupling.
    Args:
      name: Optional name for this module.
    """
    assert isinstance(flow, Elementwise)
    self.flow = flow
    self.flow.axis = -1
    self.flow.use_condition = False
    self.flow.apply_to_both_halves = False
    self.flow.split_kind = "channel"
    self.flow.masked = False
    super().__init__(name=name)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]

    assert "condition" in inputs
    condition = inputs["condition"]

    # Just concatenate the input with condition
    coupling_inputs = inputs.copy()
    coupling_inputs["x"] = jnp.concatenate([x, condition], axis=-1)
    coupling_outputs = self.flow(coupling_inputs, rng, sample=sample, **kwargs)

    split_index = x.shape[-1]
    z, _ = jnp.split(coupling_outputs["x"], jnp.array([split_index]), axis=-1)

    outputs = {"x": z, "log_det": coupling_outputs["log_det"]}
    return outputs
