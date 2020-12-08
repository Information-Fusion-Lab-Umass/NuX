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
               create_network: Optional[Callable]=None,
               axis: Optional[int]=-1,
               split_kind: str="channel",
               use_condition: bool=False,
               network_kwargs: Optional[Mapping]=None,
               apply_to_both_halves: Optional[bool]=True,
               name: str="coupling",
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
  def transform(self, x, params=None):
    pass

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    if self.use_condition:
      assert "condition" in inputs
      condition = inputs["condition"]

    if(self.split_kind == "checkerboard"):
      x = self.auto_batch(util.dilated_squeeze)(x)
      if self.use_condition:
        condition = self.auto_batch(util.dilated_squeeze)(condition)

    # Figure out the output shape
    x_shape = self.get_unbatched_shapes(sample)["x"]
    ax = self.axis%len(x_shape)
    split_index = x_shape[ax]//2
    xa, xb = jnp.split(x, indices_or_sections=jnp.array([split_index]), axis=self.axis)

    # Ensure that condition is the same size as xb so that they can be concatenated
    if self.use_condition and len(x_shape) == 3:
      H, W = x.shape[-3:-1]
      Hc, Wc = condition.shape[-3:-1]

      if Hc > H and Wc > W:
        condition = self.auto_batch(partial(hk.max_pool, strides=2, window_shape=2, padding="VALID"), expected_depth=1)(condition)
        Hc, Wc = condition.shape[-3:-1]

      assert Hc == H and Wc == W

    # Initialize the coupling layer to the identity
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

    if(self.split_kind == "checkerboard"):
      z = self.auto_batch(util.dilated_unsqueeze)(z)

    outputs = {"x": z, "log_det": log_det}
    return outputs
