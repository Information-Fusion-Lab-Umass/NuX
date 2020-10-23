import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence
from nux.flows.base import *
import nux.util as util
import nux.networks as net

__all__ = ["CouplingBase"]

class CouplingBase(Layer):

  def __init__(self,
               create_network: Optional[Callable]=None,
               axis: Optional[int]=-1,
               split_kind: str="channel",
               use_condition: bool=False,
               name: str="coupling",
               network_kwargs: Optional=None,
               **kwargs
  ):
    super().__init__(name=name, **kwargs)
    self.axis               = axis
    self.create_network     = create_network
    self.network_kwargs     = network_kwargs
    self.use_condition      = use_condition

    self.split_kind         = split_kind
    assert split_kind in ["checkerboard", "channel"]

  def get_network(self, out_shape):
    # The user can specify a custom network
    if self.create_network is not None:
      return self.create_network(out_shape)

    return util.get_default_network(out_shape, self.network_kwargs)

  def get_out_shape(self, x):
    assert 0

  def transform(self, x, params=None):
    assert 0

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

    # Initialize the coupling layer to the identity
    out_shape = self.get_out_shape(xa)
    network = self.get_network(out_shape)

    if sample == False:
      zb, log_detb = self.transform(xb, sample=False)
      network_in = jnp.concatenate([xb, condition], axis=self.axis) if self.use_condition else xb
      network_out = self.auto_batch(network, expected_depth=1)(network_in)
      za, log_deta = self.transform(xa, params=network_out, sample=False)
    else:
      zb, log_detb = self.transform(xb, sample=True)
      network_in = jnp.concatenate([zb, condition], axis=self.axis) if self.use_condition else zb
      network_out = self.auto_batch(network, expected_depth=1)(network_in)
      za, log_deta = self.transform(xa, params=network_out, sample=True)

    # Recombine
    log_det = log_deta + log_detb
    z = jnp.concatenate([za, zb], axis=self.axis)

    if(self.split_kind == "checkerboard"):
      z = self.auto_batch(util.dilated_unsqueeze)(z)

    outputs = {"x": z, "log_det": log_det}
    return outputs
