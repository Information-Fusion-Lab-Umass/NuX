import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence
from nux.flows.base import *
import nux.util as util

__all__ = ["Coupling",
           "ConditionedCoupling"]

class Coupling(AutoBatchedLayer):

  def __init__(self,
               create_network: Optional[Callable]=None,
               hidden_layer_sizes: Optional[Sequence[int]]=[1024]*4,
               n_channels: Optional[int]=256,
               kind: Optional[str]="affine",
               axis: Optional[int]=-1,
               name: str="coupling",
               **kwargs
  ):
    super().__init__(name=name, **kwargs)
    self.hidden_layer_sizes = hidden_layer_sizes
    self.n_channels         = n_channels
    self.kind               = kind
    self.axis               = axis
    self.create_network     = None

  def get_network(self, out_shape):
    if self.create_network is not None:
      return self.create_network(out_shape)
    if len(out_shape) == 1:
      return util.SimpleMLP(out_shape, self.hidden_layer_sizes, is_additive=self.kind=="additive")
    else:
      assert len(out_shape) == 3
      return util.SimpleConv(out_shape, self.n_channels, is_additive=self.kind=="additive")

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    x_shape = x.shape
    ax = self.axis%len(x_shape)
    split_index = x_shape[ax]//2
    xa, xb = jnp.split(x, indices_or_sections=jnp.array([split_index]), axis=self.axis)

    # Initialize the coupling layer to the identity
    scale_scale = hk.get_parameter("scale_scale", shape=xa.shape, dtype=xa.dtype, init=jnp.zeros)
    scale_shift = hk.get_parameter("scale_shift", shape=xa.shape, dtype=xa.dtype, init=jnp.zeros)
    network = self.get_network(xa.shape)

    # Apply the transformation
    if self.kind == "affine":
      t, log_s = network(xb)
      t = scale_shift*t
      log_s = scale_scale*log_s

      if sample == False:
          za = (xa - t)*jnp.exp(-log_s)
      else:
          za = xa*jnp.exp(log_s) + t
      log_det = -jnp.sum(log_s)
    else:
      t = network(xb)
      if sample == False:
          za = xa - t
      else:
          za = xa + t
      log_det = jnp.array(0.0)

    # Recombine
    z = jnp.concatenate([za, xb], axis=self.axis)

    outputs = {"x": z, "log_det": log_det}
    return outputs

################################################################################################################

class ConditionedCoupling(AutoBatchedLayer):

  def __init__(self,
               create_network: Optional[Callable]=None,
               hidden_layer_sizes: Optional[Sequence[int]]=[1024]*4,
               n_channels: Optional[int]=256,
               kind: Optional[str]="affine",
               axis: Optional[int]=-1,
               name: str="coupling",
               **kwargs
  ):
    super().__init__(name=name, **kwargs)
    self.hidden_layer_sizes = hidden_layer_sizes
    self.n_channels         = n_channels
    self.kind               = kind
    self.axis               = axis
    self.create_network     = None

  def get_network(self, out_shape):
    if self.create_network is not None:
      return self.create_network(out_shape)
    if len(out_shape) == 1:
      return util.SimpleMLP(out_shape, self.hidden_layer_sizes, is_additive=self.kind=="additive")
    else:
      assert len(out_shape) == 3
      return util.SimpleConv(out_shape, self.n_channels, is_additive=self.kind=="additive")

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x, condition = inputs["x"], inputs["condition"]
    x_shape = x.shape
    ax = self.axis%len(x_shape)
    split_index = x_shape[ax]//2
    xa, xb = jnp.split(x, indices_or_sections=jnp.array([split_index]), axis=self.axis)

    network = self.get_network(xa.shape)
    network_input = jnp.concatenate([xb, condition], axis=self.axis)

    # Apply the transformation
    if self.kind == "affine":
      t, log_s = network(network_input)
      if sample == False:
          za = (xa - t)*jnp.exp(-log_s)
      else:
          za = xa*jnp.exp(log_s) + t
      log_det = -jnp.sum(log_s)
    else:
      t = network(network_input)
      if sample == False:
          za = xa - t
      else:
          za = xa + t
      log_det = jnp.array(0.0)

    # Recombine
    z = jnp.concatenate([za, xb], axis=self.axis)

    outputs = {"x": z, "log_det": log_det}
    return outputs

################################################################################################################
