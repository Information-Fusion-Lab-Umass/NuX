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

__all__ = ["Coupling",
           "ConditionedCoupling"]

class Coupling(AutoBatchedLayer):

  def __init__(self,
               create_network: Optional[Callable]=None,
               layer_sizes: Optional[Sequence[int]]=[1024]*4,
               n_channels: Optional[int]=256,
               kind: Optional[str]="additive",
               parameter_norm: Optional[str]=None,
               axis: Optional[int]=-1,
               name: str="coupling",
               **kwargs
  ):
    super().__init__(name=name, **kwargs)
    self.layer_sizes        = layer_sizes
    self.n_channels         = n_channels
    self.kind               = kind
    self.axis               = axis
    self.create_network     = create_network
    self.parameter_norm     = parameter_norm

  def get_network(self, out_shape):
    # The user can specify a custom network
    if self.create_network is not None:
      return self.create_network(out_shape)

    out_dim = out_shape[-1] if self.kind == "additive" else 2*out_shape[-1]

    # Otherwise, use default networks
    if len(out_shape) == 1:
      return net.MLP(out_dim=out_dim,
                     layer_sizes=self.layer_sizes,
                     parameter_norm=self.parameter_norm,
                     nonlinearity="relu")

    else:

      return net.ConvBlock(out_channel=out_dim,
                           hidden_channel=self.n_channels,
                           parameter_norm=self.parameter_norm,
                           nonlinearity="relu")

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    x_shape = x.shape
    ax = self.axis%len(x_shape)
    split_index = x_shape[ax]//2
    xa, xb = jnp.split(x, indices_or_sections=jnp.array([split_index]), axis=self.axis)

    # Initialize the coupling layer to the identity
    scale_scale = hk.get_parameter("scale_scale", shape=xa.shape, dtype=xa.dtype, init=jnp.zeros)
    shift_scale = hk.get_parameter("shift_scale", shape=xa.shape, dtype=xa.dtype, init=jnp.zeros)
    network = self.get_network(xa.shape)

    # Apply the transformation
    if self.kind == "affine":
      network_out = network(xb)

      # Split the output and bound the scaling term
      t, log_s = jnp.split(network_out, 2, axis=-1)
      log_s = jnp.tanh(log_s)

      # Scale the parameters so that we can initialize this function to the identity
      t = shift_scale*t
      log_s = scale_scale*log_s

      if sample == False:
          za = (xa - t)*jnp.exp(-log_s)
      else:
          za = xa*jnp.exp(log_s) + t
      log_det = -jnp.sum(log_s)
    else:
      t = network(xb)
      t = shift_scale*t

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
               layer_sizes: Optional[Sequence[int]]=[1024]*4,
               n_channels: Optional[int]=256,
               kind: Optional[str]="affine",
               parameter_norm: Optional[str]=None,
               axis: Optional[int]=-1,
               name: str="conditioned_coupling",
               **kwargs
  ):
    super().__init__(name=name, **kwargs)
    self.layer_sizes = layer_sizes
    self.n_channels         = n_channels
    self.kind               = kind
    self.axis               = axis
    self.create_network     = create_network
    self.parameter_norm     = parameter_norm

  def get_network(self, out_shape):
    # The user can specify a custom network
    if self.create_network is not None:
      return self.create_network(out_shape)

    out_dim = out_shape[-1] if self.kind == "additive" else 2*out_shape[-1]

    # Otherwise, use default networks
    if len(out_shape) == 1:
      return net.MLP(out_dim=out_dim,
                     layer_sizes=self.layer_sizes,
                     parameter_norm=self.parameter_norm,
                     nonlinearity="relu")

    else:

      return net.ConvBlock(out_channel=out_dim,
                           hidden_channel=self.n_channels,
                           parameter_norm=self.parameter_norm,
                           nonlinearity="relu")

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x, condition = inputs["x"], inputs["condition"]
    x_shape = x.shape
    ax = self.axis%len(x_shape)
    split_index = x_shape[ax]//2
    xa, xb = jnp.split(x, indices_or_sections=jnp.array([split_index]), axis=self.axis)

    scale_scale = hk.get_parameter("scale_scale", shape=xa.shape, dtype=xa.dtype, init=jnp.zeros)
    shift_scale = hk.get_parameter("shift_scale", shape=xa.shape, dtype=xa.dtype, init=jnp.zeros)

    network = self.get_network(xa.shape)
    network_input = jnp.concatenate([xb, condition], axis=self.axis)

    # Apply the transformation
    if self.kind == "affine":
      network_out = network(network_input)

      # Split the output and bound the scaling term
      t, log_s = jnp.split(network_out, 2, axis=-1)
      log_s = jnp.tanh(log_s)

      # Scale the parameters so that we can initialize this function to the identity
      t = shift_scale*t
      log_s = scale_scale*log_s

      if sample == False:
          za = (xa - t)*jnp.exp(-log_s)
      else:
          za = xa*jnp.exp(log_s) + t
      log_det = -jnp.sum(log_s)
    else:
      t = network(network_input)
      t = shift_scale*t

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
