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

__all__ = ["Coupling"]

class Coupling(Layer):

  def __init__(self,
               create_network: Optional[Callable]=None,
               kind: Optional[str]="affine",
               axis: Optional[int]=-1,
               split_kind: str="channel",
               use_condition: bool=False,
               name: str="coupling",
               network_kwargs: Optional=None,
               **kwargs
  ):
    super().__init__(name=name, **kwargs)
    self.kind               = kind
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

    out_dim = out_shape[-1] if self.kind == "additive" else 2*out_shape[-1]

    # Otherwise, use default networks
    if len(out_shape) == 1:
      network_kwargs = self.network_kwargs
      if network_kwargs is None:

        network_kwargs = dict(layer_sizes=[128]*4,
                              nonlinearity="relu",
                              parameter_norm="weight_norm")
      network_kwargs["out_dim"] = out_dim

      return net.MLP(**network_kwargs)

    else:
      network_kwargs = self.network_kwargs
      if network_kwargs is None:

        network_kwargs = dict(n_blocks=2,
                              hidden_channel=16,
                              nonlinearity="relu",
                              normalization="instance_norm",
                              parameter_norm="weight_norm",
                              block_type="reverse_bottleneck",
                              squeeze_excite=True)
      network_kwargs["out_channel"] = out_dim

      return net.ResNet(**network_kwargs)

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
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
    xa_shape = xa.shape[len(self.batch_shape):]
    xb_shape = xb.shape[len(self.batch_shape):]

    # Initialize the coupling layer to the identity
    scale_init = hk.initializers.RandomNormal(stddev=0.01)
    if self.kind == "affine":
      scale_scale = hk.get_parameter("scale_scale", shape=(), dtype=x.dtype, init=scale_init)
    shift_scale = hk.get_parameter("shift_scale", shape=(), dtype=x.dtype, init=scale_init)
    network = self.get_network(xa_shape)

    # Initialize parameters for the other half of the input so that the transformation is consistent
    if self.kind == "affine":
      log_s_b = hk.get_parameter("log_s_b", shape=xb_shape, dtype=x.dtype, init=scale_init)
    t_b = hk.get_parameter("t_b", shape=xb_shape, dtype=x.dtype, init=scale_init)

    # Apply the transformation
    if self.kind == "affine":

      # First invert the part of the input that gets conditioned on
      if sample == False:
        zb = (xb - t_b)*jnp.exp(-log_s_b)
        network_in = jnp.concatenate([xb, condition], axis=self.axis) if self.use_condition else xb
      else:
        zb = xb*jnp.exp(log_s_b) + t_b
        network_in = jnp.concatenate([zb, condition], axis=self.axis) if self.use_condition else zb

      # Run the conditioner network
      network_out = self.auto_batch(network, expected_depth=1)(network_in)

      # Split the output and bound the scaling term
      t_a, log_s_a = jnp.split(network_out, 2, axis=self.axis)
      log_s_a = jnp.tanh(log_s_a)

      # Scale the parameters so that we can initialize this function to the identity
      t_a = shift_scale*t_a
      log_s_a = scale_scale*log_s_a

      # Apply the reult to the other half of the input
      if sample == False:
        za = (xa - t_a)*jnp.exp(-log_s_a)
      else:
        za = xa*jnp.exp(log_s_a) + t_a

      log_det = jnp.ones(self.batch_shape)
      log_det *= -(jnp.sum(log_s_a) + jnp.sum(log_s_b))
    else:
      # First invert the part of the input that gets conditioned on
      if sample == False:
        zb = xb - t_b
        network_in = jnp.concatenate([xb, condition], axis=self.axis) if self.use_condition else xb
      else:
        zb = xb + t_b
        network_in = jnp.concatenate([zb, condition], axis=self.axis) if self.use_condition else zb

      # Run the conditioner network
      t_a = self.auto_batch(network, expected_depth=1)(network_in)
      t_a = shift_scale*t_a

      # Apply the reult to the other half of the input
      if sample == False:
        za = xa - t_a
      else:
        za = xa + t_a
      log_det = jnp.zeros(self.batch_shape)

    # Recombine
    z = jnp.concatenate([za, zb], axis=self.axis)

    if(self.split_kind == "checkerboard"):
      z = self.auto_batch(util.dilated_unsqueeze)(z)

    outputs = {"x": z, "log_det": log_det}
    return outputs

################################################################################################################
