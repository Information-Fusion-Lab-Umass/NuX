import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping
from nux.flows.base import *
import nux.util as util

__all__ = ["Identity",
           "Scale",
           "AffineDense",
           "AffineLDU",
           "AffineSVD",
           "OneByOneConv"]

################################################################################################################

class Identity(AutoBatchedLayer):

  def __init__(self, name: str="identity", **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self, inputs: Mapping[str, jnp.ndarray], **kwargs) -> Mapping[str, jnp.ndarray]:
    return {"x": inputs["x"], "log_det": jnp.array(0.0)}

################################################################################################################

class Scale(AutoBatchedLayer):

  def __init__(self, tau, name: str="scale", **kwargs):
    super().__init__(name=name, **kwargs)
    self.tau = tau

  def call(self, inputs: Mapping[str, jnp.ndarray], sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    outputs = {}

    if sample == False:
      outputs["x"] = inputs["x"]/self.tau
    else:
      outputs["x"] = inputs["x"]*self.tau

    shape = util.tree_shapes(inputs)
    outputs["log_det"] = -jnp.log(self.tau)*jnp.prod(jnp.array(shape["x"]))

    return outputs

################################################################################################################

class AffineDense(AutoBatchedLayer):

  def __init__(self, name: str="affine_dense", **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self, inputs: Mapping[str, jnp.ndarray], sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    outputs = {}

    W_init = hk.initializers.TruncatedNormal(1/jnp.sqrt(x.shape[-1]))
    W = hk.get_parameter("W", shape=(x.shape[-1], x.shape[-1]), dtype=inputs["x"].dtype, init=W_init)
    b = hk.get_parameter("b", shape=(x.shape[-1],), dtype=inputs["x"].dtype, init=jnp.zeros)

    if sample == False:
      outputs["x"] = W@x + b
    else:
      w_inv = jnp.linalg.inv(W)
      outputs["x"] = w_inv@(x - b)

    outputs["log_det"] = jnp.linalg.slogdet(W)[1]

    return outputs

################################################################################################################

class AffineLDU(AutoBatchedLayer):

  def __init__(self, name: str="affine_ldu", **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self, inputs: Mapping[str, jnp.ndarray], sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    outputs = {}

    init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal')

    dim, dtype = inputs["x"].shape[-1], inputs["x"].dtype
    L     = hk.get_parameter("L", shape=(dim, dim), dtype=dtype, init=init)
    U     = hk.get_parameter("U", shape=(dim, dim), dtype=dtype, init=init)
    log_d = hk.get_parameter("log_d", shape=(dim,), dtype=dtype, init=jnp.zeros)
    lower_mask = jnp.ones((dim, dim), dtype=bool)
    lower_mask = jax.ops.index_update(lower_mask, jnp.triu_indices(dim), False)

    b = hk.get_parameter("b", shape=(dim,), dtype=dtype, init=jnp.zeros)

    if sample == False:
      x = inputs["x"]
      z = (U*lower_mask.T)@x + x
      z *= jnp.exp(log_d)
      z = (L*lower_mask)@z + z
      outputs["x"] = z + b
    else:
      z = inputs["x"]
      x = util.L_solve(L, z - b)
      x = x*jnp.exp(-log_d)
      outputs["x"] = util.U_solve(U, x)

    outputs["log_det"] = jnp.sum(log_d, axis=-1)
    return outputs

################################################################################################################

class AffineSVD(AutoBatchedLayer):

  def __init__(self, n_householders: int, name: str="affine_svd", **kwargs):
    super().__init__(name=name, **kwargs)
    self.n_householders = n_householders

  def call(self, inputs: Mapping[str, jnp.ndarray], sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    outputs = {}

    dim, dtype = inputs["x"].shape[-1], inputs["x"].dtype
    init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal')
    U     = hk.get_parameter("U", shape=(self.n_householders, dim), dtype=dtype, init=init)
    VT    = hk.get_parameter("U", shape=(self.n_householders, dim), dtype=dtype, init=init)
    log_s = hk.get_parameter("log_s", shape=(dim,), dtype=dtype, init=jnp.zeros)

    b = hk.get_parameter("b", shape=(dim,), dtype=dtype, init=jnp.zeros)

    if sample == False:
      x = inputs["x"]
      z = util.householder_prod(x, VT)
      z = z*jnp.exp(log_s)
      outputs["x"] = util.householder_prod(z, U) + b
    else:
      z = inputs["x"]
      x = util.householder_prod_transpose(z - b, U)
      x = x*jnp.exp(-log_s)
      outputs["x"] = util.householder_prod_transpose(x, VT)

    outputs["log_det"] = log_s.sum()
    return outputs

################################################################################################################

class OneByOneConv(AutoBatchedLayer):

  def __init__(self, name: str="one_by_one_conv", **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self, inputs: Mapping[str, jnp.ndarray], sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    outputs = {}
    height, width, channel = inputs["x"].shape

    shape, dtype = inputs["x"].shape, inputs["x"].dtype
    init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal')
    W = hk.get_parameter("W", shape=(channel, channel), dtype=dtype, init=init)
    b = hk.get_parameter("b", shape=shape, dtype=dtype, init=jnp.zeros)

    if sample == False:
      x = inputs["x"]
      z = jax.lax.conv_general_dilated(x[None],
                                       W[None,None,...],
                                       (1, 1),
                                       'SAME',
                                       (1, 1),
                                       (1, 1),
                                       dimension_numbers=('NHWC', 'HWIO', 'NHWC'))[0]
      outputs["x"] = z + b
    else:
      W_inv = jnp.linalg.inv(W)
      z = inputs["x"]
      x = jax.lax.conv_general_dilated((z - b)[None],
                                       W_inv[None,None,...],
                                       (1, 1),
                                       'SAME',
                                       (1, 1),
                                       (1, 1),
                                       dimension_numbers=('NHWC', 'HWIO', 'NHWC'))[0]
      outputs["x"] = x

    outputs["log_det"] = jnp.linalg.slogdet(W)[1]*height*width

    return outputs
