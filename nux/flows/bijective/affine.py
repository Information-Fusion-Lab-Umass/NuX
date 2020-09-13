import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Tuple, Sequence, Union, Any
from nux.flows.base import *
import nux.util as util

__all__ = ["Identity",
           "Scale",
           "AffineDense",
           "AffineLDU",
           "AffineSVD",
           "OneByOneConv",
           "LocalDense",
           "SmoothHeightWidth",
           "ConstantConv"]

################################################################################################################

class Identity(AutoBatchedLayer):

  def __init__(self, name: str="identity", **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, **kwargs) -> Mapping[str, jnp.ndarray]:
    return {"x": inputs["x"], "log_det": jnp.array(0.0)}

################################################################################################################

class Scale(AutoBatchedLayer):

  def __init__(self, tau, name: str="scale", **kwargs):
    super().__init__(name=name, **kwargs)
    self.tau = tau

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
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

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
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

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
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

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
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

  def __init__(self, weight_norm: bool=True, name: str="one_by_one_conv", **kwargs):
    super().__init__(name=name, **kwargs)
    self.weight_norm = weight_norm

    def orthogonal_init(shape, dtype):
      key = hk.next_rng_key()
      W = random.normal(key, shape=shape, dtype=dtype)
      return util.whiten(W)
    self.W_init = orthogonal_init

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    outputs = {}
    height, width, channel = inputs["x"].shape

    shape, dtype = inputs["x"].shape, inputs["x"].dtype
    W = hk.get_parameter("W", shape=(channel, channel), dtype=dtype, init=self.W_init)
    b = hk.get_parameter("b", shape=shape, dtype=dtype, init=jnp.zeros)

    if(self.weight_norm):
      W *= jax.lax.rsqrt(jnp.sum(W**2, axis=0))

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

################################################################################################################

class LocalDense(AutoBatchedLayer):

  def __init__(self,
               filter_shape: Tuple[int]=(2, 2),
               dilation: Tuple[int]=(1, 1),
               name: str="local_dense",
               W_init=None,
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.filter_shape = filter_shape
    self.dilation = dilation
    self.W_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal') if W_init is None else W_init

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:
    outputs = {}
    x_shape, x_dtype = inputs["x"].shape, inputs["x"].dtype
    h, w, c = inputs["x"].shape
    fh, fw = self.filter_shape
    dh, dw = self.dilation

    # Find the shape of the dilated_squeeze output
    H_sq, W_sq, C_sq = (h//fh, w//fw, c*fh*fw)

    W = hk.get_parameter("W", shape=(C_sq, C_sq), dtype=x_dtype, init=self.W_init)
    b = hk.get_parameter("b", shape=x_shape, dtype=x_dtype, init=jnp.zeros)*0.0

    if sample == False:
      x = inputs["x"]
      x = util.dilated_squeeze(x, self.filter_shape, self.dilation)
      z = jax.lax.conv_general_dilated(x[None],
                                       W[None,None,...],
                                       (1, 1),
                                       'SAME',
                                       (1, 1),
                                       (1, 1),
                                       dimension_numbers=('NHWC', 'HWIO', 'NHWC'))[0]
      z = util.dilated_unsqueeze(z, self.filter_shape, self.dilation)
      outputs["x"] = z + b
    else:
      W_inv = jnp.linalg.inv(W)
      z = inputs["x"]
      zmb = util.dilated_squeeze(z - b, self.filter_shape, self.dilation)
      x = jax.lax.conv_general_dilated(zmb[None],
                                       W_inv[None,None,...],
                                       (1, 1),
                                       'SAME',
                                       (1, 1),
                                       (1, 1),
                                       dimension_numbers=('NHWC', 'HWIO', 'NHWC'))[0]
      x = util.dilated_unsqueeze(x, self.filter_shape, self.dilation)
      outputs["x"] = x

    outputs["log_det"] = jnp.linalg.slogdet(W)[1]*h*w

    return outputs

################################################################################################################

class ConstantConv(AutoBatchedLayer):

  def __init__(self,
               filter_shape: Tuple[int]=(2, 2),
               name: str="constant_conv",
               W_init=None,
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.filter_shape = filter_shape
    self.W_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal') if W_init is None else W_init

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:

    outputs = {}
    x_shape, x_dtype = inputs["x"].shape, inputs["x"].dtype
    h, w, c = inputs["x"].shape
    fh, fw = self.filter_shape

    W = hk.get_state("W", shape=(fh, fw, c, c), dtype=x_dtype, init=self.W_init)
    b = hk.get_state("b", shape=x_shape, dtype=x_dtype, init=jnp.zeros)*0.0

    if sample == False:
      x = inputs["x"]
      z = jax.lax.conv_general_dilated(x[None],
                                       W,
                                       (1, 1),
                                       'SAME',
                                       (1, 1),
                                       (1, 1),
                                       dimension_numbers=('NHWC', 'HWIO', 'NHWC'))[0]
      outputs["x"] = z + b
    else:
      z = inputs["x"]
      zmb = z - b

      W_ = W.transpose((2, 3, 0, 1))
      z_ = zmb.transpose((2, 0, 1))

      x, r_sq, iters = util.CTC_solve(W_, (0, 1, 0, 1), (1, 1), 1000, z_, jnp.zeros_like(z_))
      outputs["x"] = x.transpose((1, 2, 0))

    outputs["log_det"] = jnp.array(0.0)

    return outputs

################################################################################################################

class SmoothHeightWidth(ConstantConv):

  def __init__(self,
               filter_shape: Sequence[int]=(2, 2),
               name: str="smooth_height_width",
               **kwargs):

    def W_init(shape, dtype):
      h, w, cin, cout = shape
      return jnp.broadcast_to(jnp.eye(cin, cout)[None,None], shape).astype(dtype)/(h*w)

    super().__init__(filter_shape=filter_shape, name=name, W_init=W_init, **kwargs)

################################################################################################################
