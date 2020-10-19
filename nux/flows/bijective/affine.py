import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
from nux.flows.base import *
import nux.util as util

__all__ = ["Identity",
           "Scale",
           "AffineDense",
           "AffineLDU",
           "AffineSVD",
           "OneByOneConv",
           "LocalDense"]

################################################################################################################

class Identity(Layer):

  def __init__(self, name: str="identity", **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    return {"x": inputs["x"], "log_det": jnp.array(0.0)}

################################################################################################################

class Scale(Layer):

  def __init__(self, tau, name: str="scale", **kwargs):
    super().__init__(name=name, **kwargs)
    self.tau = tau

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    outputs = {}

    if sample == False:
      outputs["x"] = inputs["x"]/self.tau
    else:
      outputs["x"] = inputs["x"]*self.tau

    shape = self.get_unbatched_shapes(sample)["x"]
    outputs["log_det"] = jnp.ones(self.batch_shape)
    outputs["log_det"] *= -jnp.log(self.tau)*jnp.prod(jnp.array(shape))

    return outputs

################################################################################################################

class AffineDense(Layer):

  def __init__(self,
               name: str="affine_dense",
               **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    outputs = {}

    x_dim, dtype = x.shape[-1], inputs["x"].dtype
    W_init = hk.initializers.TruncatedNormal(1/jnp.sqrt(x_dim))
    W = hk.get_parameter("W", shape=(x_dim, x_dim), dtype=dtype, init=W_init)
    b = hk.get_parameter("b", shape=(x_dim,), dtype=dtype, init=jnp.zeros)

    if sample == False:
      outputs["x"] = jnp.dot(x, W.T) + b
    else:
      w_inv = jnp.linalg.inv(W)
      outputs["x"] = jnp.dot(x - b, w_inv.T)

    outputs["log_det"] = jnp.linalg.slogdet(W)[1]*jnp.ones(self.batch_shape)

    return outputs

################################################################################################################

class AffineLDU(Layer):

  def __init__(self,
               name: str="affine_ldu",
               **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
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
      z = jnp.dot(x, (U*lower_mask.T).T) + x
      z *= jnp.exp(log_d)
      z = jnp.dot(z, (L*lower_mask).T) + z
      outputs["x"] = z + b
    else:
      z = inputs["x"]

      @self.auto_batch
      def invert(z):
        x = util.L_solve(L, z - b)
        x = x*jnp.exp(-log_d)
        return util.U_solve(U, x)

      outputs["x"] = invert(z)

    outputs["log_det"] = jnp.sum(log_d, axis=-1)*jnp.ones(self.batch_shape)
    return outputs

################################################################################################################

class AffineSVD(Layer):

  def __init__(self, n_householders: int, name: str="affine_svd", **kwargs):
    super().__init__(name=name, **kwargs)
    self.n_householders = n_householders

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    outputs = {}

    dim, dtype = inputs["x"].shape[-1], inputs["x"].dtype
    init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal')
    U     = hk.get_parameter("U", shape=(self.n_householders, dim), dtype=dtype, init=init)
    VT    = hk.get_parameter("U", shape=(self.n_householders, dim), dtype=dtype, init=init)
    log_s = hk.get_parameter("log_s", shape=(dim,), dtype=dtype, init=jnp.zeros)

    b = hk.get_parameter("b", shape=(dim,), dtype=dtype, init=jnp.zeros)

    if sample == False:
      x = inputs["x"]

      @self.auto_batch
      def forward(x):
        O = jnp.eye(x.size) - 2*VT.T@jnp.linalg.inv(VT@VT.T)@VT
        l = x - 2*VT.T@jnp.linalg.inv(VT@VT.T)@VT@x
        z = util.householder_prod(x, VT)
        z = z*jnp.exp(log_s)
        return util.householder_prod(z, U) + b

      outputs["x"] = forward(x)
    else:
      z = inputs["x"]

      @self.auto_batch
      def inverse(z):
        x = util.householder_prod_transpose(z - b, U)
        x = x*jnp.exp(-log_s)
        return util.householder_prod_transpose(x, VT)

      outputs["x"] = inverse(z)

    outputs["log_det"] = log_s.sum()*jnp.ones(self.batch_shape)
    return outputs

################################################################################################################

class OneByOneConv(Layer):

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
    height, width, channel = inputs["x"].shape[-3:]

    dtype = inputs["x"].dtype
    W = hk.get_parameter("W", shape=(channel, channel), dtype=dtype, init=self.W_init)
    b = hk.get_parameter("b", shape=(channel,), dtype=dtype, init=jnp.zeros)

    if self.weight_norm:
      W *= jax.lax.rsqrt(jnp.sum(W**2, axis=0))

    @partial(self.auto_batch, in_axes=(None, 0), expected_depth=1)
    def conv(W, x):
      return jax.lax.conv_general_dilated(x,
                                          W[None,None,...],
                                          (1, 1),
                                          'SAME',
                                          (1, 1),
                                          (1, 1),
                                          dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    if sample == False:
      x = inputs["x"]
      z = conv(W, x)
      outputs["x"] = z + b
    else:
      W_inv = jnp.linalg.inv(W)
      z = inputs["x"]
      x = conv(W_inv, z - b)
      outputs["x"] = x

    outputs["log_det"] = jnp.linalg.slogdet(W)[1]*height*width*jnp.ones(self.batch_shape)

    return outputs

################################################################################################################

class LocalDense(Layer):

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
    x_dtype = inputs["x"].dtype
    h, w, c = inputs["x"].shape[-3:]
    fh, fw = self.filter_shape
    dh, dw = self.dilation

    # Find the shape of the dilated_squeeze output
    H_sq, W_sq, C_sq = (h//fh, w//fw, c*fh*fw)

    W = hk.get_parameter("W", shape=(C_sq, C_sq), dtype=x_dtype, init=self.W_init)
    b = hk.get_parameter("b", shape=(c,), dtype=x_dtype, init=jnp.zeros)

    if sample == False:
      x = inputs["x"]
      @self.auto_batch
      def forward(x):
        x = util.dilated_squeeze(x, self.filter_shape, self.dilation)
        z = jax.lax.conv_general_dilated(x[None],
                                         W[None,None,...],
                                         (1, 1),
                                         'SAME',
                                         (1, 1),
                                         (1, 1),
                                         dimension_numbers=('NHWC', 'HWIO', 'NHWC'))[0]
        return util.dilated_unsqueeze(z, self.filter_shape, self.dilation) + b
      outputs["x"] = forward(x)
    else:
      W_inv = jnp.linalg.inv(W)
      z = inputs["x"]
      @self.auto_batch
      def inverse(z):
        zmb = util.dilated_squeeze(z - b, self.filter_shape, self.dilation)
        x = jax.lax.conv_general_dilated(zmb[None],
                                         W_inv[None,None,...],
                                         (1, 1),
                                         'SAME',
                                         (1, 1),
                                         (1, 1),
                                         dimension_numbers=('NHWC', 'HWIO', 'NHWC'))[0]
        return util.dilated_unsqueeze(x, self.filter_shape, self.dilation)
      outputs["x"] = inverse(z)

    outputs["log_det"] = jnp.linalg.slogdet(W)[1]*h*w*jnp.ones(self.batch_shape)

    return outputs
