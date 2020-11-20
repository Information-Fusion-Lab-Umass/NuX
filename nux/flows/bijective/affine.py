import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
from nux.flows.base import *
import nux.util as util

__all__ = ["Bias",
           "Identity",
           "Scale",
           "AffineDense",
           "AffineLDU",
           "AffineSVD",
           "OneByOneConv"]

################################################################################################################

class Bias(Layer):

  def __init__(self,
               axis: int=-1,
               name: str="bias"
  ):
    """ Adds a scalar to the input
    Args:
      axis: Which axis of the input to apply to
      name: Optional name for this module.
    """
    super().__init__(name=name)
    self.axis = axis

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    b = hk.get_parameter("b", shape=(x.shape[self.axis],), dtype=x.dtype, init=jnp.zeros)
    if sample:
      z = x + b
    else:
      z = x - b
    return {"x": z, "log_det": jnp.zeros(self.batch_shape)}

class Identity(Layer):

  def __init__(self,
               name: str="identity"
  ):
    """ No-op
    Args:
      name: Optional name for this module.
    """
    super().__init__(name=name)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    return {"x": inputs["x"], "log_det": jnp.zeros(self.batch_shape)}

################################################################################################################

class Scale(Layer):

  def __init__(self,
               scale: float,
               name: str="scale"
  ):
    """ Scale an input by a specified scalar
    Args:
      scale: Value to scale by
      name : Optional name for this module.
    """
    super().__init__(name=name)
    self.scale = scale

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    outputs = {}

    if sample == False:
      outputs["x"] = inputs["x"]/self.scale
    else:
      outputs["x"] = inputs["x"]*self.scale

    shape = self.get_unbatched_shapes(sample)["x"]
    outputs["log_det"] = jnp.ones(self.batch_shape)
    outputs["log_det"] *= -jnp.log(self.scale)*util.list_prod(shape)

    return outputs

################################################################################################################

class AffineDense(Layer):

  def __init__(self,
               name: str="affine_dense"
  ):
    """ Apply a dense matrix multiplication.  Costs O(D^3).
    Args:
      name:  Optional name for this module.
    """
    super().__init__(name=name)

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
               name: str="affine_ldu"
  ):
    """ LDU parametrized matrix multiplication.  Costs O(D^2) to invert and O(D) for a regular pass.
    Args:
      name:  Optional name for this module.
    """
    super().__init__(name=name)

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

    # Its way faster to allocate a full matrix for L and U and then mask than it
    # is to allocate only the lower/upper parts and the reshape.
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

  def __init__(self,
               n_householders: int,
               name: str="affine_svd"
  ):
    """ SVD parametrized matrix multiplication.  Costs O(K*D) where K is the number of householders.
    Args:
      n_householders: Number of householders to parametrize U and VT.
      name          : Optional name for this module.
    """
    super().__init__(name=name)
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

    # TODO: Implement block householders instead of sequential householders.
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

  def __init__(self,
               weight_norm: bool=True,
               name: str="1x1_conv"
  ):
    """ 1x1 convolution.  Uses a dense parametrization because the channel dimension will probably
        never be that big.  Costs O(C^3).  Used in GLOW https://arxiv.org/pdf/1807.03039.pdf
    Args:
      weight_norm: Should weight norm be applied to the layer?
      name       : Optional name for this module.
    """
    super().__init__(name=name)
    self.weight_norm = weight_norm

    def orthogonal_init(shape, dtype):
      key = hk.next_rng_key()
      W = random.normal(key, shape=shape, dtype=dtype)
      return util.whiten(W)
    self.W_init = orthogonal_init

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    outputs = {}
    x = inputs["x"]
    height, width, channel = x.shape[-3:]

    # Using lax.conv instead of matrix multiplication over the channel dimension
    # is faster and also more numerically stable for some reason.
    @partial(self.auto_batch, in_axes=(None, 0), expected_depth=1)
    def conv(W, x):
      return jax.lax.conv_general_dilated(x,
                                          W[None,None,...],
                                          (1, 1),
                                          'SAME',
                                          (1, 1),
                                          (1, 1),
                                          dimension_numbers=('NHWC', 'HWIO', 'NHWC'))

    dtype = x.dtype
    W = hk.get_parameter("W", shape=(channel, channel), dtype=dtype, init=self.W_init)

    # Initialize with weight norm https://arxiv.org/pdf/1602.07868.pdf
    # This seems to improve performance.
    if self.weight_norm and x.ndim > 3:
      W *= jax.lax.rsqrt(jnp.sum(W**2, axis=0))

      def g_init(shape, dtype):
        t = conv(W, x)
        g = 1/(jnp.std(t, axis=(0, 1, 2)) + 1e-5)
        return g

      def b_init(shape, dtype):
        t = conv(W, x)
        return -jnp.mean(t, axis=(0, 1, 2))/(jnp.std(t, axis=(0, 1, 2)) + 1e-5)

      g = hk.get_parameter("g", (channel,), dtype, init=g_init)
      b = hk.get_parameter("b", (channel,), dtype, init=b_init)

      W *= g

    else:
      b = hk.get_parameter("b", shape=(channel,), dtype=dtype, init=jnp.zeros)

    # Run the flow
    if sample == False:
      z = conv(W, x)
      outputs["x"] = z + b
    else:
      W_inv = jnp.linalg.inv(W)
      outputs["x"] = conv(W_inv, x - b)

    outputs["log_det"] = jnp.linalg.slogdet(W)[1]*height*width*jnp.ones(self.batch_shape)

    return outputs
