import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
from nux.internal.layer import InvertibleLayer
import nux.util as util
import nux.util.weight_initializers as init

__all__ = ["Bias",
           "Identity",
           "Scale",
           "ShiftScale",
           "ElementwiseScale",
           "AffineDense",
           "AffineLDU",
           "AffineSVD"]

################################################################################################################

class Bias(InvertibleLayer):

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

class Identity(InvertibleLayer):

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

class ShiftScale(InvertibleLayer):

  def __init__(self,
               axis=-1,
               name: str="shift_scale"
  ):
    """ Elementwise shift + scale
    Args:
      axis: Axes to apply to
      name: Optional name for this module.
    """
    super().__init__(name=name)
    self.axes = (axis,) if isinstance(axis, int) else axis
    for ax in self.axes:
      assert ax < 0, "For convenience, pass in negative indexed axes"

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    outputs = {}
    x = inputs["x"]
    x_shape = self.get_unbatched_shapes(sample)["x"]

    param_shape = tuple([x_shape[ax] for ax in self.axes])
    b     = hk.get_parameter("b", shape=param_shape, dtype=x.dtype, init=jnp.zeros)
    log_s = hk.get_parameter("log_s", shape=param_shape, dtype=x.dtype, init=jnp.zeros)

    s = util.proximal_relu(log_s) + 1e-5

    if sample == False:
      outputs["x"] = (x - b)/s
    else:
      outputs["x"] = s*x + b

    log_det = -jnp.log(s).sum()*jnp.ones(self.batch_shape)
    outputs["log_det"] = log_det

    return outputs

################################################################################################################

class Scale(InvertibleLayer):

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
    self.scale = scale*1.0

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

class ElementwiseScale(InvertibleLayer):

  def __init__(self,
               scale: jnp.ndarray=None,
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

    x = inputs["x"]

    if self.scale is None:
      scale = hk.get_parameter("scale", shape=self.unbatched_input_shapes["x"], dtype=x.dtype, init=jnp.zeros)
    else:
      scale = self.scale
      assert self.scale.shape == self.unbatched_input_shapes["x"]

    if sample == False:
      outputs["x"] = inputs["x"]/scale
    else:
      outputs["x"] = inputs["x"]*scale

    shape = self.get_unbatched_shapes(sample)["x"]
    outputs["log_det"] = -jnp.log(scale).sum()

    return outputs

################################################################################################################

class AffineDense(InvertibleLayer):

  def __init__(self,
               weight_norm: bool=True,
               spectral_norm: bool=False,
               max_singular_value: float=1.0,
               max_power_iters: int=1,
               name: str="affine_dense",
               **kwargs
  ):
    """ Apply a dense matrix multiplication.  Costs O(D^3).
    Args:
      name:  Optional name for this module.
    """
    super().__init__(name=name, **kwargs)
    assert (weight_norm and spectral_norm) == False
    self.spectral_norm = spectral_norm
    self.weight_norm = weight_norm
    self.max_singular_value = max_singular_value
    self.max_power_iters = max_power_iters

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    outputs = {}

    x_dim, dtype = x.shape[-1], inputs["x"].dtype

    if self.weight_norm:
      W, b = init.weight_with_weight_norm(x,
                                          out_dim=x_dim,
                                          w_init=hk.initializers.RandomNormal(0.1),
                                          b_init=jnp.zeros,
                                          is_training=kwargs.get("is_training", True),
                                          use_bias=True)
    elif self.spectral_norm:
      W, b = init.weight_with_good_spectral_norm(x,
                                                 out_dim=x_dim,
                                                 w_init=hk.initializers.RandomNormal(0.1),
                                                 b_init=jnp.zeros,
                                                 is_training=kwargs.get("is_training", True),
                                                 update_params=kwargs.get("is_training", True),
                                                 max_singular_value=self.max_singular_value,
                                                 max_power_iters=self.max_power_iters,
                                                 use_bias=True)
    else:
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

tri_solve = jax.scipy.linalg.solve_triangular
L_solve = partial(tri_solve, lower=True, unit_diagonal=True)
U_solve = partial(tri_solve, lower=False, unit_diagonal=True)

class AffineLDU(InvertibleLayer):

  def __init__(self,
               safe_diag: bool=True,
               use_bias: bool=True,
               name: str="affine_ldu"
  ):
    """ LDU parametrized matrix multiplication.  Costs O(D^2) to invert and O(D) for a regular pass.
    Args:
      name:  Optional name for this module.
    """
    super().__init__(name=name)
    self.safe_diag = safe_diag
    self.use_bias = use_bias

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    outputs = {}

    dim, dtype = inputs["x"].shape[-1], inputs["x"].dtype

    L     = hk.get_parameter("L", shape=(dim, dim), dtype=dtype, init=hk.initializers.RandomNormal(0.01))
    U     = hk.get_parameter("U", shape=(dim, dim), dtype=dtype, init=hk.initializers.RandomNormal(0.01))
    log_d = hk.get_parameter("log_d", shape=(dim,), dtype=dtype, init=jnp.zeros)
    lower_mask = jnp.ones((dim, dim), dtype=bool)
    lower_mask = jax.ops.index_update(lower_mask, jnp.triu_indices(dim), False)

    if self.safe_diag:
      d = util.proximal_relu(log_d) + 1e-5
      log_d = jnp.log(d)

    if self.use_bias:
      def b_init(shape, dtype):
        x = inputs["x"]
        if x.ndim == 1:
          return jnp.zeros(shape, dtype=dtype)

        # Initialize to the batch mean
        z = jnp.dot(x, (U*lower_mask.T).T) + x
        z *= jnp.exp(log_d)
        z = jnp.dot(z, (L*lower_mask).T) + z
        b = -jnp.mean(z, axis=0)
        return b

      b = hk.get_parameter("b", shape=(dim,), dtype=dtype, init=b_init)

    # Its way faster to allocate a full matrix for L and U and then mask than it
    # is to allocate only the lower/upper parts and the reshape.
    if sample == False:
      x = inputs["x"]
      z = jnp.dot(x, (U*lower_mask.T).T) + x
      z *= jnp.exp(log_d)
      z = jnp.dot(z, (L*lower_mask).T) + z
      outputs["x"] = z
      if self.use_bias:
        outputs["x"] += b
    else:
      z = inputs["x"]

      @self.auto_batch
      def invert(z):
        if self.use_bias:
          x = L_solve(L, z - b)
        else:
          x = L_solve(L, z)
        x = x*jnp.exp(-log_d)
        return U_solve(U, x)

      outputs["x"] = invert(z)

    outputs["log_det"] = jnp.sum(log_d, axis=-1)*jnp.ones(self.batch_shape)
    return outputs

################################################################################################################

class AffineSVD(InvertibleLayer):

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
