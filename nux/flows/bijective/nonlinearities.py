import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping
from nux.internal.layer import InvertibleLayer
import nux.util as util

__all__ = ["LeakyReLU",
           "LeakyReLUInv",
           "SneakyReLU",
           "Sigmoid",
           "Logit",
           "SoftplusInverse"]

class LeakyReLUInv(InvertibleLayer):

  def __init__(self,
               alpha: float=0.01,
               name: str="leaky_relu"
  ):
    """ Inverse of leaky relu
    Args:
      alpha: Slope for negative values
      name : Optional name for this module.
    """
    super().__init__(name=name)
    self.alpha = alpha

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    x_shape = self.get_unbatched_shapes(sample)["x"]
    sum_axes = util.last_axes(x_shape)

    if sample == True:
      z = jnp.where(x > 0, x, self.alpha*x)
    else:
      z = jnp.where(x > 0, x, x/self.alpha)

    log_dx_dz = jnp.where(x > 0, 0, jnp.log(self.alpha))
    log_det = log_dx_dz.sum(axis=sum_axes)*jnp.ones(self.batch_shape)

    outputs = {"x": z, "log_det": -log_det}
    return outputs

class LeakyReLU(InvertibleLayer):

  def __init__(self,
               alpha: float=0.01,
               name: str="leaky_relu"
  ):
    """ Leaky relu
    Args:
      alpha: Slope for negative values
      name : Optional name for this module.
    """
    super().__init__(name=name)
    self.alpha = alpha

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    x_shape = self.get_unbatched_shapes(sample)["x"]
    sum_axes = util.last_axes(x_shape)

    if sample == False:
      z = jnp.where(x > 0, x, self.alpha*x)
    else:
      z = jnp.where(x > 0, x, x/self.alpha)

    log_dx_dz = jnp.where(x > 0, 0, jnp.log(self.alpha))
    log_det = log_dx_dz.sum(axis=sum_axes)*jnp.ones(self.batch_shape)

    outputs = {"x": z, "log_det": log_det}
    return outputs

class SneakyReLU(InvertibleLayer):

  """ Adapted from https://github.com/didriknielsen/survae_flows/blob/master/survae/transforms/bijections/elementwise_nonlinear.py
      Originally from https://invertibleworkshop.github.io/INNF_2019/accepted_papers/pdfs/INNF_2019_paper_26.pdf """

  def __init__(self,
               alpha: float=0.1,
               name: str="sneaky_relu"
  ):
    """ Smooth version of leaky relu
    Args:
      alpha: Slope for negative values
      name : Optional name for this module.
    """
    super().__init__(name=name)

    # Sneaky ReLU uses a different convention
    self.alpha = (1.0 - alpha)/(1.0 + alpha)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    outputs = {}
    x_shape = self.get_unbatched_shapes(sample)["x"]
    sum_axes = util.last_axes(x_shape)

    if sample == False:
      x = inputs["x"]
      sqrt_1px2 = jnp.sqrt(1 + x**2)
      z = (x + self.alpha*(sqrt_1px2 - 1))/(1 + self.alpha)
      outputs["x"] = z
    else:
      z = inputs["x"]
      alpha_sq = self.alpha**2
      b = (1 + self.alpha)*z + self.alpha
      x = (jnp.sqrt(alpha_sq*(1 + b**2 - alpha_sq)) - b)/(alpha_sq - 1)
      outputs["x"] = x
      sqrt_1px2 = jnp.sqrt(1 + x**2)

    log_det = jnp.log(1 + self.alpha*x/sqrt_1px2) - jnp.log(1 + self.alpha)
    log_det = log_det.sum(axis=sum_axes)*jnp.ones(self.batch_shape)

    outputs["log_det"] = log_det
    return outputs

################################################################################################################

class Sigmoid(InvertibleLayer):

  def __init__(self,
               scale: Optional[float]=None,
               name: str="sigmoid"
  ):
    """ Sigmoid function.  Transforms data to [scale/2, 1 - scale/2]
    Args:
      scale: For interval
      name : Optional name for this module.
    """
    super().__init__(name=name)
    self.scale = scale
    self.has_scale = scale is not None

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    x_shape = self.get_unbatched_shapes(sample)["x"]
    sum_axes = util.last_axes(x_shape)

    if sample == False:
      z = jax.nn.sigmoid(x)

      if self.has_scale == True:
        z -= self.scale
        z /= 1.0 - 2*self.scale

      log_det = -(jax.nn.softplus(x) + jax.nn.softplus(-x))
    else:
      if self.has_scale == True:
        x *= 1.0 - 2*self.scale
        x += self.scale

      z = jax.scipy.special.logit(x)
      log_det = -(jax.nn.softplus(z) + jax.nn.softplus(-z))

    if self.has_scale == True:
      log_det -= jnp.log(1.0 - 2*self.scale)

    log_det = log_det.sum(axis=sum_axes)*jnp.ones(self.batch_shape)

    outputs = {"x": z, "log_det": log_det}
    return outputs

class Logit(InvertibleLayer):

  def __init__(self,
               scale: Optional[float]=0.05,
               name: str="logit"
  ):
    """ Logit function.  Transforms from [scale/2, 1 - scale/2] to the reals.
        If this is used in an image pipeline, can pass the keyword arg "generate_image"
        during sampling to return the the value between 0 and 1 so that it displays
        properly on matplotlib.
    Args:
      scale: For interval
      name : Optional name for this module.
    """
    super().__init__(name=name)
    self.scale = scale
    self.has_scale = scale is not None

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           generate_image: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    outputs = {}
    x_shape = self.get_unbatched_shapes(sample)["x"]
    sum_axes = util.last_axes(x_shape)

    if sample == False:
      if self.has_scale == True:
        x *= (1.0 - 2*self.scale)
        x += self.scale
      z = jax.scipy.special.logit(x)
      log_det = (jax.nn.softplus(z) + jax.nn.softplus(-z))
    else:
      z = jax.nn.sigmoid(x)
      log_det = (jax.nn.softplus(x) + jax.nn.softplus(-x))

      # If we are generating images, we want to pass the normalized image
      # to matplotlib!
      if generate_image:
        outputs["image"] = z

      if self.has_scale == True:
        z -= self.scale
        z /= (1.0 - 2*self.scale)

    if self.has_scale == True:
      log_det += jnp.log(1.0 - 2*self.scale)

    log_det = log_det.sum(axis=sum_axes)*jnp.ones(self.batch_shape)

    outputs["x"] = z
    outputs["log_det"] = log_det
    return outputs

################################################################################################################

class SoftplusInverse(InvertibleLayer):

  """ Adapted from https://github.com/didriknielsen/survae_flows/blob/master/survae/transforms/bijections/elementwise_nonlinear.py """

  def __init__(self,
               name: str="softplus_inv"
  ):
    """ Inverse of softplus
    Args:
      name : Optional name for this module.
    """
    super().__init__(name=name)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x_shape = self.get_unbatched_shapes(sample)["x"]
    sum_axes = util.last_axes(x_shape)

    if sample == False:
      x = inputs["x"]
      x = jnp.where(x < 0.0, 1e-5, x)
      dx = jnp.log1p(-jnp.exp(-x))
      z = x + dx
      log_det = -dx.sum(axis=sum_axes)*jnp.ones(self.batch_shape)
      outputs = {"x": z, "log_det": log_det}
    else:
      x = jax.nn.softplus(inputs["x"])
      log_det = -jnp.log1p(-jnp.exp(x)).sum(axis=sum_axes)*jnp.ones(self.batch_shape)
      outputs = {"x": x, "log_det": log_det}

    return outputs
