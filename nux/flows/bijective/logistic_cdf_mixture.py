import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence
from nux.internal.layer import InvertibleLayer
import nux.util as util
from jax.scipy.special import logsumexp
import nux.networks as net
from nux.util import logistic_cdf_mixture_logit
import nux
from nux.flows.bijective.coupling_base import Elementwise
from abc import ABC, abstractmethod

__all__ = ["LogisticMixtureLogit"]

################################################################################################################

def bisection_body(f, val):
  x, current_x, current_z, lower, upper, dx, i = val

  gt = current_x > x
  lt = 1.0 - gt

  new_z = gt*0.5*(current_z + lower) + lt*0.5*(current_z + upper)
  lower = gt*lower                   + lt*current_z
  upper = gt*current_z               + lt*upper

  current_z = new_z
  current_x = f(current_z)
  dx = current_x - x

  return x, current_x, current_z, lower, upper, dx, i + 1

def bisection(f, lower, upper, x, atol=1e-8, max_iters=10000):
  # Compute f^{-1}(x) using the bisection method.  f must be monotonic.
  z = jnp.zeros_like(x)

  def cond_fun(val):
    x, current_x, current_z, lower, upper, dx, i = val

    max_iters_reached = jnp.where(i > max_iters, True, False)
    tolerance_achieved = jnp.allclose(dx, 0.0, atol=atol)

    return ~(max_iters_reached | tolerance_achieved)

  val = (x, f(z), z, lower, upper, jnp.ones_like(x)*10.0, 0.0)
  val = jax.lax.while_loop(cond_fun, partial(bisection_body, f), val)
  x, current_x, current_z, lower, upper, dx, i = val
  return current_z

################################################################################################################

class _MixtureCDFMixin(ABC):
  def __init__(self,
               n_components: int=4,
               with_affine_coupling: bool=True,
               name: str="mixture_cdf",
               **kwargs
  ):
    """ Base class for a mixture cdf with no coupling
    Args:
      n_components: Number of mixture components to use
      name        : Optional name for this module.
    """
    super().__init__(name=name, **kwargs)
    self.n_components         = n_components
    self.with_affine_coupling = with_affine_coupling
    self.extra = 2 if with_affine_coupling else 0

  def split_theta(self, theta):
    if self.with_affine_coupling:
      weight_logits, means, log_scales, log_s, t = jnp.split(theta, jnp.array([self.n_components,
                                                                               2*self.n_components,
                                                                               3*self.n_components,
                                                                               3*self.n_components + 1]), axis=-1)
      log_s = util.constrain_log_scale(log_s)
      t = t[...,0]
      log_s = log_s[...,0]
      return weight_logits, means, log_scales, log_s, t
    else:
      weight_logits, means, log_scales = jnp.split(theta, jnp.array([self.n_components,
                                                                     2*self.n_components]), axis=-1)
      return weight_logits, means, log_scales

  def safe_init(self, x, weight_logits, means, log_scales, log_s=None, t=None, conditioned_params=False):
    """ We want to initialize this to be close to the identity funtion
        but also want the means to be spread out initially
    """
    wl_scale = hk.get_parameter("weight_logits_scale", shape=(), dtype=x.dtype, init=jnp.zeros)
    ls_scale = hk.get_parameter("log_scales_scale", shape=(), dtype=x.dtype, init=jnp.zeros)

    weight_logits *= wl_scale
    log_scales *= ls_scale

    if self.with_affine_coupling:
      batch_dim = x.ndim - len(self.batch_shape)
      name_prefix = "transform_" if conditioned_params else ""

      # Initialize log_s to divide off the stddev
      def log_s_shift_init(shape, dtype):
        if x.ndim == len(shape):
          return jnp.zeros(shape, dtype)

        z = self.f(weight_logits, means, log_scales, x)
        axes = tuple(jnp.arange(len(z.shape) - len(shape)))
        return jnp.log(jnp.std(z, axis=axes) + 1e-5)

      log_s_shape = log_s.shape[batch_dim:] if conditioned_params else log_s.shape
      log_s_shift = hk.get_parameter(f"{name_prefix}log_s_shift", shape=log_s_shape, dtype=x.dtype, init=log_s_shift_init)
      log_s_scale = hk.get_parameter(f"{name_prefix}log_s_scale", shape=log_s_shape, dtype=x.dtype, init=jnp.zeros)

      # Constrain between -1 and 1 so that things don't blow up
      log_s_shift = -jnp.maximum(-1.0, -log_s_shift)
      log_s_shift = jnp.maximum(-1.0, log_s_shift)
      log_s = log_s*log_s_scale + log_s_shift

      # Initialize t to subtract off the mean
      def t_shift_init(shape, dtype):
        if x.ndim == len(shape):
          return jnp.zeros(shape, dtype)

        z = self.f(weight_logits, means, log_scales, x)
        axes = tuple(jnp.arange(len(z.shape) - len(shape)))
        return jnp.mean(z, axis=axes)

      name_prefix = "transform_" if conditioned_params else ""
      t_shape = t.shape[batch_dim:] if conditioned_params else t.shape
      t_shift = hk.get_parameter(f"{name_prefix}t_shift", shape=t_shape, dtype=x.dtype, init=t_shift_init)
      t_scale = hk.get_parameter(f"{name_prefix}t_scale", shape=t_shape, dtype=x.dtype, init=jnp.zeros)
      t = t*t_scale + t_shift

      return weight_logits, means, log_scales, log_s, t

    return weight_logits, means, log_scales

  def mixture_forward(self, x, weight_logits, means, log_scales, log_s=None, t=None):
    # Assume that this function is auto-batched
    z, elementwise_log_det = self.f_and_elementwise_log_det(weight_logits, means, log_scales, x)

    if self.with_affine_coupling:
      z = (z - t)*jnp.exp(-log_s)
      elementwise_log_det += -log_s

    return z, elementwise_log_det

  def mixture_inverse(self, z, weight_logits, means, log_scales, log_s=None, t=None):
    # Assume that this function is auto-batched
    if self.with_affine_coupling:
      x = z*jnp.exp(log_s) + t
      elementwise_log_det = -log_s
    else:
      x = z
      elementwise_log_det = 0.0

    # If we're outside of this range, then there's a bigger problem in the rest of the network.
    lower = jnp.zeros_like(x) - 1000
    upper = jnp.zeros_like(x) + 1000

    filled_f = partial(self.f, weight_logits, means, log_scales)
    x = bisection(filled_f, lower, upper, x)
    elementwise_log_det += self.elementwise_log_det(weight_logits, means, log_scales, x)

    return x, elementwise_log_det

  @abstractmethod
  def f(self, weight_logits, means, log_scales, x):
    pass

  @abstractmethod
  def elementwise_log_det(self, weight_logits, means, log_scales, x):
    pass

  def f_and_elementwise_log_det(self, weight_logits, means, log_scales, x):
    return self.f(weight_logits, means, log_scales, x), self.elementwise_log_det(weight_logits, means, log_scales, x)

################################################################################################################

class MixtureCDF(_MixtureCDFMixin, Elementwise):

  def __init__(self,
               n_components: int=4,
               with_affine_coupling: bool=True,
               create_network: Callable=None,
               network_kwargs: Optional=None,
               use_condition: bool=False,
               condition_method: str="nin",
               coupling: bool=True,
               split_kind="channel",
               masked: bool=False,
               apply_to_both_halves: Optional[bool]=True,
               name: str="coupling_mixture_cdf",
               **kwargs
  ):
    """ Base class for a mixture cdf with coupling
    Args:
      n_components  : Number of mixture components to use
      create_network: Function to create the conditioner network.  Should accept a tuple
                      specifying the output shape.  See coupling_base.py
      use_condition : Should we use inputs["condition"] to form t([xb, condition]), s([xb, condition])?
      network_kwargs: Dictionary with settings for the default network (see get_default_network in util.py)
      name          : Optional name for this module.
    """
    super().__init__(with_affine_coupling=with_affine_coupling,
                     n_components=n_components,
                     create_network=create_network,
                     axis=-1,
                     coupling=coupling,
                     split_kind=split_kind,
                     use_condition=use_condition,
                     condition_method=condition_method,
                     apply_to_both_halves=apply_to_both_halves,
                     masked=masked,
                     name=name,
                     network_kwargs=network_kwargs,
                     **kwargs)

  def get_out_shape(self, x):
    x_shape = x.shape[len(self.batch_shape):]
    out_dim = x_shape[-1]*(3*self.n_components + self.extra)
    return x_shape[:-1] + (out_dim,)

  def transform(self, x, params=None, sample=False, rng=None, **kwargs):
    conditioned_params = params is not None
    x_shape = x.shape[len(self.batch_shape):]
    if params is None:
      theta = hk.get_parameter("theta", shape=x_shape + (3*self.n_components + self.extra,), dtype=x.dtype, init=hk.initializers.RandomNormal(1.0))
    else:
      theta = params.reshape(x.shape + (3*self.n_components + self.extra,))

    # Split the parameters
    if self.with_affine_coupling:
      if params is None:
        in_axes = (0, None, None, None, None, None)
        out_axes = (None, None, None, None, None)
      else:
        in_axes = (0, 0, 0, 0, 0, 0)
        out_axes = None#(0, 0, 0, 0, 0)
    else:
      if params is None:
        in_axes = (0, None, None, None)
        out_axes = (None, None, None)
      else:
        in_axes = (0, 0, 0, 0)
        out_axes = None#(0, 0, 0)

    params = self.split_theta(theta)
    init_fun = self.auto_batch(partial(self.safe_init, conditioned_params=conditioned_params), in_axes=in_axes, out_axes=out_axes, expected_depth=1)
    params = init_fun(x, *params)

    # Run the transform
    if sample == False:
      z, elementwise_log_det = self.auto_batch(self.mixture_forward, in_axes=in_axes, expected_depth=1)(x, *params)
    else:
      z, elementwise_log_det = self.auto_batch(self.mixture_inverse, in_axes=in_axes, expected_depth=1)(x, *params)

    return z, elementwise_log_det

################################################################################################################

class _LogisticMixtureLogitMixin():

  def __init__(self,
               n_components: int=4,
               restrict_scales: bool=True,
               safe_diag: bool=True,
               name: str="logistic_mixture_cdf_logit",
               **kwargs
  ):
    """ Mix in class for logistic mixture cdf followed by logit models.
        This works pretty well in practice.  See nux/networks/nonlinearities.py
    Args:
      n_components   : Number of mixture components to use
      restrict_scales: Whether or not to bound the scales.  If log_scales is
                       unbounded, we can get model more complex distributions
                       at the risk of numerical instability.
      name           : Optional name for this module.
    """
    super().__init__(n_components=n_components, name=name, **kwargs)
    self.restrict_scales = restrict_scales
    self.safe_diag = safe_diag

  def f(self, weight_logits, means, log_scales, x):
    if self.restrict_scales:
      if self.safe_diag == False:
        log_scales = jnp.maximum(-7.0, log_scales)
      else:
        scales = util.proximal_relu(log_scales) + 1e-5
        log_scales = jnp.log(scales)

    return logistic_cdf_mixture_logit(weight_logits, means, log_scales, x)

  def elementwise_log_det(self, weight_logits, means, log_scales, x):
    return self.f_and_elementwise_log_det(weight_logits, means, log_scales, x)[1]

  def f_and_elementwise_log_det(self, weight_logits, means, log_scales, x):
    if self.restrict_scales:
      if self.safe_diag == False:
        log_scales = jnp.maximum(-7.0, log_scales)
      else:
        scales = util.proximal_relu(log_scales) + 1e-5
        log_scales = jnp.log(scales)

    primals = weight_logits, means, log_scales, x
    tangents = jax.tree_map(jnp.zeros_like, primals[:-1]) + (jnp.ones_like(x),)
    z, dzdx = jax.jvp(logistic_cdf_mixture_logit, primals, tangents)

    elementwise_log_det = jnp.log(dzdx)
    return z, elementwise_log_det

################################################################################################################

class LogisticMixtureLogit(_LogisticMixtureLogitMixin, MixtureCDF):
  pass
