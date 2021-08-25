import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
from typing import Optional, Mapping, Callable, Sequence
import nux.util as util
from jax.scipy.special import logsumexp

__all__ = ["UnitGaussianPrior",
           "FixedGaussianPrior",
           "GaussianPrior",
           "TruncatedUnitGaussianPrior",
           "ParametrizedGaussianPrior",
           "AffineGaussianPriorDiagCov"]

class UnitGaussianPrior():

  def __init__(self):
    """ Unit Gaussian prior
    Args:
      name: Optional name for this module.
    """
    pass

  def get_params(self):
    return {}

  def __call__(self, x, rng_key=None, inverse=False, reconstruction=False, **kwargs):

    if inverse and reconstruction == False:
      x = random.normal(rng_key, x.shape)

    sum_axes = util.last_axes(x.shape[1:])
    log_pz = -0.5*(x**2).sum(axis=sum_axes) - 0.5*util.list_prod(x.shape[1:])*jnp.log(2*jnp.pi)
    return x, log_pz

################################################################################################################

class FixedGaussianPrior():
  def __init__(self, mean, std):
    self.mean = jnp.array(mean)
    self.std = jnp.array(std)

  def get_params(self):
    return ()

  def __call__(self, x, params=None, rng_key=None, inverse=False, reconstruction=False, **kwargs):

    self.mean = jnp.broadcast_to(self.mean, x.shape)
    self.std = jnp.broadcast_to(self.std, x.shape)

    if inverse and reconstruction == False:
      x = random.normal(rng_key, x.shape)
      x = self.mean + x*self.std

    sum_axes = util.last_axes(x.shape[1:])
    dx = (x - self.mean)/self.std
    log_pz = -0.5*(dx**2).sum(axis=sum_axes)
    log_pz -= jnp.log(self.std).sum()
    log_pz -= 0.5*util.list_prod(x.shape[1:])*jnp.log(2*jnp.pi)

    return x, log_pz

################################################################################################################

class GaussianPrior():

  def __init__(self):
    """ Unit Gaussian prior
    Args:
      name: Optional name for this module.
    """
    pass

  def get_params(self):
    return dict(mu=self.mu, s=self.s)

  def __call__(self, x, params=None, rng_key=None, inverse=False, reconstruction=False, **kwargs):

    if params is None:
      # self.mu, self.s = jnp.zeros((2, *x.shape[1:]))
      self.mu, self.s = random.normal(rng_key, (2, *x.shape[1:]))
    else:
      self.mu, self.s = params["mu"], params["s"]

    s = util.square_plus(self.s, gamma=1.0) + 1e-4

    if inverse and reconstruction == False:
      x = random.normal(rng_key, x.shape)
      x = self.mu + x*s

    sum_axes = util.last_axes(x.shape[1:])
    dx = (x - self.mu)/s
    log_pz = -0.5*(dx**2).sum(axis=sum_axes)
    log_pz -= jnp.log(s).sum()
    log_pz -= 0.5*util.list_prod(x.shape[1:])*jnp.log(2*jnp.pi)

    return x, log_pz

################################################################################################################

class TruncatedUnitGaussianPrior():

  def __init__(self):
    """ Unit Gaussian prior but between (-1, 1)
    Args:
      name: Optional name for this module.
    """
    cdf = lambda x: 0.5*(1 + jax.scipy.special.erf(x*jax.lax.rsqrt(2.0)))
    self.logZ = jnp.log(cdf(1.0) - cdf(-1.0))

  def get_params(self):
    return {}

  def __call__(self, x, rng_key=None, inverse=False, reconstruction=False, **kwargs):

    if inverse and reconstruction == False:
      x = random.truncated_normal(rng_key, -1, 1, x.shape)

    sum_axes = util.last_axes(x.shape[1:])
    log_pz = -0.5*(x**2).sum(axis=sum_axes) - 0.5*util.list_prod(x.shape[1:])*jnp.log(2*jnp.pi)

    log_pz -= util.list_prod(x.shape[1:])*self.logZ
    return x, log_pz

################################################################################################################

class ParametrizedGaussianPrior():

  def __init__(self, network):
    """ Unit Gaussian prior
    Args:
      name: Optional name for this module.
    """
    self.network = network

  def get_params(self):
    return {"network": self.network.get_params()}

  def __call__(self, x, params=None, aux=None, rng_key=None, is_training=True, inverse=False, reconstruction=False, **kwargs):
    if params is None:
      self.network_params = None
    else:
      self.network_params = params["network"]

    # Pass the condition through the parametrized gaussian
    theta = self.network(aux, params=self.network_params, rng_key=k1, is_training=is_training)
    mu, diag_cov = jnp.split(theta, 2, axis=-1)
    diag_cov = util.square_plus(diag_cov)
    log_diag_cov = jnp.log(diag_cov)

    if inverse and reconstruction == False:
      x = mu + random.normal(k2, x.shape)*jnp.sqrt(diag_cov)

    def logpdf(x, mu, log_diag_cov):
      dx = x - mu
      log_px = -0.5*jnp.sum(dx.ravel()**2*jnp.exp(-log_diag_cov.ravel()), axis=-1)
      return log_px - 0.5*jnp.sum(log_diag_cov.ravel()) - 0.5*x.size*jnp.log(2*jnp.pi)

    log_pz = jax.vmap(logpdf)(x, mu, log_diag_cov)

    return x, log_pz

################################################################################################################

class AffineGaussianPriorDiagCov():

  def __init__(self,
               output_dim: int,
               generative_only: bool=False,
               name: str="affine_gaussian_prior"
  ):
    assert 0
    """ Analytic solution to int N(z|0,I)N(x|Az,Sigma)dz.
        https://arxiv.org/pdf/2006.13070v1.pdf
    Args:
      name: Optional name for this module.
    """
    super().__init__(name=name)
    if generative_only == False:
      self._output_dim = output_dim
    else:
      self._input_dim = output_dim
    self.generative_only = generative_only

  @property
  def input_shape(self):
    return self.unbatched_input_shapes["x"]

  @property
  def output_shape(self):
    return self.unbatched_output_shapes["x"]

  @property
  def input_dim(self):
    if hasattr(self, "_input_dim"):
      return self._input_dim
    return util.list_prod(self.input_shape)

  @property
  def output_dim(self):
    if hasattr(self, "_output_dim"):
      return self._output_dim
    return util.list_prod(self.output_shape)

  @property
  def z_dim(self):
    return self.output_dim if self.input_dim > self.output_dim else self.input_dim

  @property
  def x_dim(self):
    return self.input_dim if self.input_dim > self.output_dim else self.output_dim

  def call(self,
           inputs,
           rng,
           sample,
           reconstruction,
           manifold_sample,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    assert len(self.unbatched_input_shapes["x"]) == 1, "Only works with 1d inputs"
    assert self.z_dim < self.x_dim

    dtype = inputs["x"].dtype
    init_fun = hk.initializers.RandomNormal(0.01)
    A = hk.get_parameter("A", shape=(self.x_dim, self.z_dim), dtype=dtype, init=init_fun)
    b = hk.get_parameter("b", shape=(self.x_dim,), dtype=dtype, init=init_fun)
    log_diag_cov = hk.get_parameter("log_diag_cov", shape=(self.x_dim,), dtype=dtype, init=init_fun)
    diag_cov = jnp.exp(log_diag_cov)

    # Go from x -> z or z -> x
    if sample == False:
      x = inputs["x"]
      x -= b

      # Compute the posterior natural parameters
      J = jnp.eye(self.z_dim) + (A.T/diag_cov)@A
      J_inv = jnp.linalg.inv(J)
      sigma_inv_x = x/diag_cov
      h = jnp.dot(sigma_inv_x, A)

      # Compute the posterior parameters
      Sigma_z = J_inv
      mu_z = jnp.dot(h, Sigma_z)

      # Sample z
      Sigma_z_chol = jnp.linalg.cholesky(Sigma_z)
      noise = random.normal(rng, mu_z.shape)
      z = mu_z + jnp.dot(noise, Sigma_z_chol.T)

      # Compute the log likelihood contribution
      J_inv_h = jnp.dot(h, J_inv.T)

      llc = 0.5*jnp.sum(h*J_inv_h, axis=-1)
      llc -= 0.5*jnp.linalg.slogdet(J)[1]
      llc -= 0.5*jnp.sum(x*sigma_inv_x, axis=-1)
      llc -= 0.5*log_diag_cov.sum()
      llc -= 0.5*self.x_dim*jnp.log(2*jnp.pi)

      outputs = {"x": z, "log_pz": llc}

    else:
      k1, k2 = random.split(rng, 2)
      z = inputs["x"]

      if reconstruction == False:
        z = random.normal(k1, z.shape)

      # Sample x
      mu_x = jnp.dot(z, A.T) + b
      if manifold_sample == False:
        noise = random.normal(k2, mu_x.shape)
      else:
        noise = jnp.zeros_like(mu_x)

      x = mu_x + jnp.sqrt(diag_cov)*noise

      # If we're doing a reconstruction, we need to compute log p(x|z)
      llc = -0.5*jnp.sum(noise**2, axis=-1)
      llc -= 0.5*jnp.sum(log_diag_cov)
      llc -= 0.5*self.x_dim*jnp.log(2*jnp.pi)

      outputs = {"x": x, "log_pz": llc}

    return outputs

################################################################################################################
