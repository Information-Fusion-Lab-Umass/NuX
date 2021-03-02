import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from functools import partial
import nux.util as util
from typing import Optional, Mapping, Callable, Sequence
from nux.internal.layer import InvertibleLayer
import haiku as hk
from haiku._src.typing import PRNGKey
from jax.scipy.special import gammaln, logsumexp
import nux
import nux.networks as net
import nux.util.weight_initializers as init
import nux.vae as vae

__all__ = ["RectangularMVP"]

@jit
def logZ(x, A, log_diag_cov):
  diag_cov = jnp.exp(log_diag_cov)

  # N(x|0,Sigma)
  log_px = -0.5*jnp.sum(x**2/diag_cov)
  log_px -= 0.5*jnp.sum(log_diag_cov)
  log_px -= 0.5*x.shape[-1]*jnp.log(2*jnp.pi)

  # N(h|0,J)|J|
  ATdiag_cov = A.T/diag_cov
  h = ATdiag_cov@x
  J = ATdiag_cov@A
  J_inv = jnp.linalg.inv(J)
  log_ph = -0.5*jnp.einsum("i,ij,j", h, J_inv, h)
  log_ph += 0.5*jnp.linalg.slogdet(J)[1] # Add log|J| to the log pdf!
  log_ph -= 0.5*h.shape[-1]*jnp.log(2*jnp.pi)

  return log_px - log_ph

################################################################################################################

class RectangularMVP(InvertibleLayer):

  def __init__(self,
               output_dim: int,
               generative_only: bool=False,
               create_network: Optional[Callable]=None,
               reverse_params: bool=False,
               network_kwargs: Optional=None,
               weight_norm: bool=False,
               spectral_norm: bool=False,
               name: str="rectangular_mvp",
               **kwargs):
    if generative_only == False:
      self._output_dim = output_dim
    else:
      self._input_dim = output_dim

    self.generative_only = generative_only
    self.reverse_params  = reverse_params
    self.weight_norm     = weight_norm
    self.spectral_norm   = spectral_norm
    self.create_network  = create_network
    self.network_kwargs  = network_kwargs
    super().__init__(name=name, **kwargs)

  @property
  def input_shape(self):
    return self.unbatched_input_shapes["x"]

  @property
  def output_shape(self):
    return self.unbatched_output_shapes["x"]

  @property
  def image_in(self):
    if self.generative_only:
      return len(self.output_shape) == 3
    return len(self.input_shape) == 3

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
  def kind(self):
    return "tall" if self.input_dim > self.output_dim else "wide"

  @property
  def small_dim(self):
    return self.output_dim if self.input_dim > self.output_dim else self.input_dim

  @property
  def big_dim(self):
    return self.input_dim if self.input_dim > self.output_dim else self.output_dim

  def pinv(self, t):
    if self.reverse_params:
      s = jnp.dot(t, self.B.T)
    else:
      s = jnp.dot(t, self.A)
      s = jnp.dot(s, self.ATA_inv.T)
    return s

  def project(self, t=None, s=None):
    if s is not None:
      if self.reverse_params:
        t_proj = jnp.dot(s, self.BBT_inv.T)
        t_proj = jnp.dot(t_proj, self.B)
      else:
        t_proj = jnp.dot(s, self.A.T)

      return t_proj
    else:
      s = self.pinv(t)
      return self.project(s=s)

    assert 0, "Must pass in either s or t"

  def orthogonal_distribution(self, s, t_proj, rng, no_noise):
    network_in = t_proj.reshape(self.batch_shape + self.input_shape) if self.image_in else s
    outputs = self.p_gamma_given_s({"x": network_in}, rng=rng, no_noise=no_noise)
    gamma, mu, log_diag_cov = outputs["x"], outputs["mu"], outputs["log_diag_cov"]

    # If we're going from image -> vector, we need to flatten the image
    if self.image_in:
      gamma        = gamma.reshape(self.batch_shape + (-1,))
      mu           = mu.reshape(self.batch_shape + (-1,))
      log_diag_cov = log_diag_cov.reshape(self.batch_shape + (-1,))

    return gamma, mu, log_diag_cov

  def likelihood_contribution(self, mu, gamma_perp, log_diag_cov, sample, big_to_small):
    batched_logZ = self.auto_batch(logZ, in_axes=(0, None, 0))
    if sample == True and big_to_small == False:
      if self.reverse_params:
        likelihood_contribution = -0.5*jnp.linalg.slogdet(self.BBT)[1]
      else:
        likelihood_contribution = 0.5*jnp.linalg.slogdet(self.ATA)[1]
    else:
      if self.reverse_params:
        likelihood_contribution = batched_logZ(mu - gamma_perp, self.B.T, log_diag_cov) + jnp.linalg.slogdet(self.BBT)[1]
      else:
        likelihood_contribution = batched_logZ(mu - gamma_perp, self.A, log_diag_cov)

    return likelihood_contribution

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           no_noise: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    # p(gamma|s) = N(gamma|mu(s), Sigma(s))
    if self.image_in:
      out_shape = self.input_shape[:-1] + (2*self.input_shape[-1],)
    else:
      out_shape = (2*self.big_dim,)
    self.p_gamma_given_s = vae.ParametrizedGaussian(out_shape=out_shape,
                                                    create_network=self.create_network,
                                                    network_kwargs=self.network_kwargs)

    #######################
    assert self.big_dim - self.small_dim > 0

    # Initialize the tall or wide matrix.  We might want to choose to parametrize a tall
    # matrix as the pseudo-inverse of a wide matrix or vice-versa.  B is wide and A is tall.
    init_fun = hk.initializers.RandomNormal(stddev=0.05)
    dtype = inputs["x"].dtype
    if self.reverse_params:
      x = inputs["x"].reshape(self.batch_shape + (-1,))

      if self.spectral_norm:
        self.B = init.weight_with_spectral_norm(x,
                                                self.small_dim,
                                                use_bias=False,
                                                w_init=init_fun,
                                                force_in_dim=self.big_dim,
                                                is_training=kwargs.get("is_training", True),
                                                update_params=kwargs.get("is_training", True))
      else:
        if self.weight_norm and self.kind == "tall":
          self.B = init.weight_with_weight_norm(x, self.small_dim, use_bias=False, force_in_dim=self.big_dim)
        else:
          self.B = hk.get_parameter("B", shape=(self.small_dim, self.big_dim), dtype=dtype, init=init_fun)
        self.B = util.whiten(self.B)
    else:
      if self.spectral_norm:
        self.A = init.weight_with_spectral_norm(x,
                                                self.big_dim,
                                                use_bias=False,
                                                w_init=init_fun,
                                                force_in_dim=self.small_dim,
                                                is_training=kwargs.get("is_training", True),
                                                update_params=kwargs.get("is_training", True))
      else:
        self.A = hk.get_parameter("A", shape=(self.big_dim, self.small_dim), dtype=dtype, init=init_fun)
        self.A = util.whiten(self.A)

    # Compute the riemannian metric matrix for later use.
    if self.reverse_params:
      self.BBT     = self.B@self.B.T
      self.BBT_inv = jnp.linalg.inv(self.BBT)
    else:
      self.ATA     = self.A.T@self.A
      self.ATA_inv = jnp.linalg.inv(self.ATA)

    #######################

    # Figure out which direction we should go
    if sample == False:
      big_to_small = True if self.kind == "tall" else False
    else:
      big_to_small = False if self.kind == "tall" else True

    #######################

    # Compute the next value
    if big_to_small:
      t = inputs["x"]

      # If we're going from image -> vector, we need to flatten the image
      if self.image_in:
        t = t.reshape(self.batch_shape + (-1,))

      # Compute the pseudo inverse and projection
      # s <- self.A^+t
      s = self.pinv(t)
      t_proj = self.project(s=s)

      # Compute the perpendicular component of t for the log contribution
      # gamma_perp <- t - AA^+t
      gamma_perp = t - t_proj

      # Find mu(s), Sigma(s).  If we have an image as input, pass in the projected input image
      # mu, Sigma <- NN(s, theta)
      _, mu, log_diag_cov = self.orthogonal_distribution(s, t_proj, rng, no_noise=True)

      # Compute the log contribution
      # L <- logZ(mu - gamma_perp|self.A, Sigma)
      likelihood_contribution = self.likelihood_contribution(mu, gamma_perp, log_diag_cov, sample=sample, big_to_small=big_to_small)

      outputs = {"x": s, "log_det": likelihood_contribution}

    else:
      s = inputs["x"]

      # Compute the mean of t.  Primarily used if we have an image as input
      t_mean = self.project(s=s)

      # Find mu(s), Sigma(s).  If we have an image as input, pass in the projected input image
      # mu, Sigma <- NN(s, theta)
      # gamma ~ N(mu, Sigma)
      gamma, mu, log_diag_cov = self.orthogonal_distribution(s, t_mean, rng, no_noise=no_noise)

      # Compute the orthogonal component of the noise
      # gamma_perp <- gamma - AA^+ gamma
      gamma_proj = self.project(t=gamma)
      gamma_perp = gamma - gamma_proj

      # Add the orthogonal features
      # t <- As + gamma_perp
      t = t_mean + gamma_perp

      # Compute the log contribution
      # L <- logZ(mu - gamma_perp|self.A, Sigma)
      likelihood_contribution = -self.likelihood_contribution(mu, gamma_perp, log_diag_cov, sample=sample, big_to_small=big_to_small)

      # Reshape to an image if needed
      if self.image_in:
        t = t.reshape(self.batch_shape + self.input_shape)

      outputs = {"x": t, "log_det": likelihood_contribution}

    return outputs
