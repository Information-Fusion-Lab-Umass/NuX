import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from functools import partial
import nux.util as util
from typing import Optional, Mapping, Callable, Sequence
from nux.flows.base import *
import haiku as hk
from haiku._src.typing import PRNGKey
from jax.scipy.special import gammaln, logsumexp
import nux
import nux.networks as net
from haiku._src.base import current_bundle_name

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

class RectangularMVP(Layer):

  def __init__(self,
               output_dim: int,
               create_network: Optional[Callable]=None,
               reverse_params: bool=True,
               network_kwargs: Optional=None,
               weight_norm: bool=True,
               name: str="rectangular_dense",
               **kwargs):
    self.output_dim     = output_dim
    self.create_network = create_network
    self.reverse_params = reverse_params
    self.network_kwargs = network_kwargs
    self.weight_norm    = weight_norm
    super().__init__(name=name, **kwargs)

  @property
  def orth_noise_name(self):
    return current_bundle_name()

  def get_network(self, out_shape):

    out_shape = out_shape[:-1] + (2*out_shape[-1],)

    # The user can specify a custom network
    if self.create_network is not None:
      return self.create_network(out_shape)

    return util.get_default_network(out_shape, self.network_kwargs)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           no_noise: Optional[bool]=False,
           **kwargs) -> Mapping[str, jnp.ndarray]:

    dtype = inputs["x"].dtype
    input_shape = self.unbatched_input_shapes["x"]
    input_dim = util.list_prod(input_shape)
    image_in    = len(input_shape) == 3

    if(input_dim > self.output_dim):
      kind      = "tall"
      small_dim = self.output_dim
      big_dim   = input_dim
    else:
      kind      = "wide"
      small_dim = input_dim
      big_dim   = self.output_dim

    #######################

    # Initialize the network parameters with weight normalization
    init_fun = hk.initializers.RandomNormal(stddev=0.05)
    if self.reverse_params:
      B = hk.get_parameter("B", shape=(small_dim, big_dim), dtype=dtype, init=init_fun)
      B *= jax.lax.rsqrt(jnp.sum(B**2, axis=1))[:,None]

    else:
      A = hk.get_parameter("A", shape=(big_dim, small_dim), dtype=dtype, init=init_fun)
      # A *= jax.lax.rsqrt(jnp.sum(A**2, axis=1))[:,None]

    # Apply weight norm.
    if self.weight_norm:

      def g_init(shape, dtype):
        x = inputs["x"]
        if image_in:
          x = x.reshape(self.batch_shape + (-1,))
        t = jnp.dot(x, B.T) if self.reverse_params else jnp.dot(x, A.T)
        g = 1/(jnp.std(t, axis=0) + 1e-5)
        return g

      if self.reverse_params:
        g = hk.get_parameter("g", shape=(small_dim,), dtype=dtype, init=g_init)
        B *= g[:,None]
      # else:
      #   g = hk.get_parameter("g", shape=(big_dim,), dtype=dtype, init=g_init)
      #   A *= g[:,None]

    # Compute the riemannian metric matrix
    if self.reverse_params:
      BBT = B@B.T
      BBT_inv = jnp.linalg.inv(BBT)
    else:
      ATA_inv = jnp.linalg.inv(A.T@A)

    # Create the haiku network
    network = self.get_network(self.unbatched_input_shapes["x"])

    #######################

    # Figure out which direction we should go
    if sample == False:
      if kind == "tall":
        big_to_small = True
      else:
        big_to_small = False
    else:
      if kind == "tall":
        big_to_small = False
      else:
        big_to_small = True

    #######################

    # Compute the next value
    if big_to_small:
      t = inputs["x"]

      # If we're going from image -> vector, we need to flatten the image
      if image_in:
        t = t.reshape(self.batch_shape + (-1,))

      # Compute the pseudo inverse
      if self.reverse_params:
        s = jnp.dot(t, B.T)
        t_proj = jnp.dot(s, BBT_inv.T)
        t_proj = jnp.dot(t_proj, B)

      else:
        s = jnp.dot(t, A)
        s = jnp.dot(s, ATA_inv.T)
        t_proj = jnp.dot(s, A.T)

      # Compute the perpendicular component of t for the log contribution
      gamma_perp = t - t_proj

      # Compute the parameters of p(gamma)
      if image_in:
        network_in = t_proj.reshape(self.batch_shape + input_shape)
      else:
        network_in = s
      network_out = self.auto_batch(network, expected_depth=1)(network_in)
      mu, log_diag_cov = jnp.split(network_out, 2, axis=-1)
      log_diag_cov = jnp.logaddexp(log_diag_cov, -10)

      # If we're going from image -> vector, we need to flatten the image
      if image_in:
        mu = mu.reshape(self.batch_shape + (-1,))
        log_diag_cov = log_diag_cov.reshape(self.batch_shape + (-1,))

      # Compute the log contribution
      batched_logZ = self.auto_batch(logZ, in_axes=(0, None, 0))
      if self.reverse_params:
        log_contribution = batched_logZ(mu - gamma_perp, B.T, log_diag_cov) + jnp.linalg.slogdet(BBT)[1]
      else:
        log_contribution = batched_logZ(mu - gamma_perp, A, log_diag_cov)

      outputs = {"x": s, "log_det": log_contribution}

    else:
      s = inputs["x"]

      # Compute the mean of t
      if self.reverse_params:
        t_mean = jnp.dot(s, BBT_inv.T)
        t_mean = jnp.dot(t_mean, B)
      else:
        t_mean = jnp.dot(s, A.T)

      # Compute the parameters of p(gamma)
      if image_in:
        network_in = t_mean.reshape(self.batch_shape + input_shape)
      else:
        network_in = s
      network_out = self.auto_batch(network, expected_depth=1)(network_in)
      mu, log_diag_cov = jnp.split(network_out, 2, axis=-1)
      log_diag_cov = jnp.logaddexp(log_diag_cov, -10)

      if image_in:
        mu = mu.reshape(self.batch_shape + (-1,))
        log_diag_cov = log_diag_cov.reshape(self.batch_shape + (-1,))

      # Sample gamma_perp
      gamma = mu
      if no_noise == False:
        gamma += random.normal(rng, mu.shape)*jnp.exp(0.5*log_diag_cov)

      if self.reverse_params:
        gamma_proj = jnp.dot(gamma, B.T)
        gamma_proj = jnp.dot(gamma_proj, BBT_inv.T)
        gamma_proj = jnp.dot(gamma_proj, B)
        gamma_perp = gamma - gamma_proj
      else:
        gamma_proj = jnp.dot(gamma, A)
        gamma_proj = jnp.dot(gamma_proj, ATA_inv.T)
        gamma_proj = jnp.dot(gamma_proj, A.T)
        gamma_perp = gamma - gamma_proj

      # Add the orthogonal features
      t = t_mean + gamma_perp

      # Compute the log contribution
      batched_logZ = self.auto_batch(logZ, in_axes=(0, None, 0))
      if self.reverse_params:
        log_contribution = -batched_logZ(mu - gamma_perp, B.T, log_diag_cov) - jnp.linalg.slogdet(BBT)[1]
      else:
        log_contribution = -batched_logZ(mu - gamma_perp, A, log_diag_cov)

      if image_in:
        t = t.reshape(self.batch_shape + input_shape)
      outputs = {"x": t, "log_det": log_contribution}

    return outputs
