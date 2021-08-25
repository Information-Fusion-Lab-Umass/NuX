import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from functools import partial
import nux.util as util
from typing import Optional, Mapping, Callable, Sequence
import nux

__all__ = ["TallMVP"]

def mvp(A, x):
  return jnp.einsum("ij,...j->...i", A, x)

def logZ(x, A, diag_cov):

  # N(x|0,Sigma)
  log_px = -0.5*jnp.sum(x**2/diag_cov)
  log_px -= 0.5*jnp.sum(jnp.log(diag_cov))
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

class TallMVP():
  """ http://proceedings.mlr.press/v130/cunningham21a/cunningham21a.pdf """

  def __init__(self,
               output_dim,
               create_network):
    self.s_dim = output_dim
    self.create_network = create_network

  def get_params(self):
    return dict(network=self.network.get_params(),
                A=self.A,
                scale=self.scale_params,
                orthogonal_noise=self.gamma_perp)

  def __call__(self, x, params=None, inverse=False, rng_key=None, is_training=True, save_orthogonal_noise=False, **kwargs):

    if inverse == False:
      t = x # Follow convention from paper
      self.t_dim = x.shape[-1]
    else:
      s = x
      assert s.shape[-1] == self.s_dim

    self.network = self.create_network(2*self.t_dim)

    self.gamma_perp = None
    if params is None:
      k1, k2 = random.split(rng_key, 2)
      self.scale_params = random.normal(k1, ())*0.01
      self.network_params = None
      self.A = random.normal(k2, shape=(self.t_dim, self.s_dim))
    else:
      self.scale_params = params["scale"]
      self.network_params = params["network"]
      self.A = params["A"]
      gamma_perp = params.get("orthogonal_noise", None)

    self.ATA = self.A.T@self.A
    self.ATA_inv = jnp.linalg.inv(self.ATA)

    if inverse == False:
      # Pseudo-inverse of t
      s = mvp(self.ATA_inv, mvp(self.A.T, t))

      # Projection of t
      t_proj = mvp(self.A, s)

      # Orthogonal component
      gamma_perp = t - t_proj
      if save_orthogonal_noise:
        self.gamma_perp = gamma_perp

      # Create the features.  Pass in t instead of s to make the code simpler.
      theta = self.network(t_proj, aux=None, params=self.network_params, rng_key=rng_key, is_training=is_training)
      theta *= self.scale_params # Initialize to identity
      mu, diag_cov = jnp.split(theta, 2, axis=-1)
      diag_cov = util.square_plus(diag_cov, gamma=1.0) + 1e-4

    else:
      k1, k2 = random.split(rng_key, 2)

      # Projection
      t_proj = jnp.einsum("ij,...j->...i", self.A, s)

      # Create the features.  Pass in t instead of s to make the code simpler.
      theta = self.network(t_proj, aux=None, params=self.network_params, rng_key=k1, is_training=is_training)
      # theta *= self.scale_params # Initialize to identity
      mu, diag_cov = jnp.split(theta, 2, axis=-1)
      diag_cov = util.square_plus(diag_cov, gamma=1.0) + 1e-4

      if gamma_perp is None:
        # Sample orthogonal noise
        gaussian_params = dict(mu=mu, s=jnp.sqrt(diag_cov))
        gamma, _ = nux.GaussianPrior()(jnp.zeros_like(mu), params=gaussian_params, rng_key=k2, inverse=True, is_training=is_training)

        # Orthogonalize the noise
        gamma_perp = gamma - mvp(self.A, mvp(self.ATA_inv, mvp(self.A.T, gamma)))

      # Compute t
      t = t_proj + gamma_perp

    # Log likelihood contribution
    llc = jax.vmap(logZ, in_axes=(0, None, 0))(mu - gamma_perp, self.A, diag_cov)

    z = s if inverse == False else t
    return z, llc

################################################################################################################

if __name__ == "__main__":
  from debug import *
  import matplotlib.pyplot as plt

  H, W, C = 8, 8, 8
  rng_key = random.PRNGKey(0)
  # x = random.normal(rng_key, (20, H, W, C))
  x = random.normal(rng_key, (10000, 2))
  # x = jnp.linspace(-2, 2, 100)[None]

  create_network = lambda out_dim: nux.CouplingResNet1D(out_dim,
                                                        working_dim=8,
                                                        hidden_dim=16,
                                                        nonlinearity=util.square_swish,
                                                        dropout_prob=0.0,
                                                        n_layers=3)

  flow = nux.Sequential([nux.TallMVP(output_dim=1,
                                     create_network=create_network),
                         nux.UnitGaussianPrior()])

  z, log_px1 = flow(x, rng_key=rng_key)
  params = flow.get_params()

  gamma_perp = flow.layers[0].gamma_perp
  reconstr, log_px2 = flow(z, params=params, rng_key=rng_key, inverse=True, reconstruction=True, gamma_perp=gamma_perp)
  # import pdb; pdb.set_trace()

  # Evaluate the log likelihood of a lot of samples
  samples, log_px = flow(jnp.zeros((10000, 1)), params=params, rng_key=rng_key, inverse=True)

  # import pdb; pdb.set_trace()

  # Get the exact log likelihood
  import scipy.stats
  from scipy.stats import gaussian_kde
  kernel = gaussian_kde(samples.T)

  px = kernel(samples.T)
  true_log_px = jnp.log(px)
  plt.hist(log_px - true_log_px, bins=50);plt.show()

  mask = jnp.linalg.norm(samples, axis=-1) < 0.5

  import pdb; pdb.set_trace()

  # fig, (ax1, ax2) = plt.subplots(1, 2);ax1.plot(x.ravel(), z.ravel());ax2.scatter(x.ravel(), reconstr.ravel(), c=flow.gamma_perp.ravel());plt.show()