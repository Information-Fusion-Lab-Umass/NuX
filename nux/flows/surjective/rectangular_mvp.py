import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from functools import partial
import nux.util as util
from typing import Optional, Mapping, Callable, Sequence
import nux
import numpy as np

__all__ = ["TallMVP", "WideMVP"]

def mvp(A, x):
  return jnp.einsum("ij,...j->...i", A, x)

solve = jax.vmap(jnp.linalg.solve, in_axes=(None, 0))

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
               create_network,
               condition_on_t=False,
               pca_init=False):
    self.s_dim = output_dim
    self.create_network = create_network
    self.is_tall = True
    self.condition_on_t = condition_on_t
    self.pca_init = pca_init

  def get_params(self):
    return dict(network=self.network.get_params(),
                A=self.A,
                scale=self.scale_params)

  def __call__(self, x, params=None, inverse=False, rng_key=None, is_training=True, reuse_orthogonal_noise=False, no_llc=False, **kwargs):

    if inverse == False:
      t = x # Follow convention from paper
      self.t_dim = x.shape[-1]
    else:
      s = x
      self.s_dim = x.shape[-1]

    if kwargs.get("gamma_perp", None) is None:
      self.network = self.create_network(2*self.t_dim)

    if params is None:
      k1, k2 = random.split(rng_key, 2)
      self.scale_params = random.normal(k1, ())*0.01
      self.network_params = None

      if self.pca_init and self.is_tall:
        assert x.shape[0] > self.s_dim
        _, s, VT = jnp.linalg.svd(x)
        self.A = VT[:self.s_dim].T*s[:self.s_dim]/jnp.sqrt(x.shape[0] - 1)
      else:

        if self.is_tall:
          A_init = jax.nn.initializers.glorot_normal(in_axis=-1, out_axis=-2, dtype=x.dtype)
        else:
          A_init = jax.nn.initializers.glorot_normal(in_axis=-2, out_axis=-1, dtype=x.dtype)
        self.A = A_init(k2, shape=(self.t_dim, self.s_dim))
    else:
      self.scale_params = params["scale"]
      self.network_params = params["network"]
      self.A = params["A"]

    self.ATA = self.A.T@self.A
    self.ATA_inv = jnp.linalg.inv(self.ATA)

    if inverse == False:
      # Pseudo-inverse of t
      # s = solve(self.ATA, mvp(self.A.T, t)) # There is a bug in jnp.linalg.solve
      s = mvp(self.ATA_inv, mvp(self.A.T, t))

      # Projection of t
      t_proj = mvp(self.A, s)

      # Orthogonal component
      gamma_perp = t - t_proj
      self.gamma_perp = gamma_perp

      # Create the features.
      cond = t if self.condition_on_t else s
      theta = self.network(cond, aux=None, params=self.network_params, rng_key=rng_key, is_training=is_training)
      theta *= self.scale_params # Initialize to identity
      mu, diag_cov = jnp.split(theta, 2, axis=-1)
      diag_cov = util.square_plus(diag_cov, gamma=1.0) + 1e-4

    else:
      k1, k2 = random.split(rng_key, 2)

      # Projection
      t_proj = jnp.einsum("ij,...j->...i", self.A, s)

      if kwargs.get("gamma_perp", None) is None:
        # Create the features.
        cond = t if self.condition_on_t else s
        theta = self.network(cond, aux=None, params=self.network_params, rng_key=k1, is_training=is_training)
        theta *= self.scale_params # Initialize to identity
        mu, diag_cov = jnp.split(theta, 2, axis=-1)
        diag_cov = util.square_plus(diag_cov, gamma=1.0) + 1e-4

        # Sample orthogonal noise
        noise = random.normal(k2, mu.shape)
        gamma = mu + noise*jnp.sqrt(diag_cov)

        # Orthogonalize the noise
        # gamma_perp = gamma - mvp(self.A, solve(self.ATA, mvp(self.A.T, gamma))) # There is a bug in jnp.linalg.solve
        gamma_perp = gamma - mvp(self.A, mvp(self.ATA_inv, mvp(self.A.T, gamma)))

      else:
        gamma_perp = kwargs.get("gamma_perp", None)

      # Compute t
      t = t_proj + gamma_perp

    # Log likelihood contribution
    if no_llc == False:
      llc = jax.vmap(logZ, in_axes=(0, None, 0))(mu - gamma_perp, self.A, diag_cov)
    else:
      llc = jnp.zeros(x.shape[:1])

    z = s if inverse == False else t

    return z, llc

class WideMVP(TallMVP):
  def __init__(self,
               output_dim,
               create_network):
    self.t_dim = output_dim
    self.create_network = create_network
    self.is_tall = False

  def __call__(self, *args, inverse=False, **kwargs):
    z, llc = super().__call__(*args, inverse=not inverse, **kwargs)
    return z, -llc

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

  mvp = nux.WideMVP(output_dim=4, create_network=create_network)
  mvp = nux.Invert(mvp)

  flow = nux.Sequential([mvp,
                         nux.UnitGaussianPrior()])

  z, log_px1 = flow(x, rng_key=rng_key)
  params = flow.get_params()


  reconstr, log_px2 = flow(z, params=params, rng_key=rng_key, inverse=True, reconstruction=True)
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