import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from functools import partial
import nux.util as util
from typing import Optional, Mapping, Callable, Sequence
import nux

__all__ = ["L2"]

################################################################################################################

class L2():
  """
  """

  def __init__(self, v_flow):
    self.v_flow = v_flow

  def get_params(self):
    return dict(v_flow=self.v_flow.get_params())

  def __call__(self, x, params=None, inverse=False, rng_key=None, **kwargs):

    if params is None:
      self.flow_params = None
    else:
      self.flow_params = params["v_flow"]

    sum_axes = util.last_axes(x.shape[1:])
    dim = util.list_prod(x.shape[1:])

    if inverse == False:
      v = jnp.linalg.norm(x, axis=-1, keepdims=True)
      w = x/v

      log_det = (1 - dim)*jnp.log(v[:,0])

      # Evaluate the likelihood of v
      _, llc = self.v_flow(v, params=self.flow_params, rng_key=rng_key, inverse=False, **kwargs)

      return w, llc + log_det
    else:
      w = x

      # Sample v
      v = jnp.zeros(w.shape[:1] + (1,))
      v, llc = self.v_flow(v, params=self.flow_params, rng_key=rng_key, inverse=True, **kwargs)

      log_det = (1 - dim)*jnp.log(v[:,0])

      z = v*w
      return z, llc + log_det

################################################################################################################

if __name__ == "__main__":
  from debug import *
  import matplotlib.pyplot as plt

  H, W, C = 8, 8, 8
  rng_key = random.PRNGKey(0)
  # x = random.normal(rng_key, (20, H, W, C))
  x = random.normal(rng_key, (10000, 2))
  # x = jnp.linspace(-2, 2, 100)[None]

  flow = nux.Sequential([L2(v_flow=nux.Sequential([nux.StaticShiftScale(0.1, 1.0), nux.UnitGaussianPrior()])),
                         nux.PowerSphericalPrior()])

  z, log_px1 = flow(x, rng_key=rng_key)
  params = flow.get_params()

  # gamma_perp = flow.layers[0].gamma_perp
  # reconstr, log_px2 = flow(z, params=params, rng_key=rng_key, inverse=True, reconstruction=True, gamma_perp=gamma_perp)
  # import pdb; pdb.set_trace()

  # Evaluate the log likelihood of a lot of samples
  samples, log_px = flow(jnp.zeros((10000, x.shape[-1])), params=params, rng_key=rng_key, inverse=True)
  plt.scatter(*samples[:,[0,2]].T, c=jnp.exp(log_px), s=3, alpha=0.3);plt.show()

  # import pdb; pdb.set_trace()

  # Get the exact log likelihood
  truth_flow = nux.Sequential([nux.NonlinearCoupling(n_layers=5,
                                                     working_dim=32,
                                                     hidden_dim=64,
                                                     nonlinearity=util.square_swish,
                                                     dropout_prob=0.0,
                                                     n_resnet_layers=3,
                                                     K=8,
                                                     kind="logistic",
                                                     with_affine_coupling=True),
                               nux.UnitGaussianPrior()])

  from nux.training.trainer import DensityEstimation

  density_estimator = DensityEstimation(truth_flow,
                                        samples,
                                        n_batches=10000,
                                        batch_size=256,
                                        max_iters=10,
                                        lr=1e-2,
                                        retrain=True)
  z2, log_px2 = density_estimator(samples)

  plt.hist(log_px - log_px2, bins=1000);plt.show()
  import pdb; pdb.set_trace()

  # Get the exact log likelihood
  import scipy.stats
  from scipy.stats import gaussian_kde
  kernel = gaussian_kde(samples.T)

  px = kernel(samples.T)
  true_log_px = jnp.log(px)
  plt.hist(log_px - true_log_px, bins=1000);plt.show()

  import pdb; pdb.set_trace()