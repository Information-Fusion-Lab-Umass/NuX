import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import nux.util as util
import einops
from nux.flows.bijective.nonlinearities import SquareLogit, SquareSigmoid
from nux.flows.bijective.affine import StaticShiftScale, StaticScale, CenterAndScale
from nux.flows.base import Sequential, NoOp

################################################################################################################

class SineWave():
  """
  Assuming x in R, returns (z in (-1, 1), k in Z).
  """
  def __init__(self,
               flow,
               feature_network=None,
               period=1,
               continuous=False):
    self.flow            = flow
    self.feature_network = feature_network
    if feature_network is None:
      self.feature_network = NoOp()
    self.period = period

    self.continuous = continuous

  def get_params(self):
    return {"feature_network": self.feature_network.get_params(),
            "qugx": self.flow.get_params()}

  def forward(self, x):
    angular_freq = 2*jnp.pi/self.period
    scaled_x = angular_freq*x

    # Compute the transform
    z = jnp.sin(scaled_x)

    # Get the log det
    log_det = jnp.log(jnp.abs(jnp.cos(scaled_x)))

    sum_axes = util.last_axes(x.shape[1:])
    log_det = jnp.sum(log_det, axis=sum_axes)

    log_det += util.list_prod(x.shape[1:])*jnp.log(angular_freq)

    # Find out how many half cycles x is at
    if self.continuous == False:
      half_cycles = util.st_round(2*x/self.period)
    else:
      half_cycles = 2*x/self.period
    return z, log_det, half_cycles

  def inverse(self, z, half_cycles):
    angular_freq = 2*jnp.pi/self.period

    # Inverse transform
    x = 1/angular_freq*jnp.arcsin(((-1)**half_cycles)*z) + self.period/2*half_cycles

    # Log det
    log_det = jnp.log(jnp.abs(jnp.cos(angular_freq*x)))

    sum_axes = util.last_axes(x.shape[1:])
    log_det = jnp.sum(log_det, axis=sum_axes)

    log_det += util.list_prod(x.shape[1:])*jnp.log(angular_freq)

    return x, log_det

  def __call__(self, x, params=None, inverse=False, rng_key=None, half_cycles=None, is_training=True, **kwargs):
    if params is None:
      self.f_params = None
      self.q_params = None
    else:
      self.f_params = params["feature_network"]
      self.q_params = params["qugx"]

    k1, k2, k3 = random.split(rng_key, 3)

    if inverse == False:
      z, log_det, half_cycles = self.forward(x)
      self.half_cycles = jnp.round(half_cycles)

      # Evaluate the number of half_cycles with a discrete flow
      f = self.feature_network(z, aux=None, params=self.f_params, rng_key=k1)

      if is_training == True:
      # if True or is_training == True:
        _, log_pmf = self.flow(half_cycles, aux=f, params=self.q_params, inverse=False, rng_key=k2)
      else:
        # Importance sample
        n_importance_samples = 256
        noise = random.uniform(k3, (n_importance_samples, *self.half_cycles.shape)) - 0.5
        def apply(dequantized_half_cycles):
          _, log_pmf = self.flow(dequantized_half_cycles, aux=f, params=self.q_params, inverse=False, rng_key=k2)
          return log_pmf

        log_pmfs = jax.vmap(apply)(self.half_cycles + noise)
        log_pmf = jax.scipy.special.logsumexp(log_pmfs, axis=0)
    else:
      # Sample the number of half_cycles
      if half_cycles is None:
        f = self.feature_network(x, aux=None, params=self.f_params, rng_key=k1)
        flow_in = jnp.zeros(x.shape, dtype=jnp.int32)
        half_cycles, log_pmf = self.flow(flow_in, aux=f, params=self.q_params, inverse=True, rng_key=k2, reconstruction=False)
        half_cycles = jnp.round(half_cycles)
      else:
        log_pmf = jnp.zeros(x.shape[:1])

      z, log_det = self.inverse(x, half_cycles)

    log_det += log_pmf
    return z, log_det

################################################################################################################

class TriangleWave(SineWave):

  def forward(self, x):
    freq = 1/self.period
    arg = jnp.floor(2*freq*x + 0.5)
    z = 4*freq*(x - 0.5*self.period*arg)*(-1)**arg

    # Get the log det
    log_det = util.list_prod(x.shape[1:])*jnp.log(4*freq)

    # Find out how many half cycles x is at
    if self.continuous == False:
      half_cycles = util.st_round(2*x/self.period)
    else:
      half_cycles = 2*x/self.period
    return z, log_det, half_cycles

  def inverse(self, z, half_cycles):
    freq = 1/self.period

    x = 0.5*half_cycles*self.period + 0.25*self.period*z*(-1)**half_cycles

    # Get the log det
    log_det = util.list_prod(x.shape[1:])*jnp.log(4*freq)
    return x, log_det

class SawtoothWave(SineWave):

  def forward(self, x):
    z = 2*jnp.remainder(x/self.period - 0.5, 1) - 1
    log_det = util.list_prod(x.shape[1:])*jnp.log(2/self.period)

    # Find out how many cycles x is at
    if self.continuous == False:
      cycles = util.st_round(x/self.period)
    else:
      cycles = x/self.period
    return z, log_det, cycles

  def inverse(self, z, cycles):
    x = self.period*z/2 + self.period*cycles
    log_det = util.list_prod(x.shape[1:])*jnp.log(2/self.period)
    return x, log_det

################################################################################################################

class _LogitMixin():
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.unsquash = Sequential([StaticShiftScale(2.0, -1.0),
                                CenterAndScale(0.999), # Bounds between (-38, 38) or so
                                SquareLogit()])

  def __call__(self, x, *args, inverse=False, **kwargs):
    if inverse == False:
      z, log_det = super().__call__(x, *args, inverse=False, **kwargs)
      z, log_det_unsquash = self.unsquash(z, inverse=False)
    else:
      z, log_det_unsquash = self.unsquash(x, inverse=True)
      z, log_det = super().__call__(z, *args, inverse=True, **kwargs)

    log_det = log_det + log_det_unsquash
    return z, log_det

class SineLogit(SineWave, _LogitMixin):
  pass

class TriangleLogit(TriangleWave, _LogitMixin):
  pass

class SawtoothLogit(SawtoothWave, _LogitMixin):
  pass

################################################################################################################

if __name__ == "__main__":
  from debug import *
  import matplotlib.pyplot as plt
  # from nux.priors.discrete import DiscreteLogisticPrior
  from nux.priors.gaussian import TruncatedUnitGaussianPrior, GaussianPrior

  H, W, C = 8, 8, 8
  rng_key = random.PRNGKey(0)
  # x = random.normal(rng_key, (20, H, W, C))
  x = random.normal(rng_key, (10000, 2))
  # x = jnp.linspace(-2, 2, 100)[None]

  # q_flow = DiscreteLogisticPrior()
  q_flow = GaussianPrior()

  flow = Sequential([TriangleWave(q_flow, None, period=2), TruncatedUnitGaussianPrior()])
  # flow = Sequential([SineWave(q_flow, None), TruncatedUnitGaussianPrior()])
  # flow = Sequential([SawtoothWave(q_flow, None, period=10), TruncatedUnitGaussianPrior()])

  z, log_px1 = flow(x, rng_key=rng_key)
  params = flow.get_params()

  half_cycles = flow.layers[0].half_cycles
  reconstr, log_px2 = flow(z, params=params, rng_key=rng_key, inverse=True, reconstruction=True, half_cycles=half_cycles)
  # import pdb; pdb.set_trace()

  # Evaluate the log likelihood of a lot of samples
  samples, log_px = flow(jnp.zeros((10000, 2)), params=params, rng_key=rng_key, inverse=True)

  import pdb; pdb.set_trace()

  # Get the exact log likelihood
  import scipy.stats
  from scipy.stats import gaussian_kde
  kernel = gaussian_kde(samples.T)

  px = kernel(samples.T)
  true_log_px = jnp.log(px)
  plt.hist(log_px - true_log_px, bins=50);plt.show()

  mask = jnp.linalg.norm(samples, axis=-1) < 0.5

  import pdb; pdb.set_trace()

  # fig, (ax1, ax2) = plt.subplots(1, 2);ax1.plot(x.ravel(), z.ravel());ax2.scatter(x.ravel(), reconstr.ravel(), c=flow.half_cycles.ravel());plt.show()