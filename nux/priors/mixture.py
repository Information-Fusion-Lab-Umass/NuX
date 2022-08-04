import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
from typing import Optional, Mapping, Callable, Sequence
import nux.util as util

__all__ = ["Mixture"]

class Mixture():

  def __init__(self, K, component, cluster_init=False):
    self.K = K
    self.component = component
    self.cluster_init = cluster_init

  def get_params(self):
    return dict(component=self.component_params, pi=self.pi)

  def __call__(self, x, params=None, rng_key=None, inverse=False, reconstruction=False, **kwargs):

    if params is None:
      def apply_and_params(key):
        z, llc = self.component(x, params=None, rng_key=key, inverse=inverse, reconstruction=reconstruction, **kwargs)
        return self.component.get_params()

      keys = random.split(rng_key, self.K)
      if self.cluster_init == False:
        self.component_params = jax.vmap(apply_and_params)(keys)
        self.pi = jnp.zeros((self.K,))
        # self.pi += random.normal(rng_key, self.pi.shape)*0.01
      else:
        x_flat = x.reshape(x.shape[:1] + (-1,))
        # Cluster x into K groups and then initialize each component with a group
        @partial(jax.vmap, in_axes=(None, 0))
        @partial(jax.vmap, in_axes=(0, None))
        def distance(x, y):
          return jnp.sum((x - y)**2)
        adjancency_matrix = distance(x_flat, x_flat)
        cluster_assignments = util.spectral_cluster(adjancency_matrix, k=self.K)

        mask = cluster_assignments[:,None] == jnp.arange(self.K)

        component_params = []
        for i, key in enumerate(keys):
          x_masked = x_flat[mask[:,i]].reshape((-1,) + x.shape[1:])
          self.component(x_masked, params=None, rng_key=key, inverse=inverse, reconstruction=reconstruction, **kwargs)
          component_params.append(self.component.get_params())
        self.component_params = jax.tree_util.tree_map(lambda *xs: jnp.array(xs), *component_params)

        # Based on cluster assignments
        self.pi = jnp.log(mask.sum(axis=0)/x.shape[0])

    else:
      self.pi = params["pi"]
      self.component_params = params["component"]

    if inverse and reconstruction == False:
      k = random.categorical(rng_key, logits=self.pi, axis=0, shape=x.shape[:1])

      def sample(x, params, key):
        # Assume x is unbatched
        x, log_pz = self.component(x[None], params=params, rng_key=key, inverse=True, reconstruction=False, **kwargs)
        return x[0], log_pz[0]

      keys = random.split(rng_key, x.shape[0])
      selected_params = jax.tree_util.tree_map(lambda x: x[k], self.component_params)
      x, log_pz = jax.vmap(sample)(x, selected_params, keys)

    else:
      # Compute the pmf for all of the mixture components
      def apply_over_each_component(component_params, key):
        _, log_pz = self.component(x, rng_key=key, params=component_params, inverse=False, **kwargs)
        return log_pz

      keys = random.split(rng_key, self.K)
      log_pzs = jax.vmap(apply_over_each_component)(self.component_params, keys)
      pi = jax.nn.softmax(self.pi, axis=0)
      log_pz = util.lse(log_pzs, b=pi[:,None], axis=0)

    return x, log_pz

if __name__ == "__main__":
  from debug import *
  import matplotlib.pyplot as plt
  import scipy.stats
  from .gaussian import GaussianPrior

  rng_key = random.PRNGKey(0)
  x = random.randint(rng_key, minval=-10, maxval=10, shape=(1000,))

  # x = jnp.arange(-100, 100)*1.0
  # x = x[:,None]

  x = random.randint(rng_key, minval=-10, maxval=10, shape=(2000, 2))

  prior = Mixture(5, GaussianPrior(use_scale=True, random_init=True), cluster_init=False)
  _, log_pz = prior(x, rng_key=rng_key)
  params = prior.get_params()

  samples, _ = prior(x, params=params, rng_key=rng_key, inverse=True)

  prior = Mixture(5, GaussianPrior(use_scale=True, random_init=False), cluster_init=True)
  prior(samples, rng_key=rng_key)

  import pdb; pdb.set_trace()

