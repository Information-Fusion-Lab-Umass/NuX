import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
import nux

class Mixture():
  # This is implemented in mixture.py
  # NuX can take advantage of JAX to easily create a mixture of flows

  def __init__(self, K, component):
    # K is the number of mixture components to use
    self.K = K

    # Component is the flow that we will copy K times
    self.component = component

  def get_params(self):
    return dict(component=self.component_params, pi=self.pi)

  def __call__(self, x, params=None, rng_key=None, inverse=False, reconstruction=False, **kwargs):

    if params is None:
      # Initialize the parameters for each of the flows in one pass using vmap
      @jax.vmap
      def apply_and_params(key):
        z, llc = self.component(x, params=None, rng_key=key, inverse=inverse, reconstruction=reconstruction, **kwargs)
        return self.component.get_params()

      keys = random.split(rng_key, self.K)
      self.component_params = apply_and_params(keys)

      # Create the mixture weights
      self.pi = random.normal(rng_key, (self.K,))*0.01
    else:
      self.pi = params["pi"]
      self.component_params = params["component"]

    if inverse and reconstruction == False:
      # To sample from the flow, we first need to sample a different mixture component
      # for each of the items in the batch
      k = random.categorical(rng_key, logits=self.pi, axis=0, shape=x.shape[:1])

      def sample(x, params, key):
        # Assume x is unbatched because every unbatched element
        x, log_pz = self.component(x[None], params=params, rng_key=key, inverse=True, reconstruction=False, **kwargs)
        return x[0], log_pz[0]

      # For each item in the batch, select a different mixture component
      keys = random.split(rng_key, x.shape[0])
      selected_params = jax.tree_map(lambda x: x[k], self.component_params)

      # Evaluate the flow on only the component
      x, log_pz = jax.vmap(sample)(x, selected_params, keys)

    else:
      # Compute the pmf for all of the mixture components
      def apply_over_each_component(component_params, key):
        _, log_pz = self.component(x, rng_key=key, params=component_params, inverse=False, **kwargs)
        return log_pz

      keys = random.split(rng_key, self.K)
      log_pzs = jax.vmap(apply_over_each_component)(self.component_params, keys)
      pi = jax.nn.softmax(self.pi, axis=0)

      # Weighted logsumexp to get the full log likelihood
      log_pz = util.lse(log_pzs, b=pi[:,None], axis=0)

    return x, log_pz

if __name__ == "__main__":
  from debug import *

  rng_key = random.PRNGKey(0)
  x = random.normal(rng_key, shape=(32, 8))

  # Any normalizing flow will work
  component = nux.Sequential([nux.DenseMVP(), nux.UnitGaussianPrior()])

  # Pass the component to the Mixture model
  flow = Mixture(5, component)

  # Initialize.
  flow(x, rng_key=rng_key)
  params = flow.get_params()

  # Notice that the parameters have a leading axis size of 5
  print(util.tree_shapes(params))
  import pdb; pdb.set_trace()

