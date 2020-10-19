import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping
from nux.flows.base import *
import nux.util as util
import numpyro; numpyro.set_platform("gpu")
import numpyro.distributions as dists
from jax.scipy.special import logsumexp
from haiku._src.typing import PRNGKey

__all__ = ["UnitGaussianPrior",
           "GMMPrior"]

class UnitGaussianPrior(Layer):

  def __init__(self, name: str="unit_gaussian_prior", **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           t: Optional[float]=1.0,
           ignore_prior: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    outputs = {}
    x_shape = self.get_unbatched_shapes(sample)["x"]
    sum_axes = tuple(-jnp.arange(1, 1 + len(x_shape)))

    normal = dists.Normal(0, t)

    if sample == False:
      x = inputs["x"]
      log_pz = normal.log_prob(x).sum(axis=sum_axes)
      outputs = {"x": x, "log_pz": log_pz}
    else:
      z = inputs["x"]
      if ignore_prior:
        outputs = {"x": z, "log_pz": jnp.zeros(self.batch_shape)}
      else:
        x = normal.sample(rng, z.shape)
        log_pz = normal.log_prob(x).sum(axis=sum_axes)
        outputs = {"x": x, "log_pz": log_pz}

    return outputs

################################################################################################################

class GMMPrior(Layer):

  def __init__(self, n_classes: int, name: str="gmm_prior", **kwargs):
    super().__init__(name=name, **kwargs)
    self.n_classes = n_classes

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           ignore_prior: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    y = inputs.get("y", -1)
    outputs = {}
    x_shape = self.get_unbatched_shapes(sample)["x"]
    sum_axes = tuple(-jnp.arange(1, 1 + len(x_shape)))
    x_flat = x.reshape(self.batch_shape + (-1,))

    # Keep these fixed.  Learning doesn't make much difference apparently.
    means         = hk.get_state("means", shape=(self.n_classes, x_flat.shape[-1]), dtype=x.dtype, init=hk.initializers.RandomNormal())
    log_diag_covs = hk.get_state("log_diag_covs", shape=(self.n_classes, x_flat.shape[-1]), dtype=x.dtype, init=jnp.zeros)

    # Compute the log pdfs of each mixture component
    normal = dists.Normal(means, jnp.exp(log_diag_covs))
    log_pdfs = self.auto_batch(normal.log_prob)(x_flat)
    log_pdfs = log_pdfs.sum(axis=-1)

    if sample == False:
      # Compute p(x,y) = p(x|y)p(y) if we have a label, p(x) otherwise
      def log_prob(y, log_pdfs):
        return jax.lax.cond(y >= 0,
                            lambda a: log_pdfs[y] + jnp.log(self.n_classes),
                            lambda a: logsumexp(log_pdfs) - jnp.log(self.n_classes),
                            None)
      outputs["log_pz"] = self.auto_batch(log_prob)(y, log_pdfs)
      outputs["x"] = x

    else:
      if ignore_prior:
        outputs = {"x": x, "log_pz": jnp.array(0.0)}
      else:
        # Sample from all of the clusters
        xs = normal.sample(rng)

        def sample(log_pdfs, y, rng):

          def no_label(y):
            y = dists.CategoricalLogits(jnp.zeros(self.n_classes)).sample(rng, (1,))[0]
            return y, logsumexp(log_pdfs) - jnp.log(self.n_classes)

          def with_label(y):
            return y, log_pdfs[y] - jnp.log(self.n_classes)

          # Either sample or use a specified cluster
          return jax.lax.cond(y < 0, no_label, with_label, y)

        n_keys = int(jnp.prod(jnp.array(self.batch_shape)))
        rngs = random.split(rng, n_keys).reshape(self.batch_shape + (-1,))
        y, log_pz = self.auto_batch(sample)(log_pdfs, y, rngs)

        # Take a specific cluster
        outputs = {"x": xs[y].reshape(x.shape), "log_pz": log_pz}

    outputs["prediction"] = jnp.argmax(log_pdfs)

    return outputs
