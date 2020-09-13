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

class UnitGaussianPrior(AutoBatchedLayer):

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

    normal = dists.Normal(0, t)

    if sample == False:
      x = inputs["x"]
      log_pz = normal.log_prob(x).sum()
      outputs = {"x": x, "log_pz": log_pz}
    else:
      z = inputs["x"]
      if ignore_prior:
        outputs = {"x": z, "log_pz": jnp.array(0.0)}
      else:
        x = normal.sample(rng, z.shape)
        log_pz = normal.log_prob(x).sum()
        outputs = {"x": x, "log_pz": log_pz}
    return outputs

################################################################################################################

class GMMPrior(AutoBatchedLayer):

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

    # Keep these fixed.  Learning doesn't make much difference apparently.
    means         = hk.get_state("means", shape=(self.n_classes, x.shape[-1]), dtype=x.dtype, init=hk.initializers.RandomNormal())
    log_diag_covs = hk.get_state("log_diag_covs", shape=(self.n_classes, x.shape[-1]), dtype=x.dtype, init=jnp.zeros)

    # Compute the log pdfs of each mixture component
    normal = dists.Normal(means, jnp.exp(log_diag_covs))
    log_pdfs = normal.log_prob(x).sum(axis=-1)

    if sample == False:
      # Compute p(x,y) = p(x|y)p(y) if we have a label, p(x) otherwise
      outputs["log_pz"] = jax.lax.cond(y >= 0,
                                       lambda a: log_pdfs[y] + jnp.log(self.n_classes),
                                       lambda a: logsumexp(log_pdfs) - jnp.log(self.n_classes),
                                       None)
      outputs["x"] = x

    else:
      if ignore_prior:
        outputs = {"x": x, "log_pz": jnp.array(0.0)}
      else:
        # Sample from all of the clusters
        xs = normal.sample(rng)

        def no_label(y):
          rng = hk.next_rng_key() # NEED TO SPLIT THE KEY
          y = dists.CategoricalLogits(jnp.zeros(self.n_classes)).sample(rng, (1,))[0]
          return y, logsumexp(log_pdfs) - jnp.log(self.n_classes)

        def with_label(y):
          return y, log_pdfs[y] - jnp.log(self.n_classes)

        # Either sample or use a specified cluster
        y, log_pz = jax.lax.cond(y < 0, no_label, with_label, y)

        # Take a specific cluster
        outputs = {"x": xs[y], "log_pz": log_pz}

    outputs["prediction"] = jnp.argmax(log_pdfs)

    return outputs
