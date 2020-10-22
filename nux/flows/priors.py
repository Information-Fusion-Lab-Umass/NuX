import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping
from nux.flows.base import *
import nux.util as util
# import numpyro; numpyro.set_platform("gpu") # Not compatible with new version of JAX!
# import numpyro.distributions as dists
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
           t: Optional[float]=0.2,
           ignore_prior: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    outputs = {}
    # x_shape = self.get_unbatched_shapes(sample)["x"]
    # sum_axes = tuple(-jnp.arange(1, 1 + len(x_shape)))

    # normal = dists.Normal(0, t)
    @self.auto_batch
    def unit_gaussian(x):
      return -0.5*(1/(t**2)*jnp.sum(x.ravel()**2) + x.size*jnp.log(t**2*2*jnp.pi))

    if sample == False:
      x = inputs["x"]
      # log_pz = normal.log_prob(x).sum(axis=sum_axes)
      log_pz = unit_gaussian(x)
      outputs = {"x": x, "log_pz": log_pz}
    else:
      z = inputs["x"]
      if ignore_prior:
        outputs = {"x": z, "log_pz": jnp.zeros(self.batch_shape)}
      else:
        # x = normal.sample(rng, z.shape)
        # log_pz = normal.log_prob(x).sum(axis=sum_axes)
        x = random.normal(rng, z.shape)*t
        log_pz = unit_gaussian(x)
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
    outputs = {}
    x_shape = self.get_unbatched_shapes(sample)["x"]
    sum_axes = tuple(-jnp.arange(1, 1 + len(x_shape)))
    x_flat = x.reshape(self.batch_shape + (-1,))
    y = inputs.get("y", jnp.ones(self.batch_shape, dtype=jnp.int32)*-1)

    # Keep these fixed.  Learning doesn't make much difference apparently.
    means         = hk.get_state("means", shape=(self.n_classes, x_flat.shape[-1]), dtype=x.dtype, init=hk.initializers.RandomNormal())
    log_diag_covs = hk.get_state("log_diag_covs", shape=(self.n_classes, x_flat.shape[-1]), dtype=x.dtype, init=jnp.zeros)

    @partial(jax.vmap, in_axes=(0, 0, None))
    def diag_gaussian(mean, log_diag_cov, x_flat):
      dx = x_flat - mean
      log_pdf = jnp.dot(dx*jnp.exp(-log_diag_cov), dx)
      log_pdf += log_diag_cov.sum()
      log_pdf += x_flat.size*jnp.log(2*jnp.pi)
      return -0.5*log_pdf

    log_pdfs = self.auto_batch(partial(diag_gaussian, means, log_diag_covs))(x_flat)

    # # Compute the log pdfs of each mixture component
    # normal = dists.Normal(means, jnp.exp(log_diag_covs))
    # log_pdfs = self.auto_batch(normal.log_prob)(x_flat)
    # log_pdfs = log_pdfs.sum(axis=-1)

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
        # xs = normal.sample(rng)
        xs = random.normal(rng, x_flat.shape)

        def sample(log_pdfs, y, rng):

          def no_label(y):
            y = random.randint(rng, minval=0, maxval=self.n_classes, shape=(1,))[0]
            # y = dists.CategoricalLogits(jnp.zeros(self.n_classes)).sample(rng, (1,))[0]
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
