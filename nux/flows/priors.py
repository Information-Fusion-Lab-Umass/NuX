import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence
from nux.internal.layer import InvertibleLayer
import nux.util as util
from jax.scipy.special import logsumexp
from haiku._src.typing import PRNGKey
import nux.vae as vae

__all__ = ["UnitGaussianPrior",
           "ParametrizedGaussianPrior",
           "AffineGaussianPriorDiagCov",
           "GMMPrior"]

class UnitGaussianPrior(InvertibleLayer):

  def __init__(self,
               name: str="unit_gaussian_prior"
  ):
    """ Unit Gaussian prior
    Args:
      name: Optional name for this module.
    """
    super().__init__(name=name)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           t: Optional[float]=1.0,
           reconstruction: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    @self.auto_batch
    def unit_gaussian(x):
      return -0.5*(1/(t**2)*jnp.sum(x.ravel()**2) + x.size*jnp.log(t**2*2*jnp.pi))

    x = inputs["x"]

    if sample == True and reconstruction == False:
      x = random.normal(rng, x.shape)*t

    log_pz = unit_gaussian(x)
    outputs = {"x": x, "log_pz": log_pz}
    return outputs

################################################################################################################

class ParametrizedGaussianPrior(InvertibleLayer):

  def __init__(self,
               create_network: Optional[Callable]=None,
               network_kwargs: Optional=None,
               name: str="parametrized_gaussian_prior"
  ):
    """ Unit Gaussian prior
    Args:
      name: Optional name for this module.
    """
    super().__init__(name=name)
    self.create_network = create_network
    self.network_kwargs = network_kwargs

  @property
  def network(self):
    if hasattr(self, "_net") == False:
      out_shape = self.x_shape[:-1] + (2*self.x_shape[-1],)
      network = vae.ParametrizedGaussian(out_shape=out_shape,
                                         create_network=self.create_network,
                                         network_kwargs=self.network_kwargs)
      self._net = network

    return self._net

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           reconstruction: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    self.x_shape = self.get_unbatched_shapes(sample)["x"]

    assert "condition" in inputs
    condition = inputs["condition"]

    # Pass the condition through the parametrized gaussian
    outputs = self.network({"x": condition}, rng, **kwargs)
    mu, log_diag_cov = outputs["mu"], outputs["log_diag_cov"]

    x = inputs["x"]

    if sample == True and reconstruction == False:
      x = outputs["x"]

    @self.auto_batch
    def logpdf(x, mu, log_diag_cov):
      dx = x - mu
      log_px = -0.5*jnp.sum(dx.ravel()**2*jnp.exp(-log_diag_cov.ravel()), axis=-1)
      return log_px - 0.5*jnp.sum(log_diag_cov.ravel()) - 0.5*x.size*jnp.log(2*jnp.pi)

    log_pz = logpdf(x, mu, log_diag_cov)

    outputs = {"x": x, "log_pz": log_pz}
    return outputs

################################################################################################################

class AffineGaussianPriorDiagCov(InvertibleLayer):

  def __init__(self,
               output_dim: int,
               generative_only: bool=False,
               name: str="affine_gaussian_prior"
  ):
    """ Analytic solution to int N(z|0,I)N(x|Az,Sigma)dz.
        https://arxiv.org/pdf/2006.13070v1.pdf
    Args:
      name: Optional name for this module.
    """
    super().__init__(name=name)
    if generative_only == False:
      self._output_dim = output_dim
    else:
      self._input_dim = output_dim
    self.generative_only = generative_only

  @property
  def input_shape(self):
    return self.unbatched_input_shapes["x"]

  @property
  def output_shape(self):
    return self.unbatched_output_shapes["x"]

  @property
  def input_dim(self):
    if hasattr(self, "_input_dim"):
      return self._input_dim
    return util.list_prod(self.input_shape)

  @property
  def output_dim(self):
    if hasattr(self, "_output_dim"):
      return self._output_dim
    return util.list_prod(self.output_shape)

  @property
  def z_dim(self):
    return self.output_dim if self.input_dim > self.output_dim else self.input_dim

  @property
  def x_dim(self):
    return self.input_dim if self.input_dim > self.output_dim else self.output_dim

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           reconstruction: Optional[bool]=False,
           manifold_sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    assert len(self.unbatched_input_shapes["x"]) == 1, "Only works with 1d inputs"
    assert self.z_dim < self.x_dim

    dtype = inputs["x"].dtype
    init_fun = hk.initializers.RandomNormal(0.01)
    A = hk.get_parameter("A", shape=(self.x_dim, self.z_dim), dtype=dtype, init=init_fun)
    b = hk.get_parameter("b", shape=(self.x_dim,), dtype=dtype, init=init_fun)
    log_diag_cov = hk.get_parameter("log_diag_cov", shape=(self.x_dim,), dtype=dtype, init=init_fun)
    diag_cov = jnp.exp(log_diag_cov)

    # Go from x -> z or z -> x
    if sample == False:
      x = inputs["x"]
      x -= b

      # Compute the posterior natural parameters
      J = jnp.eye(self.z_dim) + (A.T/diag_cov)@A
      J_inv = jnp.linalg.inv(J)
      sigma_inv_x = x/diag_cov
      h = jnp.dot(sigma_inv_x, A)

      # Compute the posterior parameters
      Sigma_z = J_inv
      mu_z = jnp.dot(h, Sigma_z)

      # Sample z
      Sigma_z_chol = jnp.linalg.cholesky(Sigma_z)
      noise = random.normal(rng, mu_z.shape)
      z = mu_z + jnp.dot(noise, Sigma_z_chol.T)

      # Compute the log likelihood contribution
      J_inv_h = jnp.dot(h, J_inv.T)

      llc = 0.5*jnp.sum(h*J_inv_h, axis=-1)
      llc -= 0.5*jnp.linalg.slogdet(J)[1]
      llc -= 0.5*jnp.sum(x*sigma_inv_x, axis=-1)
      llc -= 0.5*log_diag_cov.sum()
      llc -= 0.5*self.x_dim*jnp.log(2*jnp.pi)

      outputs = {"x": z, "log_pz": llc}

    else:
      k1, k2 = random.split(rng, 2)
      z = inputs["x"]

      if reconstruction == False:
        z = random.normal(k1, z.shape)

      # Sample x
      mu_x = jnp.dot(z, A.T) + b
      if manifold_sample == True:
        noise = random.normal(k2, mu_x.shape)
      else:
        noise = jnp.zeros_like(mu_x)

      x = mu_x + jnp.sqrt(diag_cov)*noise

      # If we're doing a reconstruction, we need to compute log p(x|z)
      llc = -0.5*jnp.sum(noise**2, axis=-1)
      llc -= 0.5*jnp.sum(log_diag_cov)
      llc -= 0.5*self.x_dim*jnp.log(2*jnp.pi)

      outputs = {"x": x, "log_pz": llc}

    return outputs

################################################################################################################

class GMMPrior(InvertibleLayer):

  def __init__(self,
               n_classes: int,
               name: str="gmm_prior"
  ):
    """ Gaussian mixture model prior.  Can be used for classificaiton.
    Args:
      n_classes: Number of mixture components.
      name     : Optional name for this module.
    """
    super().__init__(name=name)
    self.n_classes = n_classes

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           reconstruction: Optional[bool]=False,
           is_training: bool=True,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    outputs = {}
    x_shape = self.get_unbatched_shapes(sample)["x"]

    # Make sure that we're using 1d inputs
    x = inputs["x"].reshape(self.batch_shape + (-1,))
    x_dim = x.shape[-1]

    # Work with one-hot labels
    y_one_hot = inputs.get("y", None)
    if y_one_hot is not None:
      assert y_one_hot.shape == self.batch_shape + (self.n_classes,)
      y_one_hot *= 1.0
    else:
      if sample == False:
        # Assign equal probability to each class
        y_one_hot = jnp.ones(self.batch_shape + (self.n_classes,))
      else:
        # Sample class labels
        y = random.randint(rng, minval=0, maxval=self.n_classes, shape=self.batch_shape)
        y_one_hot = y[...,None] == jnp.arange(self.n_classes)[...,:]
        y_one_hot *= 1.0

    # GMM parameters.  Assume uniform mixture component weights so that things are differentiable.
    means         = hk.get_parameter("means", shape=(self.n_classes, x_dim), dtype=x.dtype, init=hk.initializers.RandomNormal())
    log_diag_covs = hk.get_parameter("log_diag_covs", shape=(self.n_classes, x_dim), dtype=x.dtype, init=jnp.ones)
    diag_covs = util.proximal_relu(log_diag_covs) + 1e-3
    log_diag_covs = jnp.log(diag_covs)

    # Sample a new input
    if sample == True and reconstruction == False:
      # Sample from all of the clusters
      noise = random.normal(rng, self.batch_shape + (self.n_classes, x_dim))
      xs = means + jnp.exp(0.5*log_diag_covs)*noise

      # Select the mixture component
      x = xs*y_one_hot[...,None]
      x = x.sum(axis=-2)

    # Evaluate the log pdf for each mixture component
    @partial(jax.vmap, in_axes=(0, 0, None))
    def diag_gaussian(mean, log_diag_cov, x):
      dx = x - mean
      log_pdf = jnp.sum(dx**2*jnp.exp(-log_diag_cov), axis=-1)
      log_pdf += log_diag_cov.sum()
      log_pdf += x_dim*jnp.log(2*jnp.pi)
      return -0.5*log_pdf

    # Last axis will be across the mixture components
    log_pdfs = self.auto_batch(partial(diag_gaussian, means, log_diag_covs))(x)

    # Make a class prediction
    y_pred = jnp.argmax(log_pdfs, axis=-1)
    y_pred_one_hot = y_pred[...,None] == jnp.arange(self.n_classes)[...,:]
    y_pred_one_hot *= 1.0

    # Compute p(x,y) = p(x|y)p(y) if we have a label, p(x) otherwise.
    # If we have a label, zero out all but the label index then reduce.
    # Otherwise, reduce over all of the indices.
    if is_training:

      # Apply the label masks
      if "y_is_labeled" in inputs:
        y_is_labeled = inputs["y_is_labeled"][...,None].astype(bool)
        y_one_hot = y_one_hot*y_is_labeled + jnp.ones_like(y_one_hot)*(~y_is_labeled)

      log_pz = util.lse(log_pdfs, b=y_one_hot, axis=-1)
      # log_pz = logsumexp(log_pdfs, b=y_one_hot, axis=-1)
    else:
      # If we're doing classification, use the predicted label
      if "y" in inputs:
        log_pz = util.lse(log_pdfs, b=y_pred_one_hot, axis=-1)
      else:
        log_pz = logsumexp(log_pdfs, axis=-1)

    # Account for p(y)=1/N or 1/N when we take the mean
    log_pz -= jnp.log(self.n_classes)

    # p(y|x) is a categorical distribution
    log_pygx = jax.nn.log_softmax(log_pdfs)
    if is_training:
      log_pygx *= y_one_hot

      if "y_is_labeled" in inputs:
        # This time, zero out values that aren't labeled
        log_pygx *= y_is_labeled

    else:
      if "y" in inputs:
        log_pygx *= y_pred_one_hot

    log_pygx = log_pygx.sum(axis=-1)

    # Reshape the output
    x = x.reshape(self.batch_shape + x_shape)

    outputs = {"x": x, "log_pz": log_pz, "log_pygx": log_pygx}
    outputs["prediction"] = y_pred
    outputs["prediction_one_hot"] = outputs["prediction"][...,None] == jnp.arange(self.n_classes)[...,:]
    return outputs

################################################################################################################
