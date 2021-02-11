import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from functools import partial
import nux.util as util
from typing import Optional, Mapping, Callable, Sequence
from nux.internal.layer import Layer, InvertibleLayer
import haiku as hk
from haiku._src.typing import PRNGKey
from jax.scipy.special import gammaln, logsumexp
import nux
import nux.networks as net

__all__ = ["ParametrizedGaussian",
           "GaussianVAE"]

class ParametrizedGaussian(Layer):

  def __init__(self,
               out_shape: Sequence[int],
               create_network: Optional[Callable]=None,
               network_kwargs: Optional=None,
               name: str="parametrized_gaussian",
  ):
    self.create_network = create_network
    self.network_kwargs = network_kwargs
    self.out_shape      = out_shape
    super().__init__(name=name)

  def get_generator_network(self):

    # The user can specify a custom network
    if self.create_network is not None:
      return self.create_network(self.out_shape)

    return util.get_default_network(self.out_shape, network_kwargs=self.network_kwargs)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           no_noise: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    network = self.get_generator_network()

    # network_in = inputs["x"]
    network_out = network(inputs, rng)["x"]
    # network_out = self.auto_batch(network, expected_depth=1, in_axes=(0, None))(network_in, rng)
    mu, log_diag_cov = jnp.split(network_out, 2, axis=-1)

    diag_cov = util.proximal_relu(log_diag_cov) + 1e-5
    log_diag_cov = jnp.log(diag_cov)

    x = mu
    if no_noise == False:
      x += random.normal(rng, mu.shape)*jnp.sqrt(diag_cov)

    outputs = {"x": x,
               "mu": mu,
               "log_diag_cov": log_diag_cov}

    return outputs

################################################################################################################

class GaussianVAE(InvertibleLayer):

  def __init__(self,
               output_dim: int,
               create_generator: Optional[Callable]=None,
               create_inference: Optional[Callable]=None,
               generator_network_kwargs: Optional=None,
               inference_network_kwargs: Optional=None,
               name: str="gaussian_vae",
               **kwargs):
    self.output_dim               = output_dim
    self.create_generator         = create_generator
    self.create_inference         = create_inference
    self.generator_network_kwargs = generator_network_kwargs
    self.inference_network_kwargs = inference_network_kwargs
    super().__init__(name=name, **kwargs)

  def get_generator(self, out_shape):
    out_shape = out_shape[:-1] + (2*out_shape[-1],)

    # The user can specify a custom network
    if self.create_network is not None:
      return self.create_generator(out_shape)

    return util.get_default_network(out_shape, network_kwargs=self.generator_network_kwargs)

  def get_inference(self, out_shape):
    out_shape = out_shape[:-1] + (2*out_shape[-1],)

    # The user can specify a custom network
    if self.create_network is not None:
      return self.create_inference(out_shape)

    return util.get_default_network(out_shape, network_kwargs=self.inference_network_kwargs)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           no_noise: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    if sample == False:
      network = self.get_generator(self.unbatched_input_shapes["x"])

      network_in = inputs["x"]
      network_out = self.auto_batch(network, expected_depth=1, in_axes=(0, None))(network_in, rng)
      mu, log_diag_cov = jnp.split(network_out, 2, axis=-1)

    else:
      network = self.get_inference(self.unbatched_output_shapes["x"])

      network_in = inputs["x"]
      network_out = self.auto_batch(network, expected_depth=1, in_axes=(0, None))(network_in, rng)
      mu, log_diag_cov = jnp.split(network_out, 2, axis=-1)
