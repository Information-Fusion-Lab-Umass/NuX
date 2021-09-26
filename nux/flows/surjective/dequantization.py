import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
from typing import Optional, Mapping, Callable
from nux.flows.bijective.affine import StaticScale
from nux.flows.bijective.nonlinearities import Sigmoid

__all__ = ["UniformDequantization",
           "VariationalDequantization",
           "FusedDequantizationAndPadding"]

class UniformDequantization():

  def __init__(self, n_samples: int=1):
    """ Uniform dequantization.  See section 3.1 here https://arxiv.org/pdf/1511.01844.pdf
    Args:
      name : Optional name for this module.
    """
    self.n_samples = n_samples

  def get_params(self):
    return {}

  def __call__(self, x, params=None, inverse=False, rng_key=None, **kwargs):

    if inverse == False:
      noise = random.uniform(rng_key, x.shape + (self.n_samples,))
      noise = noise.mean(axis=-1) # Bates distribution
      z = x + noise
    else:
      z = util.st_floor(x)

    log_det = jnp.zeros(x.shape[:1])
    return z, log_det

################################################################################################################

class VariationalDequantization():
  def __init__(self, flow, feature_network=None):
    """ Variational dequantization https://arxiv.org/pdf/1902.00275.pdf
    """
    self.flow = flow
    self.feature_network = feature_network

  def get_params(self):
    return dict(log_qugx=self.flow.get_params(),
                feature_params=self.feature_network.get_params())

  def __call__(self, x, params=None, inverse=False, rng_key=None, **kwargs):
    if params is None:
      self.q_params = None
      self.feature_params = None
    else:
      self.q_params = params["log_qugx"]
      self.feature_params = params["feature_params"]

    k1, k2 = random.split(rng_key, 2)

    if inverse == False:
      if self.feature_network is not None:
        aux = self.feature_network(x, params=self.feature_params, rng_key=k2)
      else:
        aux = None
      noise, log_qugx = self.flow(jnp.zeros(x.shape), aux=aux, params=self.q_params, rng_key=k1, inverse=True)
      z = x + noise
    else:
      z_continuous = x
      z = util.st_floor(z_continuous)
      noise = z_continuous - z

      if self.feature_network is not None:
        aux = self.feature_network(z, params=self.feature_params, rng_key=k2)
      else:
        aux = None
      _, log_qugx = self.flow(noise, aux=aux, params=self.q_params, rng_key=k1, inverse=False)

    return z, -log_qugx

################################################################################################################

class FusedDequantizationAndPadding():
  """
  Full preprocessing for an image.
  Dequantization -> scale -> logit -> padding
  """
  def __init__(self,
               quantize_bits: int,
               output_channel: int,
               flow: Callable,
               feature_network: Callable):

    self.quantize_bits   = quantize_bits
    self.out_channel     = output_channel
    self.flow            = flow
    self.feature_network = feature_network
    self.scale           = StaticScale(2**self.quantize_bits)
    self.sigmoid         = Sigmoid()

  def get_params(self):
    return {"feature_network": self.feature_network.get_params(),
            "qugx": self.flow.get_params(),
            "scale": self.scale.get_params(),
            "sigmoid": self.sigmoid.get_params()}

  def __call__(self, x, params=None, aux=None, inverse=False, rng_key=None, **kwargs):
    if params is None:
      self.f_params = None
      self.q_params = None
      self.scale_params = None
      self.sigmoid_params = None
    else:
      self.f_params = params["feature_network"]
      self.q_params = params["qugx"]
      self.scale_params = params["scale"]
      self.sigmoid_params = params["sigmoid"]

    k1, k2 = random.split(rng_key, 2)

    assert len(x.shape[1:]) == 3, "Only supporting 3d inputs"

    if inverse == False:
      C = x.shape[-1]
      assert self.out_channel > C
      self.data_channel = C

      # Extract features to condition on
      f = self.feature_network(x, aux=None, params=self.f_params, rng_key=k1, **kwargs)

      # Generate noise
      flow_in = jnp.zeros(x.shape[:-1] + (self.out_channel,))
      noise, log_qugs = self.flow(flow_in, aux=f, params=self.q_params, inverse=True, rng_key=k2, **kwargs)

      # Squash part of the noise to be between 0 and 1
      dequant_noise, log_det1 = self.sigmoid(noise[...,:C], params=self.sigmoid_params, rng_key=rng_key, inverse=False)

      # Add this noise for dequantization
      x += dequant_noise

      # Scale the dequantized image between 0 and 1
      x, log_det2 = self.scale(x, params=self.scale_params, rng_key=rng_key, inverse=False)

      # Unsquash the scaled dequantized image to the reals
      x, log_det3 = self.sigmoid(x, params=self.sigmoid_params, rng_key=rng_key, inverse=True, scale=0.05)

      # Concatenate the padding noise
      z = jnp.concatenate([x, noise[...,C:]], axis=-1)

    else:
      C = self.data_channel

      # Split the noise and the image
      x, padding_noise = x[...,:C], x[...,C:]

      # Squash the image from the reals to (0, 1)
      x, log_det3 = self.sigmoid(x, params=self.sigmoid_params, rng_key=rng_key, inverse=False, scale=0.05)
      # x, log_det3 = self.sigmoid(x, params=self.sigmoid_params, rng_key=rng_key, inverse=False)

      # Scale the image from (0, 1) to the full range
      x, log_det2 = self.scale(x, params=self.scale_params, rng_key=rng_key, inverse=True)

      # Extract dequantization noise
      z = util.st_floor(x)
      dequant_noise = x - z

      # Unsquash the dequantization noise so that we can evaluate it
      noise, log_det1 = self.sigmoid(dequant_noise, params=self.sigmoid_params, rng_key=rng_key, inverse=True)

      # Evaluate the likelihood of the dequantization noise
      f = self.feature_network(z, aux=None, params=self.f_params, rng_key=k1, **kwargs)
      full_noise = jnp.concatenate([noise, padding_noise], axis=-1)
      _, log_qugs = self.flow(full_noise, aux=f, params=self.q_params, inverse=False, rng_key=k2, **kwargs)

    # Compute the full likelihood contribution
    log_det = -log_qugs
    log_det += log_det1
    log_det += log_det2
    log_det -= log_det3

    return z, log_det
