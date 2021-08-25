import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
from typing import Optional, Mapping, Callable
from nux.flows.base import NoOp

__all__ = ["Padding"]

class Padding():
  """
  Padding along the last axis
  """
  def __init__(self,
               padding_dim: int,
               flow: Callable,
               feature_network=None):

    self.padding_dim     = padding_dim
    self.flow            = flow
    if feature_network is None:
      feature_network = NoOp()
    self.feature_network = feature_network

  def get_params(self):
    return {"feature_network": self.feature_network.get_params(),
            "qugx": self.flow.get_params()}

  def __call__(self, x, params=None, inverse=False, rng_key=None, **kwargs):
    if params is None:
      self.f_params = None
      self.q_params = None
    else:
      self.f_params = params["feature_network"]
      self.q_params = params["qugx"]

    k1, k2 = random.split(rng_key, 2)
    C = x.shape[-1]

    outputs = {}

    if inverse == False:

      # Extract features to condition on
      f = self.feature_network(x, aux=None, params=self.f_params, rng_key=k1)

      # Generate noise
      flow_in = jnp.zeros(x.shape[:-1] + (self.padding_dim,))
      noise, log_qugs = self.flow(flow_in, aux=f, params=self.q_params, inverse=True, rng_key=k2)

      # Concatenate the padding noise
      z = jnp.concatenate([x, noise], axis=-1)
    else:
      # Split the noise and the image
      z_dim = x.shape[-1] - self.padding_dim
      z, noise = x[...,:z_dim], x[...,z_dim:]

      # Evaluate the likelihood of the dequantization noise
      f = self.feature_network(z, aux=None, params=self.f_params, rng_key=k1)
      _, log_qugs = self.flow(noise, aux=f, params=self.q_params, inverse=False, rng_key=k2)

    # Compute the full likelihood contribution
    log_det = -log_qugs

    return z, log_det
