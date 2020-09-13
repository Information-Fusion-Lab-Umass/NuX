import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap, jit
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Sequence
from nux.flows.base import *
import nux.util as util

__all__ = ["GaussianMADE",
           "MAF"]

class GaussianMADE(hk.Module):

  def __init__(self, input_sel, dim, hidden_layer_sizes, method='sequential', name=None):
    super().__init__(name=name)

    # Store the dimensions for later
    self.dim = dim
    self.hidden_layer_sizes = list(hidden_layer_sizes)

    """ Build the autoregressive masks """
    layer_sizes = self.hidden_layer_sizes + [dim]

    self.input_sel = input_sel

    # Build the hidden layer masks
    masks = []
    sel = input_sel
    prev_sel = sel

    if method == 'random':
      key = hk.next_rng_key()
      keys = random.split(key, len(layer_sizes))
      key_iter = iter(keys)

    for size in layer_sizes:

      # Choose the degrees of the next layer
      if method == 'random':
        sel = random.randint(next(key_iter), shape=(size,), minval=min(jnp.min(sel), dim - 1), maxval=dim)
      else:
        sel = jnp.arange(size)%max(1, dim - 1) + min(1, dim - 1)

      # Create the new mask
      mask = (prev_sel[:,None] <= sel).astype(jnp.int32)
      prev_sel = sel
      masks.append(mask)

    # Will use these at runtime to mask the linear layers
    self.masks = tuple(masks)

    # # Build the mask for the matrix between the input and output
    # self.skip_mask = (input_sel[:,None] < input_sel).astype(jnp.int32)

    # Build the output layers.  Remember that we need to output mu and sigma.  Just need
    # a triangular matrix for the masks
    self.out_mask = (prev_sel[:,None] < input_sel).astype(jnp.int32)

  def __call__(self, inputs, **kwargs):

    w_init = hk.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='truncated_normal')

    # Main autoregressive transform
    x = inputs
    layer_sizes = [self.dim] + self.hidden_layer_sizes + [self.dim]
    for i, (mask, input_size, output_size) in enumerate(zip(self.masks, layer_sizes[:-1], layer_sizes[1:])):
      w = hk.get_parameter('w_%d'%i, [input_size, output_size], jnp.float32, init=w_init)
      b = hk.get_parameter('b_%d'%i, [output_size], jnp.float32, init=jnp.zeros)
      w_masked = w*mask
      x = jnp.dot(x, w_masked) + b

    # # Skip connection # Implemented this wrong probably
    # w_skip = hk.get_parameter('w_skip', [self.dim, self.dim], jnp.float32, init=w_init)
    # x += jnp.dot(x, w_skip*self.skip_mask)

    # Split into two parameters
    w_mu = hk.get_parameter('w_mu', [self.dim, self.dim], jnp.float32, init=w_init)
    w_alpha = hk.get_parameter('w_alpha', [self.dim, self.dim], jnp.float32, init=w_init)

    mu = jnp.dot(x, w_mu*self.out_mask)
    alpha = jnp.dot(x, w_alpha*self.out_mask)
    alpha = jnp.tanh(alpha)

    return mu, alpha

################################################################################################################

class MAF(AutoBatchedLayer):

  def __init__(self, hidden_layer_sizes:Sequence[int], method: str='sequential', name: str="maf", **kwargs):
    super().__init__(name=name, **kwargs)
    self.hidden_layer_sizes = hidden_layer_sizes
    self.method = method

  def call(self, inputs: Mapping[str, jnp.ndarray], rng: jnp.ndarray=None, sample: Optional[bool]=False, **kwargs) -> Mapping[str, jnp.ndarray]:

    def initialize_input_sel(shape, dtype):
      dim = shape[-1]
      if self.method == 'random':
        rng = hk.next_rng_key()
        input_sel = random.randint(rng, shape=(dim,), minval=1, maxval=dim+1)
      else:
        input_sel = jnp.arange(1, dim + 1)
      return input_sel

    dim = inputs["x"].shape[-1]
    input_sel = hk.get_state("input_sel", (dim,), jnp.int32, init=initialize_input_sel)
    made = GaussianMADE(input_sel, dim, self.hidden_layer_sizes, self.method)

    if sample == False:
      x = inputs["x"]
      mu, alpha = made(x)
      z = (x - mu)*jnp.exp(-alpha)
      log_det = -alpha.sum(axis=-1)
      outputs = {"x": z, "log_det": log_det}
    else:
      z = inputs["x"]
      x = jnp.zeros_like(z)
      u = z

      # We need to build output a dimension at a time
      def carry_body(carry, inputs):
        x, idx = carry, inputs
        mu, alpha = made(x)
        w = mu + u*jnp.exp(alpha)
        x = jax.ops.index_update(x, idx, w[idx])
        return x, alpha[idx]

      indices = jnp.nonzero(input_sel == (1 + jnp.arange(x.shape[0])[:,None]))[1]
      x, alpha_diag = jax.lax.scan(carry_body, x, indices)
      log_det = -alpha_diag.sum(axis=-1)
      outputs = {"x": x, "log_det": log_det}

    return outputs
