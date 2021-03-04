import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap, jit
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Sequence
from nux.internal.layer import Layer
import nux.util as util
from jax.scipy.special import logsumexp
import nux.util.weight_initializers as init
import nux.networks as net

__all__ = ["MADE"]

class MADE(Layer):

  def __init__(self,
               input_sel,
               dim,
               hidden_layer_sizes,
               method="shuffled_sequential",
               nonlinearity="relu",
               parameter_norm="weight_norm",
               n_components=4,
               triangular_jacobian=False,
               name=None):
    super().__init__(name=name)
    self.method              = method
    self.parameter_norm      = parameter_norm
    self.n_components        = n_components
    self.triangular_jacobian = triangular_jacobian

    if nonlinearity == "relu":
      self.nonlinearity = jax.nn.relu
    elif nonlinearity == "leaky_relu":
      self.nonlinearity = partial(jax.nn.leaky_relu, negative_slope=0.1)
    elif nonlinearity == "tanh":
      self.nonlinearity = jnp.tanh
    elif nonlinearity == "sigmoid":
      self.nonlinearity = jax.nn.sigmoid
    elif nonlinearity == "swish":
      self.nonlinearity = jax.nn.swish(x)
    elif nonlinearity == "lipswish":
      self.nonlinearity = lambda x: jax.nn.swish(x)/1.1
    elif nonlinearity == "elu":
      self.nonlinearity = jax.nn.elu
    elif nonlinearity == "logistic_logit":
      self.nonlinearity = net.LogisticLogit(n_components=8)
    else:
      assert 0, "Invalid nonlinearity"

    # Store the dimensions for later
    self.dim = dim
    self.hidden_layer_sizes = list(hidden_layer_sizes)
    self.input_sel = input_sel
    self.w_init = hk.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="truncated_normal")

  def next_mask(self, prev_sel, size, rng):

    # Choose the degrees of the next layer
    max_connection = self.dim - 1 if self.triangular_jacobian == False else self.dim

    if self.method == "random":
      sel = random.randint(rng, shape=(size,), minval=min(jnp.min(sel), max_connection), maxval=dim)
    elif "sequential" in self.method:
      sel = jnp.arange(size)%max(1, max_connection) + min(1, max_connection)
      if self.method == "shuffled_sequential":
        sel = random.permutation(rng, sel)
    else:
      assert 0, "Invalid mask method"

    # Create the new mask
    mask = (prev_sel[:,None] <= sel).astype(jnp.int32)
    return mask, sel

  def gen_masks(self, input_sel, layer_sizes, rng):
    rngs = random.split(rng, len(layer_sizes))

    self.masks = []
    self.sels = [input_sel]

    prev_sel = input_sel
    for size, rng in zip(layer_sizes, rngs):
      mask, prev_sel = self.next_mask(prev_sel, size, rng)
      self.masks.append(mask)
      self.sels.append(prev_sel)

    if self.triangular_jacobian == False:
      self.out_mask = prev_sel[:,None] < input_sel
    else:
      self.out_mask = prev_sel[:,None] <= input_sel

  def get_params(self, i, x, output_size):
    w_init = hk.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="truncated_normal")

    # Pass a singly batched input to the parameter functions.
    # Don't use autobatching here because we might end up reducing
    x, reshape = self.make_singly_batched(x)

    if self.parameter_norm == "weight_norm":
      w, b = init.weight_with_weight_norm(x=x,
                                          out_dim=output_size,
                                          name_suffix=str(i),
                                          w_init=self.w_init,
                                          b_init=jnp.zeros,
                                          is_training=True,
                                          use_bias=True)
    elif self.parameter_norm == "spectral_norm":
      w, b = init.weight_with_spectral_norm(x=x,
                                            out_dim=output_size,
                                            name_suffix=str(i),
                                            w_init=self.w_init,
                                            b_init=jnp.zeros,
                                            is_training=True,
                                            use_bias=True)
    else:
      w = hk.get_parameter(f"w_{i}", (output_size, x.shape[-1]), x.dtype, init=self.w_init)
      b = hk.get_parameter(f"b_{i}", (output_size,), init=jnp.zeros)

    # x = reshape(x)

    return w.T, b

  def call(self,
           inputs,
           rng=None,
           is_training=True,
           update_params=True,
           **kwargs):

    if self.triangular_jacobian:
      dx = jnp.ones_like(inputs)

    # Main autoregressive transform
    x = inputs["x"]
    layer_sizes = [self.dim] + self.hidden_layer_sizes + [self.dim]
    self.gen_masks(self.input_sel, layer_sizes[1:], rng)

    prev_sel = self.sels[0]
    for i, (mask, sel, input_size, output_size) in enumerate(zip(self.masks, \
                                                                 self.sels[1:], \
                                                                 layer_sizes[:-1], \
                                                                 layer_sizes[1:])):
      w, b = self.get_params(i, x, output_size)

      w_masked = w*mask
      x = jnp.dot(x, w_masked) + b

      if self.triangular_jacobian:
        nonlinearity_grad = jax.grad(self.nonlinearity)
        for i in range(x.ndim):
          nonlinearity_grad = vmap(nonlinearity_grad)

        diag_mask = prev_sel[:,None] == sel
        dx = jnp.dot(dx, (w*diag_mask))

        if i < len(self.masks) - 1:
          dx *= nonlinearity_grad(x)

      if i < len(self.masks) - 1:
        x = self.nonlinearity(x)

      prev_sel = sel

    w_mu = hk.get_parameter("w_mu", [self.dim, self.dim], x.dtype, init=self.w_init)
    mu = jnp.dot(x, w_mu*self.out_mask)

    if self.triangular_jacobian:
      diag_mask = prev_sel[:,None] == self.input_sel
      dmu = jnp.dot(dx, (w_mu*diag_mask))
      return mu, dmu

    w_alpha = hk.get_parameter("w_alpha", [self.dim, self.dim], x.dtype, init=self.w_init)
    alpha = jnp.dot(x, w_alpha*self.out_mask)
    alpha_bounded = jnp.tanh(alpha)

    return {"mu": mu, "alpha": alpha_bounded}
