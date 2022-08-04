import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import nux.util as util
import einops
from .layers import WeightNormDense, WeightNormConv

class InputConvexNN():
  def __init__(self, hidden_dim, aug_dim, n_hidden_layers, image=False):
    self.hidden_dim = hidden_dim
    self.aug_dim = aug_dim
    self.total_dim = self.hidden_dim + self.aug_dim
    self.n_hidden_layers = n_hidden_layers
    self.s = partial(util.square_plus, gamma=1.0)
    self.mvp = lambda w, x: jnp.einsum("ij,bj->bi", w, x)

    self.image = image
    if self.image:
      self.affine = partial(WeightNormConv, (3, 3))
    else:
      self.affine = WeightNormDense

  def get_params(self):
    return dict(L_in=self.l_in.get_params(),
                Laug_in=self.laug_in.get_params(),
                Lp=self.Lp,
                L=self.L,
                Laug=self.Laug,
                Lp_out=self.lp_out.get_params(),
                L_out=self.l_out.get_params(),
                w0=self.w0,
                w1=self.w1)

  def __call__(self, x, params=None, aux=None, rng_key=None, is_training=True):
    # If aux is passed in, would need a partially convex neural network.
    k1, k2, k3, k4, k5 = random.split(rng_key, 5)

    if params is None:
      self.L_in = None
      self.Laug_in = None
      self.Lp = None
      self.L = None
      self.Laug = None
      self.Lp_out = None
      self.L_out = None
      self.w0, self.w1 = jnp.zeros((2,))
    else:
      self.L_in = params["L_in"]
      self.Laug_in = params["Laug_in"]
      self.Lp = params["Lp"]
      self.L = params["L"]
      self.Laug = params["Laug"]
      self.Lp_out = params["Lp_out"]
      self.L_out = params["L_out"]
      self.w0, self.w1 = params["w0"], params["w1"]

    # First layer to project to the right dim
    self.l_in = self.affine(self.hidden_dim, before_square_plus=True)
    self.laug_in = self.affine(self.aug_dim, before_square_plus=True)
    ht = self.l_in(x, params=self.L_in, rng_key=k1)
    ha = self.laug_in(x, params=self.Laug_in, rng_key=k2)
    h = self.s(jnp.concatenate([ht, ha], axis=-1))

    self.lp = self.affine(self.hidden_dim, positive=True, before_square_plus=True)
    self.l = self.affine(self.hidden_dim, before_square_plus=True)
    self.laug = self.affine(self.aug_dim, before_square_plus=True)
    def scan_block(carry, inputs):
      h, x = carry
      Lp, L, Laug, (k1, k2, k3) = inputs

      ht = self.lp(h, params=Lp, rng_key=k1) + self.l(x, params=L, rng_key=k2)
      ha = self.laug(x, params=Laug, rng_key=k3)
      h = self.s(jnp.concatenate([ht, ha], axis=-1))
      if Lp is None:
        return (h, x), (self.lp.get_params(), self.l.get_params(), self.laug.get_params())
      return (h, x), ()

    keys = random.split(k3, 3*self.n_hidden_layers)
    keys = keys.reshape((self.n_hidden_layers, 3, -1))
    if params is None:
      weights = []
      for i in range(self.n_hidden_layers):
        (h, x), w = scan_block((h, x), (None, None, None, keys[i]))
        weights.append(w)
      self.Lp, self.L, self.Laug = jax.tree_util.tree_map(lambda *xs: jnp.array(xs), *zip(weights))[0]

    else:
      (h, _), _ = jax.lax.scan(scan_block, (h, x), (self.Lp, self.L, self.Laug, keys), unroll=1)

    self.lp_out = self.affine(1, positive=True)
    self.l_out = self.affine(1)
    F = self.lp_out(h, params=self.Lp_out, rng_key=k4)
    F += self.l_out(x, params=self.L_out, rng_key=k5)

    # Avg. pool
    if self.image:
      sum_axes = util.last_axes(F.shape[1:])
      F = jnp.mean(F, axis=sum_axes)[...,None]
    else:
      sum_axes = -1

    # Should be strictly convex
    half_norm_sq = jnp.sum(x**2, axis=sum_axes)/2
    F = self.s(self.w0)*half_norm_sq + self.s(self.w1)*F
    return F

if __name__ == "__main__":
  from debug import *

  rng_key = random.PRNGKey(1)
  dim = 4
  batch_size = 16
  # x = random.normal(rng_key, (batch_size, dim))
  x = random.normal(rng_key, (batch_size, 8, 8, 2))

  hidden_dim = 2
  aug_dim = 1
  n_hidden_layers = 4
  net = InputConvexNN(hidden_dim, aug_dim, n_hidden_layers, image=True)

  z = net(x, rng_key=rng_key)
  params = net.get_params()

  x = random.normal(rng_key, (100, *x.shape[1:]))
  z = net(x, params=params, rng_key=rng_key)

  # Ensure that the network has a psd hessian
  def F(x):
    z = net(x[None], params=params, rng_key=rng_key)[0]
    return z.sum(axis=-1)

  H = jax.vmap(jax.hessian(F))(x)
  total_dim = util.list_prod(x.shape[1:])
  H = H.reshape((-1, total_dim, total_dim))
  import pdb; pdb.set_trace()
  s = jnp.linalg.svd(H, compute_uv=False)
  assert jnp.all(s > 0)

  import pdb; pdb.set_trace()
