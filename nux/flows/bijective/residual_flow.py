import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import nux.util as util
from nux.flows.base import Flow
from fax import implicit

__all__ = ["ResidualFlow"]

################################################################################################################

class ResidualFlow(Flow):
  """ https://arxiv.org/pdf/1906.02735.pdf """

  def __init__(self, res_block, n_exact=4, n_total=8):
    self.res_block = res_block
    self.n_exact   = n_exact
    self.n_total   = n_total

  def get_params(self):
    return self.res_block.get_params()

  def log_det_and_surrogate(self, rng_key, v, vjp):

    def scan_body(carry, inputs):
      k = inputs
      w = carry
      w = vjp(w)
      term = self.vdot(w, v)

      log_det_term = term/k
      grad_term = -w

      w *= -1
      return w, (log_det_term, grad_term)

    k = jnp.arange(1, self.n_total + 1)

    w, (log_det_terms, grad_terms) = jax.lax.scan(scan_body, v, k, unroll=1)

    roulette_coeff = util.geometric_roulette_coefficients(rng_key, self.n_total - self.n_exact)
    roulette_coeff = jnp.hstack([jnp.ones(self.n_exact), roulette_coeff])

    log_det_est = (log_det_terms*util.broadcast_to_first_axis(roulette_coeff, log_det_terms.ndim)).sum(axis=0)
    log_det_est = jax.lax.stop_gradient(log_det_est)

    b = (grad_terms*util.broadcast_to_first_axis(roulette_coeff, grad_terms.ndim)).sum(axis=0)
    b += v # k=0 term
    b = jax.lax.stop_gradient(b)

    surrogate_objective = self.vdot(vjp(b) + b, v)

    return log_det_est, surrogate_objective

  def __call__(self, x, params=None, inverse=False, rng_key=None, no_llc=False, _test_resflow=False, **kwargs):
    self.params = params

    sum_axes = util.last_axes(x.shape[1:])
    self.vdot = lambda x, y: jnp.sum(x*y, axis=sum_axes)

    def apply_fun(x, **kwargs):
      if "sv_update" not in kwargs:
        sv_update = False if inverse == True else True
      else:
        sv_update = kwargs.pop("sv_update")
      x = self.res_block(x, params=self.params, rng_key=rng_key, sv_update=sv_update, **kwargs)
      return x

    if self.params is None:
      apply_fun(x)
      self.params = self.get_params()

    if inverse == False:
      if no_llc == True:
        z = x + apply_fun(x)
      else:
        gx, _vjp = jax.vjp(apply_fun, x); vjp = lambda x: _vjp(x)[0]
        z = x + gx
    else:

      z0 = x
      def fp(inputs):
        x, params, rng_key = inputs
        def _fp(z):
          gz = self.res_block(z, params=params, rng_key=rng_key, sv_update=False, **kwargs)
          return x - gz
        return _fp

      z = implicit.two_phase_solve(fp, z0, (x, self.params, rng_key))

      if no_llc == False:
        _, _vjp = jax.vjp(apply_fun, z); vjp = lambda x: _vjp(x)[0]

    if _test_resflow:
      self.test(x, params, rng_key)

    if no_llc:
      return z, jnp.zeros(z.shape[0])

    k1, k2 = random.split(rng_key, 2)
    v = random.normal(k1, x.shape)
    log_det, surrogate = self.log_det_and_surrogate(k2, v, vjp)
    res_log_det = jax.lax.stop_gradient(log_det) + util.only_gradient(surrogate)

    return z, res_log_det

  def test(self, x, params, rng_key):
    def apply_fun(x):
      x = self.res_block(x, params=params, rng_key=rng_key)
      return x

    gx, _vjp = jax.vjp(apply_fun, x); vjp = lambda x: _vjp(x)[0]
    z = x + gx

    # Check that the function is invertible
    x_reconstr, _ = self(z, params=params, rng_key=rng_key, inverse=True, no_llc=True)

    # Check that the surrogate objective is correct
    k1, k2 = random.split(rng_key, 2)
    v = random.normal(k1, x.shape)
    self.n_exact, self.n_total = 50, 50
    log_det, surrogate = self.log_det_and_surrogate(k2, v, vjp)

    # Check that the function is Lipschitz continuous with a max val of 1
    J = jax.vmap(jax.jacobian(lambda x: apply_fun(x[None])[0]))(x)
    total_dim = util.list_prod(x.shape[1:])
    J = J.reshape(x.shape[:1] + (total_dim, total_dim))
    l1_norm = jax.vmap(partial(jnp.linalg.norm, ord=1))(J)
    linf_norm = jax.vmap(partial(jnp.linalg.norm, ord=jnp.inf))(J)
    l2_norm = jax.vmap(partial(jnp.linalg.norm, ord=2))(J)

    assert jnp.all(l1_norm < 1.0)
    assert jnp.all(linf_norm < 1.0)
    assert jnp.all(l2_norm < 1.0)
    assert jnp.allclose(surrogate, self.vdot(v, v))
    assert jnp.allclose(x, x_reconstr, atol=1e-5)

################################################################################################################

if __name__ == "__main__":
  from debug import *
  import nux

  rng_key = random.PRNGKey(1)
  x_shape = (16, 4, 4, 3)
  # x_shape = (7, 3)
  x = random.normal(rng_key, x_shape)

  filter_shape   = (1, 1)
  hidden_channel = 16
  dropout_prob   = 0.2
  n_layers       = 1
  res_block = nux.LipschitzConvResBlock(filter_shape,
                                        hidden_channel,
                                        n_layers,
                                        dropout_prob)

  # res_block = nux.LipschitzDenseResBlock(hidden_channel,
  #                                            n_layers,
  #                                            dropout_prob)

  flow = ResidualFlow(res_block)

  z, log_det = flow(x, rng_key=rng_key)
  params = flow.get_params()

  x_reconstr, log_det2 = flow(z, params=params, rng_key=rng_key, inverse=True)

  import pdb; pdb.set_trace()
