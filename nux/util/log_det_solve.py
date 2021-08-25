import jax
from jax import random
import jax.numpy as jnp
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
from collections import namedtuple
from .misc import last_axes, broadcast_to_first_axis, list_prod, only_gradient
from .iterative import conjugate_gradient, cg_and_lanczos_quad

class LogDetSolve():

  def __init__(self, flow, lanczos_quad=False):
    self.flow = flow
    self.lanczos_quad = lanczos_quad

  def get_params(self):
    return self.flow.get_params()

  def __call__(self, x, params=None, inverse=False, rng_key=None, no_llc=False, exact=False, **kwargs):
    if inverse:
      no_llc = True

    k1, k2 = random.split(rng_key, 2)

    sum_axes = last_axes(x.shape[1:])
    self.vdot = lambda x, y: jnp.sum(x*y, axis=sum_axes)

    def apply_fun(x):
      z, _ = self.flow(x, params=params, inverse=inverse, rng_key=k1, no_llc=True, **kwargs)
      return z

    if params is None:
      apply_fun(x)
      params = self.get_params()

    if no_llc == False:

      z, _vjp = jax.vjp(apply_fun, x); vjp = lambda t: _vjp(t)[0]
      jvp = lambda t: jax.jvp(apply_fun, (x,), (t,))[1]
      self.A = lambda t: jvp(vjp(t))

      # Solve for an estimate/lower bound of the log det and for an unbiased
      # estimate of its gradients
      def llc_estimate(rng_key):
        v = random.normal(rng_key, x.shape)

        # Solve v^TH^{-1} and compute a log det esimate or lower bound
        if self.lanczos_quad:
          Hinv_v, log_det = cg_and_lanczos_quad(self.A, v, debug=False)
        else:
          cg_result = conjugate_gradient(self.A, v, debug=False)
          Hinv_v = cg_result.x
          total_dim = list_prod(v.shape[1:])
          log_det = total_dim - self.vdot(v, Hinv_v)

        Hinv_v = jax.lax.stop_gradient(Hinv_v)
        log_det = jax.lax.stop_gradient(log_det)

        # Compute the surrogate objective
        surrogate = self.vdot(Hinv_v, self.A(v))
        # assert jnp.allclose(surrogate, self.vdot(v, v))

        # Return a dummy value to display and optimize
        llc = log_det + only_gradient(surrogate)
        return 0.5*llc

      log_det = llc_estimate(k2)
      # keys = random.split(k2, 10240)
      # log_dets = jax.vmap(llc_estimate)(keys)
      # log_det = log_dets.mean(axis=0)

      if False:
        # Compare against the true log-det
        total_dim = list_prod(x.shape[1:])
        G = jax.vmap(jax.jacobian(lambda x: apply_fun(x[None])[0]))(x)
        G_flat = G.reshape(x.shape[:1] + (total_dim, total_dim))
        log_det_true = jnp.linalg.slogdet(G_flat)[1]
        import pdb; pdb.set_trace()

    else:
      z = apply_fun(x)
      log_det = jnp.zeros(z.shape[:1])

    return z, log_det

################################################################################################################

if __name__ == "__main__":
  from debug import *
  import nux

  rng_key = random.PRNGKey(1)
  dim = 3
  batch_size = 8
  x = random.normal(rng_key, (batch_size, dim))

  hidden_dim = 32
  aug_dim = 32
  n_hidden_layers = 4
  flow = nux.Repeat(nux.CPFlow(hidden_dim, aug_dim, n_hidden_layers), 5)

  # hidden_dim = 32
  # n_resnet_layers = 3
  # n_layers = 4
  # flow = nux.ResidualFlowModel(hidden_dim, n_resnet_layers, n_layers)

  flow = LogDetSolve(flow, lanczos_quad=False)

  z, log_det = flow(x, rng_key=rng_key)
  params = flow.get_params()
  import pdb; pdb.set_trace()

  reconstr, _ = flow(z, params, inverse=True, rng_key=rng_key, no_llc=True)

  import pdb; pdb.set_trace()
