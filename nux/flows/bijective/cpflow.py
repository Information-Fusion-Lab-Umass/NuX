import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import nux.util as util
from nux.nn.convex import InputConvexNN
from jax.scipy import optimize

__all__ = ["CPFlow"]

################################################################################################################

class CPFlow():

  def __init__(self, hidden_dim, aug_dim, n_hidden_layers, lanczos_quad=False):
    self.lanczos_quad = lanczos_quad
    self.F = InputConvexNN(hidden_dim, aug_dim, n_hidden_layers)

  def get_params(self):
    return self.F.get_params()

  def __call__(self, x, params=None, inverse=False, rng_key=None, no_llc=False, exact=False, **kwargs):
    if inverse:
      no_llc = True

    k1, k2 = random.split(rng_key, 2)

    sum_axes = util.last_axes(x.shape[1:])
    self.vdot = lambda x, y: jnp.sum(x*y, axis=sum_axes)

    self.params = params
    if params is None:
      self.F(x, params=None, rng_key=k1)
      self.params = self.F.get_params()

    def unbatched_potential(x):
      return self.F(x[None], params=self.params, rng_key=k1).sum()

    self.f = jax.vmap(jax.grad(unbatched_potential))

    if inverse == False:
      z = self.f(x)
    else:
      def invert_unbatched(x):
        def closure(z):
          return unbatched_potential(z) - jnp.vdot(z, x)
        z = optimize.minimize(closure, x, method="l-bfgs-experimental-do-not-rely-on-this") # This is slow
        # z = optimize.minimize(closure, x, method="BFGS") # This is slow
        return z
      z = jax.vmap(invert_unbatched)(x).x

    if no_llc == False:

      if exact:
        if inverse == False:
          hessian = jax.vmap(jax.hessian(unbatched_potential))(x)
        else:
          hessian = jax.vmap(jax.hessian(unbatched_potential))(z)
        log_det = jnp.linalg.slogdet(hessian)[1]
      else:

        @jax.vmap
        def hvp(x, p):
          def unbatched_hvp(p):
            return jax.jvp(jax.grad(unbatched_potential), (x,), (p,))[1]
          return unbatched_hvp(p)

        if inverse == False:
          self.H = partial(hvp, x)
        else:
          self.H = partial(hvp, z)

        def llc_estimate(rng_key):
          v = random.normal(rng_key, x.shape)

          # Solve v^TH^{-1} and compute a log det esimate or lower bound
          if self.lanczos_quad:
            Hinv_v, log_det = util.cg_and_lanczos_quad(self.H, v, debug=False)
          else:
            cg_result = util.conjugate_gradient(self.H, v, debug=False)
            Hinv_v = cg_result.x
            total_dim = util.list_prod(v.shape[1:])
            log_det = total_dim - self.vdot(v, Hinv_v)

          Hinv_v = jax.lax.stop_gradient(Hinv_v)
          log_det = jax.lax.stop_gradient(log_det)

          # Compute the surrogate objective
          surrogate = self.vdot(Hinv_v, self.H(v))

          # Return a dummy value to display and optimize
          llc = log_det + util.only_gradient(surrogate)
          return llc

        log_det = llc_estimate(k2)
        # keys = random.split(k2, 10240)
        # log_det = jax.vmap(llc_estimate)(keys).mean(axis=0)

    else:
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
  n_hidden_layers = 20
  flow = CPFlow(hidden_dim, aug_dim, n_hidden_layers, lanczos_quad=True)

  z, log_det = flow(x, rng_key=rng_key)
  params = flow.get_params()
  reconstr, _ = flow(z, params, inverse=True, rng_key=rng_key, no_llc=True)

  import pdb; pdb.set_trace()
