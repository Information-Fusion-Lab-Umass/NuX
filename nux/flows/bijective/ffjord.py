import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
import nux.util as util
import einops
from nux.flows.base import Flow
import diffrax

__all__ = ["FFJORD"]

class FFJORD(Flow):

  def __init__(self, vf):
    self.vf = vf

  def get_params(self):
    return self.vf.get_params()

  def __call__(self, x, params=None, rng_key=None, inverse=False, no_llc=False, log_px_estimate=False, save_times=None, t0=0.0, t1=1.0, **kwargs):
    compute_log_likelihood = not no_llc

    if params is None:
      t = random.uniform(rng_key, minval=0, maxval=1, shape=(x.shape[:1]))
      self.vf(x, t=t, params=None, rng_key=rng_key)
      params = self.get_params()

    k1, k2 = random.split(rng_key, 2)

    if log_px_estimate:
      v = random.normal(k1, x.shape)

    def f(t, x_and_logpx, args):
      x, log_px, _ = x_and_logpx

      if inverse == False:
        t = 1.0 - t

      def compute_updates(x):

        def apply_flow(x):
          return self.vf(x, t, params=params, rng_key=k2, **kwargs)

        if compute_log_likelihood and False:

          if log_px_estimate:
            dxdt, dudxv = jax.jvp(apply_flow, (x,), (v,))
            sum_axes = util.last_axes(x.shape[len(x.shape[:1]):])
            dlogpxdt = -jnp.sum(dudxv*v, axis=sum_axes)
          else:
            assert x.ndim == 2 # TODO
            eye = jnp.eye(x.shape[-1])
            def jvp(dx):
              dx = jnp.broadcast_to(dx, x.shape)
              dxdt, d2dx_dtdx = jax.jvp(apply_flow, (x,), (dx,))
              return dxdt, d2dx_dtdx

            dxdt, d2dx_dtdx = jax.vmap(jvp)(eye)
            dxdt = dxdt[0]
            dlogpxdt = -jnp.einsum("ibi->b", d2dx_dtdx)
        else:
          dxdt = apply_flow(x)
          dlogpxdt = jnp.zeros_like(log_px)

        return dlogpxdt, dxdt

      dlogpxdt, dxdt = compute_updates(x)
      dgradlogpxdt = jnp.zeros_like(dxdt)

      if inverse == False:
        dxdt = -dxdt
      return dxdt, dlogpxdt, dgradlogpxdt

    term = diffrax.ODETerm(f)
    solver = diffrax.Dopri5()
    if save_times is None:
      saveat = diffrax.SaveAt(ts=[t1])
    else:
      saveat = diffrax.SaveAt(ts=save_times)

    solution = diffrax.diffeqsolve(term, solver, saveat=saveat, t0=t0, t1=t1, dt0=0.01, y0=(x, jnp.zeros(x.shape[:1]), jnp.zeros(x.shape)))
    outs = solution.ys

    if save_times is None:
      outs = jax.tree_util.tree_map(lambda x: x[0], outs)

    return outs[:-1]
