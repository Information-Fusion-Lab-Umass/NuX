import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from functools import partial
from typing import Optional, Mapping, Callable, Sequence
from nux.internal.base import CustomFrame
from haiku._src.typing import PRNGKey
import haiku._src.base as hk_base
from nux.flows.bijective.residual_flows.power_series import unbiased_neumann_vjp_terms

def _fixed_point(f, x_init, max_iters, atol):
  # http://www.autodiff.org/Docs/euroad/Second%20EuroAd%20Workshop%20-%20Sebastian%20Schlenkrich%20-%20Differentianting%20Fixed%20Point%20Iterations%20with%20ADOL-C.pdf

  def cond_fun(val):
    x_prev, x, i = val
    max_iters_reached = jnp.where(i >= max_iters, True, False)
    tolerance_achieved = jnp.allclose(x_prev - x, 0.0, atol=atol)
    return ~(max_iters_reached | tolerance_achieved)

  def body_fun(val):
    _, x, i = val
    fx = f(x)
    return x, fx, i + 1

  _, x, N = jax.lax.while_loop(cond_fun, body_fun, (x_init, f(x_init), 0.0))
  return x, N

def contractive_fixed_point(apply_fun, params, state, x_current, z):
  # Step of fixed point iteration
  gx, state = apply_fun(params, state, x_current)
  # return z - gx
  return 0.5*(z - gx) + 0.5*x_current

def steffensen_fixed_point(apply_fun, params, state, xn, z):
  def f(x):
    gx, _ = apply_fun(params, state, x)
    return z - gx

  fx = f(xn)
  ffx = f(fx)

  dxn  = fx - xn
  dgxn = ffx - fx
  d2xn = dgxn - dxn

  if xn.ndim == 1:
    xnp1 = ffx - jnp.sum(dgxn*d2xn)/jnp.sum(d2xn**2)*dgxn
  else:
    xnp1 = ffx - jnp.sum(dgxn*d2xn, axis=-1)[:,None]/jnp.sum(d2xn**2, axis=-1)[:,None]*dgxn
  return xnp1

@partial(jax.custom_vjp, nondiff_argnums=(0,))
def fixed_point(apply_fun, params, state, z, roulette_rng):
  # Invert a contractive function using fixed point iterations.

  def fixed_point_iter(x):
    return contractive_fixed_point(apply_fun, params, state, x, z)
    # return steffensen_fixed_point(apply_fun, params, state, x, z)

  x, N = _fixed_point(fixed_point_iter, z, 1000, 1e-7)
  return x

def fixed_point_fwd(apply_fun, params, state, z, roulette_rng):
  # We need to save frame_data and the fixed point solution for backprop
  x = fixed_point(apply_fun, params, state, z, roulette_rng)
  return x, (params, state, x, z, roulette_rng)

def fixed_point_bwd(apply_fun, ctx, dLdx):
  assert 0, "Bro, did you seriously just try to backprop through a fixed point iteration?"
  params, state, x, z, roulette_rng = ctx

  with hk_base.frame_stack(CustomFrame.create_from_params_and_state(params, state)):

    _, vjp_x = jax.vjp(lambda x: contractive_fixed_point(apply_fun, params, state, x, z), x)
    # _, vjp_x = jax.vjp(lambda x: steffensen_fixed_point(apply_fun, params, state, x, z), x)

    def rev_iter(zeta):
      zetaT_dFdx, = vjp_x(zeta)
      return dLdx + zetaT_dFdx

    zeta, N = _fixed_point(rev_iter, dLdx, 100, 1e-4)

    # Go from zeta to the gradient of the frame data
    _, vjp_u = jax.vjp(lambda params: contractive_fixed_point(apply_fun, params, state, x, z), params)
    # _, vjp_u = jax.vjp(lambda params: steffensen_fixed_point(apply_fun, params, state, x, z), params)
    dparams, = vjp_u(zeta)

    # Also handle the gradient wrt z here.  To do this, we need to solve (dx/dz)^{-1}dx.
    # Do this with vjps against terms in the neumann series for dx/dz
    _, vjp_x = jax.vjp(lambda x: apply_fun(params, state, x)[0], x, has_aux=False)
    terms = unbiased_neumann_vjp_terms(vjp_x, dLdx, roulette_rng, n_terms=40, n_exact=40)
    dx_star = terms.sum(axis=0)

    return dparams, None, dx_star, None

fixed_point.defvjp(fixed_point_fwd, fixed_point_bwd)
