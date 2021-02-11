import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from functools import partial
from jax.flatten_util import ravel_pytree
import haiku as hk

""" Taken from https://github.com/google/jax/blob/4a20eea8285d6396b50451ed884c0fe00e382821/docs/notebooks/Custom_derivative_rules_for_Python_code.ipynb
    and refactored to match http://www.autodiff.org/Docs/euroad/Second%20EuroAd%20Workshop%20-%20Sebastian%20Schlenkrich%20-%20Differentianting%20Fixed%20Point%20Iterations%20with%20ADOL-C.pdf"""

__all__ = ["fixed_point",
           "fixed_point_scan",
           "fixed_point_while",
           "fixed_point_fun"]

def fixed_point_scan(f, x_init, max_iters, relaxed=False):

  def body_fun(carry, inputs):
    _, x = carry
    fx = f(x)
    if relaxed:
      fx = 0.5*fx + 0.5*x
    carry = (x, fx)
    return carry, ()

  (_, x), _ = jax.lax.scan(body_fun, (x_init, f(x_init)), jnp.arange(max_iters))
  return x, max_iters

def fixed_point_while(f, x_init, max_iters):
  atol = 1e-5
  relaxed = True

  def cond_fun(val):
    x_prev, x, i = val
    max_iters_reached = jnp.where(i >= max_iters, True, False)

    flat_x_prev = ravel_pytree(x_prev)[0]
    flat_x = ravel_pytree(x)[0]
    tolerance_achieved = jnp.allclose(flat_x_prev, flat_x, atol=atol)
    return ~(max_iters_reached | tolerance_achieved)

  if relaxed == False:
    def body_fun(val):
      _, x, i = val
      fx = f(x)
      return x, fx, i + 1
  else:
    def body_fun(val):
      _, x, i = val
      fx = jax.tree_multimap(lambda x, y: 0.5*x + 0.5*y, x, f(x))
      return x, fx, i + 1

  _, x, N = jax.lax.while_loop(cond_fun, body_fun, (x_init, f(x_init), 0.0))
  return x, N

@partial(jax.custom_vjp, nondiff_argnums=(0,))
def fixed_point(f, u, x_init, max_iters, *nondiff_args):

  def fixed_point_iter(x):
    return f(u, x, *nondiff_args)

  x, N = fixed_point_while(fixed_point_iter, x_init, max_iters)
  return x

def fixed_point_fwd(f, u, x_init, max_iters, *nondiff_args):
  x = fixed_point(f, u, x_init, max_iters, *nondiff_args)
  return x, (u, x, max_iters, *nondiff_args)

def fixed_point_rev(f, ctx, dLdx):
  u, x, max_iters, *nondiff_args = ctx

  max_iters = 5000

  _, vjp_x = jax.vjp(lambda x: f(u, x, *nondiff_args), x)

  def rev_iter(zeta):
    zetaT_dFdx, = vjp_x(zeta)
    return jax.tree_multimap(lambda x, y: x + y, dLdx, zetaT_dFdx)

  zeta, N = fixed_point_while(rev_iter, dLdx, max_iters)

  _, vjp_u = jax.vjp(lambda u: f(u, x, *nondiff_args), u)
  dLdu, = vjp_u(zeta)

  if len(nondiff_args) == 0:
    return dLdu, None, None

  return dLdu, None, None, (None,)*len(nondiff_args)

# # Use this if we want second derivatives.
# def fixed_point_rev(f, ctx, dLdx):
#   u, x, max_iters, *nondiff_args = ctx

#   max_iters = 20

#   def rev_iter(f, packed, zeta):
#     ctx, dLdx = packed
#     u, x, max_iters, *nondiff_args = ctx

#     _, vjp_x = jax.vjp(lambda x: f(u, x, *nondiff_args), x)
#     zetaT_dFdx, = vjp_x(zeta)
#     return jax.tree_multimap(lambda x, y: x + y, dLdx, zetaT_dFdx)

#   packed = (ctx, dLdx)
#   zeta = fixed_point(partial(rev_iter, f), packed, dLdx, max_iters)

#   _, vjp_u = jax.vjp(lambda u: f(u, x, *nondiff_args), u)
#   dLdu, = vjp_u(zeta)

#   if len(nondiff_args) == 0:
#     return dLdu, None, None

#   return dLdu, None, None, (None,)*len(nondiff_args)

fixed_point.defvjp(fixed_point_fwd, fixed_point_rev)

################################################################################################################

@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def fixed_point_fun(phi, F, n_iters, x, zeta, theta):
  # y = phi(x,theta) where x is the fixed point of F(x,theta).
  for i in range(n_iters):
    x, vjp = jax.vjp(lambda x: F(x, theta), x)
    y, xbar = jax.value_and_grad(phi)(x, theta)
    dx, = vjp(zeta)
    zeta = jax.tree_multimap(jnp.add, xbar, dx)
  return y, x, zeta

def fixed_point_fun_fwd(phi, F, n_iters, x, zeta, theta):
  for i in range(n_iters):
    x, vjp = jax.vjp(F, x, theta)
    y, (xbar, thetabar) = jax.value_and_grad(phi, argnums=(0, 1))(x, theta)

    xhat, thetahat = vjp(zeta)
    zeta, thetabar = jax.tree_multimap(jnp.add, (xbar, thetabar), (xhat, thetahat))

  dydx = jax.tree_multimap(jnp.add, xbar, zeta)
  dydx = jax.tree_multimap(jnp.subtract, dydx, xhat)

  ctx = (xbar, thetabar, dydx)
  return (y, x, zeta), ctx

def fixed_point_fun_bwd(phi, F, n_iters, ctx, g):
  dy, dx, dzeta = g
  xbar, thetabar, dydx = ctx

  dtheta = jax.tree_multimap(jnp.multiply, dy, thetabar)
  dx = jax.tree_map(lambda x: dy*x, dydx)
  return dx, None, dtheta

fixed_point_fun.defvjp(fixed_point_fun_fwd, fixed_point_fun_bwd)
