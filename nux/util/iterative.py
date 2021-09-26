import jax
from jax import random
import jax.numpy as jnp
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
from collections import namedtuple

__all__ = ["tridiag_solve",
           "tridiag_rayleigh_quotient_iteration",
           "tridiag_eigenvector_from_eigenvalues",
           "bisection",
           "conjugate_gradient",
           "psd_implicit_mat_logdet_surrogate",
           "cg_and_lanczos_quad",
           "weighted_jacobi",
           "newtons_with_grad",
           "newtons"]


################################################################################################################

def bisection_body(f, val):
  x, current_x, current_z, lower, upper, dx, i = val

  gt = current_x > x
  lt = 1.0 - gt

  new_z = gt*0.5*(current_z + lower) + lt*0.5*(current_z + upper)
  lower = gt*lower                   + lt*current_z
  upper = gt*current_z               + lt*upper

  current_z = new_z
  current_x = f(current_z)
  dx = current_x - x

  return x, current_x, current_z, lower, upper, dx, i + 1

def bisection(f, lower, upper, x, atol=1e-8, max_iters=20000):
  # Compute f^{-1}(x) using the bisection method.  f must be monotonic.
  z = jnp.zeros_like(x)

  def cond_fun(val):
    x, current_x, current_z, lower, upper, dx, i = val

    max_iters_reached = jnp.where(i > max_iters, True, False)
    tolerance_achieved = jnp.allclose(dx, 0.0, atol=atol)
    first_iter = jnp.where(i > 0.0, False, True)
    return ~(max_iters_reached | tolerance_achieved) | first_iter

  val = (x, f(z), z, lower, upper, jnp.ones_like(x)*10.0, 0.0)
  val = jax.lax.while_loop(cond_fun, partial(bisection_body, f), val)
  x, current_x, current_z, lower, upper, dx, i = val
  return current_z

################################################################################################################

from .misc import last_axes, broadcast_to_first_axis, list_prod

cg_result = namedtuple("cg_result", ["x", "rsq", "n_iters"])

def conjugate_gradient(A, b, debug=False, max_iters=1000, tol=1e-3):
  # Linear solve Ax = b
  sum_axes = last_axes(b.shape[1:])
  broadcast = lambda x: broadcast_to_first_axis(x, b.ndim)
  vdot = lambda x, y: jnp.sum(x*y, axis=sum_axes)

  x = jnp.zeros_like(b)
  r = b - A(x)
  p = r
  rsq = vdot(r, r)
  alpha = jnp.ones_like(rsq)
  beta = jnp.zeros_like(rsq)

  def safe_div(a, b):
    return jnp.where(jnp.abs(b) < 1e-7, 0.0, a/b)

  def cond(carry):
    x, r, p, rsq, i = carry
    max_iters_reached = jnp.where(i >= max_iters, True, False)
    tolerance_achieved = jnp.allclose(rsq, 0.0, atol=tol)
    first_iter = jnp.where(i > 0.0, False, True)
    return ~(max_iters_reached | tolerance_achieved) | first_iter

  def body(carry):
    x, r, p, rsq, i = carry

    Ap = A(p)
    alpha = safe_div(rsq, vdot(Ap, p))
    x = x + broadcast(alpha)*p

    r = r - broadcast(alpha)*Ap
    # r = b - A(x) # More numerically stable

    rsq_new = vdot(r, r)
    beta = safe_div(rsq_new, rsq)
    p = r + broadcast(beta)*p
    rsq = rsq_new

    carry = (x, r, p, rsq, i + 1)
    return carry

  carry = x, r, p, rsq, 0.0
  carry = jax.lax.while_loop(cond, body, carry)
  x, r, p, rsq, n_iters = carry

  if debug:
    import pdb; pdb.set_trace()
  return cg_result(x, rsq, n_iters)

from .misc import only_gradient
def psd_implicit_mat_logdet_surrogate(A, v, lower_bound=True):
  # Compute a cheap bound on \log|A| but an unbiased gradient estimate
  # v should be random vector to use in Hutchinson's trace estimator

  cg_result = conjugate_gradient(A, v)
  A_inv_v = jax.lax.stop_gradient(cg_result.x)

  sum_axes = last_axes(v.shape[1:])
  vdot = lambda x, y: jnp.sum(x*y, axis=sum_axes)

  # Compute an unbiased gradient estimate
  Av = A(v)
  surrogate = vdot(A_inv_v, Av)

  # Compute a bound
  total_dim = list_prod(v.shape[1:])
  if lower_bound:
    log_det = total_dim - vdot(v, A_inv_v)
  else:
    log_det = vdot(v, jax.lax.stop_gradient(Av)) - total_dim

  # Return a value to display that will optimize correctly
  llc = log_det + only_gradient(surrogate)
  return llc

################################################################################################################

def tridiag_solve(a, b, c, d):
  # a is lower band, b is diagonal, c is upper band
  # Assumes that a starts with 0 and c ends with 0
  zero = jnp.zeros(d.shape[1:])
  iszero = lambda x: jnp.abs(x) < 1e-7

  def body(carry, inputs):
    a, b, c, d = inputs
    c_prime, d_prime = carry

    denominator = (b - a*c_prime)
    c_prime = jnp.where(iszero(denominator), zero, c/denominator)
    d_prime = jnp.where(iszero(denominator), zero, (d - a*d_prime)/denominator)
    return (c_prime, d_prime), (c_prime, d_prime)

  _, (c_prime, d_prime) = jax.lax.scan(body, (zero, zero), (a, b, c, d))

  def body(carry, inputs):
    c_prime, d_prime = inputs
    x = carry
    x = d_prime - c_prime*x
    return x, x

  _, x = jax.lax.scan(body, zero, (c_prime, d_prime), reverse=True)
  return x

from .misc import last_axes, broadcast_to_first_axis

def tridiag_rayleigh_quotient_iteration(a, b, c, mu, tol=1e-8, max_iters=30):
  x0 = b
  b_prime = b - mu
  vdot = lambda x, y: jnp.sum(x*y, axis=last_axes(x0.shape))
  normalize = lambda y: y*broadcast_to_first_axis(jax.lax.rsqrt(vdot(y, y)), x0.ndim)

  def cond(carry):
    x, x_old, i = carry
    max_iters_reached = jnp.where(i >= max_iters, True, False)
    tolerance_achieved = jnp.allclose(x, x_old, atol=tol)
    first_iter = jnp.where(i > 0.0, False, True)
    return ~(max_iters_reached | tolerance_achieved) | first_iter

  def body(carry):
    x_old, _, i = carry
    x = tridiag_solve(a, b_prime, c, x_old)
    x = normalize(x)
    return (x, x_old, i + 1)

  (x, _, n_iters) = jax.lax.while_loop(cond, body, (x0, x0, 0.0))
  return x

def tridiag_eigenvector_from_eigenvalues(a, b, c, eigvals):
  vmapped_fun = jax.vmap(tridiag_rayleigh_quotient_iteration, in_axes=(None, None, None, 0))
  eigenvectors = vmapped_fun(a, b, c, eigvals)
  return eigenvectors.T # Keep same convention as if we used jnp.linalg.eigh

################################################################################################################

def cg_and_lanczos_quad(A, b, max_iters=-1, debug=False):
  # Linear solve Ax = b
  sum_axes = last_axes(b.shape[1:])
  broadcast = lambda x: broadcast_to_first_axis(x, b.ndim)
  vdot = lambda x, y: jnp.sum(x*y, axis=sum_axes)
  if max_iters == -1:
    max_iters = list_prod(b.shape[1:])

  x = jnp.zeros_like(b)
  r = b - A(x)
  p = r
  rsq = vdot(r, r)
  alpha = jnp.ones_like(rsq)
  beta = jnp.zeros_like(rsq)

  def safe_div(a, b):
    return jnp.where(jnp.abs(b) < 1e-7, 0.0, a/b)

  def scan_body(carry, inputs):
    x, r, p, rsq, alpha, beta = carry

    Ap = A(p)
    alpha_old = alpha
    alpha = safe_div(rsq, vdot(Ap, p))
    x = x + broadcast(alpha)*p

    # r = r - broadcast(alpha)*Ap
    r = b - A(x) # More numerically stable

    rsq_new = vdot(r, r)
    beta_old = beta
    beta = safe_div(rsq_new, rsq)
    p = r + broadcast(beta)*p
    rsq = rsq_new

    # In the first iteration beta_old=0 and alpha_old=1
    # This makes the first T_off=0
    T_diag = safe_div(1, alpha) + safe_div(beta_old, alpha_old)
    T_off = safe_div(jnp.sqrt(beta_old), alpha_old)

    carry = (x, r, p, rsq, alpha, beta)
    if debug:
      return carry, (T_diag, T_off, carry)
    return carry, (T_diag, T_off)

  carry = x, r, p, rsq, alpha, beta
  carry, T_terms = jax.lax.scan(scan_body, carry, jnp.arange(max_iters), unroll=1)
  T_diag, T_off = T_terms[:2]
  x, r, p, rsq, alpha, beta = carry
  T_diag, T_off = T_diag.T, T_off.T

  # The log det will be wrong if we use too many of the T terms
  total_dim = list_prod(x.shape[1:])
  mask = jnp.arange(T_diag.shape[-1]) < total_dim
  T_diag, T_off = T_diag*mask, T_off*mask

  # Compute the eigenvalues of T.
  eigh = partial(jax.scipy.linalg.eigh_tridiagonal, eigvals_only=True)
  L = jax.vmap(eigh)(T_diag, T_off[:,1:])[...,::-1]

  # eigh_tridiagonal is not fully implemented, so iteratively find the eigenvectors
  a, d, c = T_off, T_diag, jnp.roll(T_off, -1, axis=-1)
  V = jax.vmap(tridiag_eigenvector_from_eigenvalues)(a, d, c, L)

  # Compute the log det of J^TJ
  log_L = jnp.where(L <= 1e-8, 0.0, jnp.log(L))
  log_L = jnp.where(mask, log_L, 0.0)
  log_det = vdot(b, b)*jax.vmap(jnp.vdot)(V[:,0,:]**2, log_L)

  if debug:
    scan_values = T_terms[-1]
    import pdb; pdb.set_trace()

  return x, log_det

################################################################################################################

def weighted_jacobi(fun, x, diagonal=None, alpha=1.0, max_iters=1000):
  b = x

  def loop(carry, inputs):
    x = carry
    if diagonal is not None:
      x_new = x - alpha*(fun(x) - b)/diagonal
    else:
      x_new = x - alpha*(fun(x) - b)
    return x_new, ()

  x, _ = jax.lax.scan(loop, x, jnp.arange(max_iters), unroll=10)
  return x

################################################################################################################

def brents(f, lower, upper, x, atol=1e-8, max_iters=20000, eps=1e-7):
  assert 0, "Not tested"
  def swap(a, b, fa, fb):
    swap_mask = jnp.abs(fa) < jnp.abs(fb)
    tmp_a = jnp.where(swap_mask, b, a)
    tmp_b = jnp.where(~swap_mask, a, b)
    a = tmp_a
    b = tmp_b
    fa, fb = f(a), f(b)
    assert jnp.all(jnp.abs(fa) > jnp.abs(fb))
    return a, b, fa, fb

  a, b = lower, upper
  fa, fb = f(a), f(b)
  a, b, fa, fb = swap(a, b, fa, fb)
  c, s, d = a, a, a
  fc, fs, fd = f(c), f(s), f(d)
  mflag = jnp.ones_like(x, dtype=bool)
  i = 0.0
  mflag, fa, fb, fc, fs, a, b, c, s, d, i = val

  def cond_fun(val):
    mflag, fa, fb, fc, fs, a, b, c, s, d, i = val

    max_iters_reached = jnp.where(i > max_iters, True, False)
    ba_close = jnp.allclose(b - a, 0.0, atol=atol)
    fb_zero = jnp.allclose(fb, 0.0, atol=atol)
    fs_zero = jnp.allclose(fs, 0.0, atol=atol)

    return ~(max_iters_reached | ba_close | fb_zero | fs_zero)

  def body(val):
    mflag, fa, fb, fc, fs, a, b, c, s, d, i = val

    fb_minus_fa = fb - fa
    fc_minus_fb = fc - fb
    fc_minus_fa = fc - fa

    fa_not_fc = jnp.abs(fb_minus_fa) < eps
    fb_not_fc = jnp.abs(fc_minus_fb) < eps

    # Inverse quadratic interpolation
    iqi  = a*fb*fc/(fb_minus_fa*fc_minus_fa)
    iqi -= b*fa*fc/(fb_minus_fa*fc_minus_fb)
    iqi += c*fa*fb/(fc_minus_fa*fc_minus_fb)

    # Secant method
    sm = b - fb*(b - a)/fb_minus_fa

    # Compute s
    s = jnp.where(fa_not_fc&fb_not_fc, iqi, sm)

    # Bisection method possibly
    cond1 = (s > jnp.maximum(b, 0.25*(3*a + b))) | (s < jnp.minimum(b, 0.25*(3*a + b)))
    cond2 = mflag  & (jnp.abs(s - b) >= 0.5*jnp.abs(c - b))
    cond3 = ~mflag & (jnp.abs(s - b) >= 0.5*jnp.abs(d - c))
    cond4 = mflag  & (jnp.abs(b - c) < eps)
    cond5 = ~mflag & (jnp.abs(d - c) < eps)
    mflag = cond1 | cond2 | cond3 | cond4 | cond5
    s = jnp.where(mflag, 0.5*(a + b), s)

    # Remaining steps
    fs = f(s)
    d = c
    c = b
    b = jnp.where(fa*fs < 0, s, b)
    a = jnp.where(fa*fs >= 0, s, a)
    a, b, fa, fb = swap(a, b, fa, fb)

    val = mflag, fa, fb, fc, fs, a, b, c, s, d, i + 1
    return val

  while cond_fun(val):
    val = body(val)

  return val

################################################################################################################

def newtons_with_grad(f_and_df, x, atol=1e-6, max_iters=1000):
  # Compute f^{-1}(x) using the newtons method.  f must be monotonic.

  def cond_fun(val):
    z, dz, i = val
    max_iters_reached = jnp.where(i > max_iters, True, False)
    tolerance_achieved = jnp.allclose(dz, 0.0, atol=atol)
    first_iter = jnp.where(i > 0.0, False, True)
    return ~(max_iters_reached | tolerance_achieved) | first_iter

  def body_fun(val):
    z, dz, i = val
    fz, dfz = f_and_df(z)
    z_new = z - (fz - x)/dfz
    dz = z - z_new
    return z_new, dz, i + 1

  z, dz, n_iters = jax.lax.while_loop(cond_fun, body_fun, (x, x, 0.0))
  return z

def newtons(f, x, atol=1e-6, max_iters=1000):
  # Compute f^{-1}(x) using the newtons method.  f must be monotonic.

  ones = jnp.ones_like(x)
  f_and_df = lambda z: jax.jvp(f, (z,), (ones,))
  return newtons_with_grad(f_and_df, x, atol=atol, max_iters=max_iters)

################################################################################################################

if __name__ == "__main__":
  import nux.util as util
  from debug import *
  import matplotlib.pyplot as plt
  from nux.flows.bijective.logistic_cdf_mixture_logit import logistic_cdf_mixture_logit

  K = 32
  x_shape = (32, 32, 3)

  rng_key = random.PRNGKey(0)
  x = random.normal(rng_key, x_shape)
  weight_logits, means, scales = random.normal(rng_key, (3, *x.shape, K))
  scales = util.square_plus(scales, gamma=1.0) + 1e-4

  f = lambda x: logistic_cdf_mixture_logit(weight_logits, means, scales, x)
  z = f(x)

  # lower = jnp.zeros_like(z) - 1000
  # upper = jnp.zeros_like(z) + 1000
  # val = bisection(f, lower, upper, z)
  # reconstr = val[2]

  reconstr = newtons(f, z)


  import pdb; pdb.set_trace()

