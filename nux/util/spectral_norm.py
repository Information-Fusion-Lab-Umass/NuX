import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
import nux.util as util
from typing import Optional, Mapping, Callable, Sequence, Any, Union, Tuple
import haiku as hk

__all__ = ["spectral_norm_apply",
           "spectral_norm_conv_apply",
           "check_spectral_norm",
           "induced_norm_conv"]

################################################################################################################

def check_spectral_norm(pytree):
  """ Check the spectral norm of the leaves of pytree """
  def get_sn(val):
    return jnp.linalg.norm(val, ord=2) if val.ndim == 2 else 0

  return jax.tree_util.tree_map(get_sn, pytree)

################################################################################################################

def spectral_norm_iter(W, uv):
  """ Perform a single spectral norm iteration """
  u, v = uv

  # Perform the spectral norm iterations
  v = W.T@u
  v *= jax.lax.rsqrt(jnp.dot(v, v) + 1e-12)

  u = W@v
  u *= jax.lax.rsqrt(jnp.dot(u, u) + 1e-12)

  return (u, v)

def spectral_norm_apply(W: jnp.ndarray,
                        u: jnp.ndarray,
                        v: jnp.ndarray,
                        scale: float,
                        n_iters: int,
                        update_params: bool):
  """ Perform at most n_iters single spectral norm iterations """

  if update_params:

    # Perform the spectral norm iterations
    # For some reason using a for loop is way faster and uses way less memory
    # than using the lax loops in fixed_point.py
    if n_iters is None:
      (u, v) = util.fixed_point(spectral_norm_iter, W, (u, v), 5000)
    else:
      for i in range(n_iters):
        uv = spectral_norm_iter(W, (u, v))
        u, v = uv

        # Relaxation method
        # https://hal-cea.archives-ouvertes.fr/cea-01403292/file/residual-method.pdf Eq.(8)
        u = 0.5*uv[0] + 0.5*u
        v = 0.5*uv[1] + 0.5*v

    u = jax.lax.stop_gradient(u)
    v = jax.lax.stop_gradient(v)

  # Estimate the largest singular value of W
  sigma = jnp.einsum("i,ij,j", u, W, v)

  # Scale coefficient to account for the fact that sigma can be an under-estimate.
  factor = jnp.where(scale < sigma, scale/sigma, 1.0)

  return W*factor, u, v

################################################################################################################

def spectral_norm_conv_iter(W, uv, stride, padding):
  """ Perform a single spectral norm iteration """
  u, v = uv

  # Perform the spectral norm iterations
  v = jax.lax.conv_transpose(u[None], W, strides=stride, padding=padding, dimension_numbers=("NHWC", "HWIO", "NHWC"), transpose_kernel=True)[0]
  v *= jax.lax.rsqrt(jnp.sum(v**2) + 1e-12)

  u = jax.lax.conv_general_dilated(v[None], W, window_strides=stride, padding=padding, dimension_numbers=("NHWC", "HWIO", "NHWC"))[0]
  u *= jax.lax.rsqrt(jnp.sum(u**2) + 1e-12)

  return (u, v)

def spectral_norm_conv_apply(W: jnp.ndarray,
                             u: jnp.ndarray,
                             v: jnp.ndarray,
                             stride: Sequence[int],
                             padding: Union[str, Sequence[Tuple[int, int]]],
                             scale: float,
                             n_iters: int,
                             update_params: bool):
  """ Perform n_iters single spectral norm iterations """
  height, width, C_in, C_out = W.shape

  if update_params:
    # Perform the spectral norm iterations
    if n_iters is None:
      body = partial(spectral_norm_conv_iter, stride=stride, padding=padding)
      (u, v) = util.fixed_point(body, W, (u, v), 5000)

    else:
      while_loop = False
      if while_loop == False:
        # For loops are way faster than scan on GPUs when n_iters is low
        for i in range(n_iters):
          uv = spectral_norm_conv_iter(W, (u, v), stride, padding)

          # Relaxation method
          # https://hal-cea.archives-ouvertes.fr/cea-01403292/file/residual-method.pdf Eq.(8)
          u = 0.5*uv[0] + 0.5*u
          v = 0.5*uv[1] + 0.5*v
      else:
        assert 0, "This causes memory usage to blow up!"
        body = partial(spectral_norm_conv_iter, stride=stride, padding=padding)
        (u, v) = util.fixed_point(body, W, (u, v), 20)

    u = jax.lax.stop_gradient(u)
    v = jax.lax.stop_gradient(v)

  # Estimate the largest singular value of W
  Wv = jax.lax.conv_general_dilated(v[None], W, window_strides=stride, padding=padding, dimension_numbers=("NHWC", "HWIO", "NHWC"))[0]
  sigma = jnp.sum(u*Wv)

  # Scale coefficient to account for the fact that sigma can be an under-estimate.
  factor = jnp.where(scale < sigma, scale/sigma, 1.0)

  return W*factor, u, v

################################################################################################################

def f(mvp, A, x):
  y_hat = mvp(A, x)
  y_norm = jax.lax.rsqrt(jnp.sum(y_hat**2))
  y = y_hat*y_norm
  return y, y_hat

def F(mvp, mvpT, ut, W):
  vt, _ = f(mvpT, W, ut)
  utp1, utp1_hat = f(mvp, W, vt)
  return utp1, (vt, utp1_hat)

@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def max_singular_value(mvp, mvpT, W, ut, zt):
  utp1, vjp, (vt, utp1_hat) = jax.vjp(partial(F, mvp, mvpT), ut, W, has_aux=True)
  sigma = jnp.vdot(ut, utp1_hat)

  vjp_u, _ = vjp(zt)
  ztp1 = utp1_hat + vjp_u

  return sigma, utp1, ztp1

@max_singular_value.defjvp
def max_singular_value_jvp(mvp, mvpT, primals, tangents):
  W, ut, zt = primals
  utp1, vjp, (vt, utp1_hat) = jax.vjp(partial(F, mvp, mvpT), ut, W, has_aux=True)

  # Use this so that we can compute the outer product when we're doing convolutions
  sigma, uvT = jax.value_and_grad(lambda W: jnp.vdot(ut, mvp(W, vt)))(W)

  vjp_u, vjp_W = vjp(zt)
  ztp1 = utp1_hat + vjp_u
  dsigma = uvT + vjp_W

  tangent_out = jnp.sum(tangents[0]*dsigma)

  return (sigma, utp1, ztp1), (tangent_out, jnp.zeros_like(utp1), jnp.zeros_like(ztp1))

################################################################################################################

def max_singular_value_general(*,
                               mvp,
                               mvpT,
                               W,
                               u,
                               zeta,
                               n_iters):

  def g(mvp, A, x):
    y_hat = mvp(A, x)
    y_norm = jax.lax.rsqrt(jnp.sum(y_hat**2))
    y = y_hat*y_norm
    return y

  def ϕ(u, W):
    v = g(mvpT, W, u)
    return jnp.vdot(u, W@v)

  def F(u, W):
    v = g(mvpT, W, u)
    u = g(mvp, W, v)
    return u

  return util.fixed_point_fun(ϕ, F, n_iters, u, zeta, W)

def max_singular_value2(W,
                       u,
                       zeta,
                       n_iters):
  mvp = lambda A, x: A@x
  mvpT = lambda A, x: A.T@x
  sigma, u, zeta = max_singular_value_general(mvp=mvp,
                                              mvpT=mvpT,
                                              W=W,
                                              u=u,
                                              zeta=zeta,
                                              n_iters=n_iters)
  u, zeta = jax.lax.stop_gradient((u, zeta))
  return sigma, u, zeta

def max_singular_value_conv2(W,
                            u,
                            zeta,
                            n_iters,
                            stride,
                            padding):

  def mvp(A, x):
    return jax.lax.conv_general_dilated(x[None],
                                        A,
                                        window_strides=stride,
                                        padding=padding,
                                        dimension_numbers=("NHWC", "HWIO", "NHWC"))[0]

  def mvpT(A, x):
    return jax.lax.conv_transpose(x[None],
                                  A,
                                  strides=stride,
                                  padding=padding,
                                  dimension_numbers=("NHWC", "HWIO", "NHWC"),
                                  transpose_kernel=True)[0]

  sigma, u, zeta = max_singular_value_general(mvp=mvp,
                                              mvpT=mvpT,
                                              W=W,
                                              u=u,
                                              zeta=zeta,
                                              n_iters=n_iters)
  u, zeta = jax.lax.stop_gradient((u, zeta))
  return sigma, u, zeta

################################################################################################################

def induced_norm_general(*,
                         mvp,
                         mvpT,
                         u,
                         zeta,
                         W,
                         p,
                         q,
                         n_iters):
  theta = (W, p, q)

  def norm(x, p):
    # x is expected to be positive, so don't need absolute value
    return (x**p).sum()**(1/p)

  def g(mvp, A, x, s, t):
    y_hat = mvp(A, x)
    y_phase = jnp.sign(y_hat)
    y_mag_hat = jnp.abs(y_hat)**s

    # Divide out the largest value before computing the norm.
    # This constant will cancel out when we normalize
    max_val = jax.lax.stop_gradient(y_mag_hat.max())
    y_mag_hat /= max_val

    y_mag = y_mag_hat/norm(y_mag_hat, t)
    return y_phase*y_mag

  def ϕ(u, theta):
    W, p, q = theta
    s = 1/(p-1)
    t = p
    v = g(mvpT, W, u, s, t)
    return jnp.vdot(u, mvp(W, v))

  def F(u, theta):
    W, p, q = theta
    v = g(mvpT, W, u, 1/(p - 1), p)
    u = g(mvp, W, v, q - 1, q/(q - 1))
    return u

  return util.fixed_point_fun(ϕ, F, n_iters, u, zeta, theta)

def induced_norm(W,
                 p,
                 q,
                 u,
                 zeta,
                 n_iters):
  mvp = lambda A, x: A@x
  mvpT = lambda A, x: A.T@x
  sigma, u, zeta = induced_norm_general(mvp=mvp,
                                        mvpT=mvpT,
                                        W=W,
                                        p=p,
                                        q=q,
                                        u=u,
                                        zeta=zeta,
                                        n_iters=n_iters)
  u, zeta = jax.lax.stop_gradient((u, zeta))
  return sigma, u, zeta

def induced_norm_conv(W,
                      p,
                      q,
                      u,
                      zeta,
                      n_iters,
                      stride,
                      padding):

  def mvp(A, x):
    return jax.lax.conv_general_dilated(x[None],
                                        A,
                                        window_strides=stride,
                                        padding=padding,
                                        dimension_numbers=("NHWC", "HWIO", "NHWC"))[0]

  def mvpT(A, x):
    return jax.lax.conv_transpose(x[None],
                                  A,
                                  strides=stride,
                                  padding=padding,
                                  dimension_numbers=("NHWC", "HWIO", "NHWC"),
                                  transpose_kernel=True)[0]

  sigma, u, zeta = induced_norm_general(mvp=mvp,
                                        mvpT=mvpT,
                                        W=W,
                                        p=p,
                                        q=q,
                                        u=u,
                                        zeta=zeta,
                                        n_iters=n_iters)
  u, zeta = jax.lax.stop_gradient((u, zeta))
  return sigma, u, zeta

################################################################################################################

if __name__ == "__main__":
  from debug import *

  def loss(W_hat):
    return (jnp.sin(W_hat)**2).sum()

  def truth(W):
    sigma = jnp.linalg.svd(W, compute_uv=False)[0]
    W_hat = W/sigma
    return loss(W_hat)

  def mvp(A, x):
    return A@x

  def mvpT(A, x):
    return A.T@x

  def comp(W, u, z):
    # sigma, u, z = max_singular_value(mvp, mvpT, W, u, z)
    sigma, u, z = max_singular_value2(W, u, z, 100)
    # sigma, u, z = induced_norm(W, 2.0, 2.0, u, z, 100)
    W_hat = W/sigma
    (u, z) = jax.lax.stop_gradient((u, z))
    return loss(W_hat), (u, z)

  key = random.PRNGKey(0)
  dim1, dim2 = 4, 3
  W = random.normal(key, (dim1, dim2))
  u, z = random.normal(key, (2, dim1))

  # for i in range(10):
  #   sigma, u, z = max_singular_value2(mvp, mvpT, W, u, z)
    # sigma, u, z = max_singular_value(mvp, mvpT, W, u, z)

  dW_true = jax.grad(truth)(W)
  dW_comp, (u, z) = jax.grad(comp, has_aux=True)(W, u, z)

  import pdb; pdb.set_trace()
