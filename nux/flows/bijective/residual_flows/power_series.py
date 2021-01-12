import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from functools import partial
import nux.util as util
from typing import Optional, Mapping, Callable, Sequence

def geometric_roulette_coefficients(k_start, k_end, key):
  # Compute the roulette coefficients using a geometric distribution
  k = jnp.arange(k_start, k_end)
  p = 0.5
  u = random.uniform(key, (1,))[0]
  N = jnp.floor(jnp.log(u)/jnp.log(1 - p)) + 1
  p_N_geq_k = (1 - p)**k

  # Zero out the terms that are over N
  roulette_coeff = jnp.where(k > N, 0.0, 1/p_N_geq_k)

  return roulette_coeff

def unbiased_neumann_coefficients(key, n_terms, n_exact):
  # Compute the standard scaling terms for the neumann series
  k = jnp.arange(0, n_terms)
  coeff = (-1)**k

  # Compute the roulette scaling terms
  roulette_coeff = geometric_roulette_coefficients(0, n_terms - n_exact, key)

  # We don't want to apply this to the exact terms
  roulette_coeff = jnp.hstack([jnp.ones(n_exact), roulette_coeff])

  return coeff*roulette_coeff

################################################################################################################

def vjp_iterations_scan(vjp_fun, v, n_terms):

  @jit
  def scan_fun(carry, inputs):
    w = carry
    w_updated, = vjp_fun(w)
    return w_updated, w

  k = jnp.arange(n_terms - 1)
  w, terms = jax.lax.scan(scan_fun, v, k)
  terms = jnp.concatenate([terms, w[None]], axis=0)

  return terms

def vjp_iterations_for_loop(vjp_fun, v, n_terms):

  terms = [None]*n_terms

  w = v
  terms[0] = w
  for i in range(1, n_terms):
    w, = vjp_fun(w)
    terms[i] = w

  return jnp.array(terms)

def unbiased_neumann_vjp_terms(vjp_fun, v, rng, n_terms=10, n_exact=4):
  # This function assumes that we start at k=0!

  # Compute the terms in the power series.
  # terms = vjp_iterations_scan(vjp_fun, v, n_terms)
  terms = vjp_iterations_for_loop(vjp_fun, v, n_terms)

  # Compute the coefficients for each term
  coeff = unbiased_neumann_coefficients(rng, n_terms, n_exact)
  coeff = util.broadcast_to_first_axis(coeff, terms.ndim)

  return coeff*terms

################################################################################################################

def jacobian_power_iterations(J, n_terms):

  def scan_fun(carry, inputs):
    J_k = carry
    J_kp1 = J@J_k
    return J_kp1, J_k

  k = jnp.arange(n_terms - 1)
  jac_K, terms = jax.lax.scan(scan_fun, J, k)
  I = jnp.expand_dims(jnp.eye(J.shape[-1]), axis=tuple(range(len(J.shape) - 2)))
  I = jnp.broadcast_to(I, J.shape)
  terms = jnp.concatenate([I[None], terms], axis=0)

  return terms

def neumann_jacobian_terms(J, rng, n_terms=10, n_exact=4):

  terms = jacobian_power_iterations(J, n_terms)

  # Compute the coefficients for each term
  coeff = unbiased_neumann_coefficients(rng, n_terms, n_exact)
  coeff = util.broadcast_to_first_axis(coeff, terms.ndim)

  return coeff*terms
