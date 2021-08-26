import jax.numpy as jnp
import jax
from jax import random

def geometric_roulette_coefficients(key, n_terms):
  # Compute the roulette coefficients using a geometric distribution
  k = jnp.arange(n_terms)
  p = 0.5
  u = random.uniform(key, (1,))[0]
  N = jnp.floor(jnp.log(u)/jnp.log(1 - p)) + 1
  p_N_geq_k = (1 - p)**k

  # Zero out the terms that are over N
  roulette_coeff = jnp.where(k > N, 0.0, 1/p_N_geq_k)
  return roulette_coeff
