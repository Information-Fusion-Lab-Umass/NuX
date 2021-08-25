import jax
import jax.numpy as jnp

def st_floor(x):
  return x - jax.lax.stop_gradient(x) + jax.lax.stop_gradient(jnp.floor(x))

def st_round(x):
  return x - jax.lax.stop_gradient(x) + jax.lax.stop_gradient(jnp.round(x))
