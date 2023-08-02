import jax.numpy as jnp
from functools import partial
import jax

def householder(x, v):
  return x - 2*jnp.einsum('i,j,j', v, v, x)/jnp.sum(v**2)

def householder_prod_body(carry, inputs):
  x = carry
  v = inputs
  return householder(x, v), 0

def householder_prod(x, vs):
  return jax.lax.scan(householder_prod_body, x, vs)[0]

def householder_prod_transpose(x, vs):
  return jax.lax.scan(householder_prod_body, x, vs[::-1])[0]

def householder_apply(U, log_s, VT, z):
  # Compute Az
  x = householder_prod(z, VT)
  x = x*jnp.exp(log_s)
  x = jnp.pad(x, (0, U.shape[1] - z.shape[0]))
  x = householder_prod(x, U)
  return x

def householder_pinv_apply(U, log_s, VT, x):
  # Compute A^+@x and also return U_perp^T@x
  UTx = householder_prod_transpose(x, U)
  z, UperpTx = jnp.split(UTx, jnp.array([log_s.shape[0]]))
  z = z*jnp.exp(-log_s)
  z = householder_prod_transpose(z, VT)
  return z, UperpTx

def householder_to_dense(U, log_s, VT):
  return jax.vmap(partial(householder_apply, U, log_s, VT))(jnp.eye(VT.shape[0])).T

def householder_pinv_to_dense(U, log_s, VT):
  return jax.vmap(partial(householder_pinv_apply, U, log_s, VT))(jnp.eye(U.shape[0]))[0].T

if __name__ == "__main__":
  from debug import *
  import matplotlib.pyplot as plt

  rng_key = random.PRNGKey(0)
  x = random.normal(rng_key, (3, 64))
