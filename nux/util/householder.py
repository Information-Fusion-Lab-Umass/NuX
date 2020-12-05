import jax.numpy as jnp
from jax import jit
from functools import partial
import jax

@jit
def householder(x, v):
  return x - 2*jnp.einsum('i,j,j', v, v, x)/jnp.sum(v**2)

@jit
def householder_prod_body(carry, inputs):
  x = carry
  v = inputs
  return householder(x, v), 0

@jit
def householder_prod(x, vs):
  return jax.lax.scan(householder_prod_body, x, vs)[0]

@jit
def householder_prod_transpose(x, vs):
  return jax.lax.scan(householder_prod_body, x, vs[::-1])[0]

@jit
def householder_apply(U, log_s, VT, z):
  # Compute Az
  x = householder_prod(z, VT)
  x = x*jnp.exp(log_s)
  x = jnp.pad(x, (0, U.shape[1] - z.shape[0]))
  x = householder_prod(x, U)
  return x

@jit
def householder_pinv_apply(U, log_s, VT, x):
  # Compute A^+@x and also return U_perp^T@x
  UTx = householder_prod_transpose(x, U)
  z, UperpTx = jnp.split(UTx, jnp.array([log_s.shape[0]]))
  z = z*jnp.exp(-log_s)
  z = householder_prod_transpose(z, VT)
  return z, UperpTx

@jit
def householder_to_dense(U, log_s, VT):
  return jax.vmap(partial(householder_apply, U, log_s, VT))(jnp.eye(VT.shape[0])).T

@jit
def householder_pinv_to_dense(U, log_s, VT):
  return jax.vmap(partial(householder_pinv_apply, U, log_s, VT))(jnp.eye(U.shape[0]))[0].T
