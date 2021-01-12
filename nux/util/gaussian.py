import jax.numpy as jnp
from jax import jit
import jax

@jit
def gaussian_chol_cov_logpdf(x, mean, cov_chol):
  dx = x - mean
  y = jax.lax_linalg.triangular_solve(cov_chol, dx, lower=True, transpose_a=True)
  log_px = -0.5*jnp.sum(y**2) - jnp.log(jnp.diag(cov_chol)).sum() - 0.5*x.shape[0]*jnp.log(2*jnp.pi)
  return log_px

@jit
def gaussian_centered_full_cov_logpdf(x, cov):
  cov_inv = jnp.linalg.inv(cov)
  log_px = -0.5*jnp.sum(jnp.dot(x, cov_inv.T)*x, axis=-1)
  return log_px - 0.5*jnp.linalg.slogdet(cov)[1] - 0.5*x.shape[-1]*jnp.log(2*jnp.pi)

@jit
def gaussian_full_cov_logpdf(x, mean, cov):
  dx = x - mean
  cov_inv = jnp.linalg.inv(cov)
  log_px = -0.5*jnp.sum(jnp.dot(dx, cov_inv.T)*dx, axis=-1)
  return log_px - 0.5*jnp.linalg.slogdet(cov)[1] - 0.5*x.shape[-1]*jnp.log(2*jnp.pi)

@jit
def gaussian_centered_diag_cov_logpdf(x, log_diag_cov):
  log_px = -0.5*jnp.sum(x**2*jnp.exp(-log_diag_cov), axis=-1)
  return log_px - 0.5*jnp.sum(log_diag_cov) - 0.5*x.shape[-1]*jnp.log(2*jnp.pi)

@jit
def gaussian_diag_cov_logpdf(x, mean, log_diag_cov):
  dx = x - mean
  log_px = -0.5*jnp.sum(dx**2*jnp.exp(-log_diag_cov), axis=-1)
  return log_px - 0.5*jnp.sum(log_diag_cov) - 0.5*x.shape[-1]*jnp.log(2*jnp.pi)

@jit
def unit_gaussian_logpdf(x):
  if(x.ndim > 1):
    return jax.vmap(unit_gaussian_logpdf)(x)
  return -0.5*jnp.dot(x, x) - 0.5*x.shape[-1]*jnp.log(2*jnp.pi)
