import jax
import jax.numpy as jnp

def cartesian_to_spherical(x):
  r = jnp.sqrt(jnp.sum(x**2))
  denominators = jnp.sqrt(jnp.cumsum(x[::-1]**2)[::-1])[:-1]
  phi = jnp.arccos(x[:-1]/denominators)

  last_value = jnp.where(x[-1] >= 0, phi[-1], 2*jnp.pi - phi[-1])
  phi = jax.ops.index_update(phi, -1, last_value)

  return jnp.hstack([r, phi])

def spherical_to_cartesian(phi_x):
  r = phi_x[0]
  phi = phi_x[1:]
  return r*jnp.hstack([1.0, jnp.cumprod(jnp.sin(phi))])*jnp.hstack([jnp.cos(phi), 1.0])

def spherical_interpolation(x, y, N):
  x_flat, unflatten = jax.flatten_util.ravel_pytree(x)
  y_flat, _ = jax.flatten_util.ravel_pytree(y)
  sx, sy = cartesian_to_spherical(x_flat), cartesian_to_spherical(y_flat)
  spherical_interpolation_points = jnp.linspace(sx, sy, N)
  interpolation_points_flat = jax.vmap(spherical_to_cartesian)(spherical_interpolation_points)
  return jax.vmap(unflatten)(interpolation_points_flat)