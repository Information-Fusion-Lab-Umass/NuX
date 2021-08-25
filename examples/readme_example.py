import jax
import jax.numpy as jnp
import nux

# Generate some fake data
rng_key = jax.random.PRNGKey(0)
batch_size, dim = 8, 3
x = jax.random.normal(rng_key, (batch_size, dim))

# Create your normalizing flow
flow = nux.Sequential([nux.DenseMVP(),
                       nux.SneakyReLU(),
                       nux.UnitGaussianPrior()])

# Initialize the flow with data-dependent initialization
z, log_px = flow(x, rng_key=rng_key)

# Retrieve the initialized parameters from the flow
params = flow.get_params()

# Generate samples
x_samples, log_px_samples = flow(jnp.zeros_like(x), params=params, inverse=True, rng_key=rng_key)

# Pass the samples to the latent space
z, log_px = flow(x_samples, params=params, inverse=False)

# Reconstruct the samples
x_reconstr, log_px_reconstr = flow(z, params=params, inverse=True, reconstruction=True)

assert jnp.allclose(x_samples, x_reconstr)
assert jnp.allclose(log_px_samples, log_px)
assert jnp.allclose(log_px, log_px_reconstr)