### Check out an improved version of NuX called [generax](https://github.com/EddieCunningham/generax)!

# NuX - Normalizing Flows using JAX

[![.github/workflows/CI.yml](https://github.com/MilesCranmer/NuX/actions/workflows/CI.yml/badge.svg)](https://github.com/MilesCranmer/NuX/actions/workflows/CI.yml)

## What is NuX?
NuX is a library for building [normalizing flows](https://arxiv.org/pdf/1912.02762.pdf) using [JAX](https://github.com/google/jax).

## What are normalizing flows?
Normalizing flows learn a parametric model over an unknown probability density function using data.  We assume that data points are sampled i.i.d. from an unknown distribution p(x).  Normalizing flows learn a parametric approximation of the true data distribution using maximum likelihood learning.  The learned distribution can be efficiently sampled from and has a log likelihood that can be evaluated exactly.

## Why use NuX?
It is easy to build, train and evaluate normalizing flows with NuX

```python
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

# Generate samples.  The prior layer will use the input's shape for sampling.
x_samples, log_px_samples = flow(jnp.zeros_like(x), params=params, inverse=True, rng_key=rng_key)

# Pass the samples to the latent space
z, log_px = flow(x_samples, params=params, inverse=False)

# Reconstruct the samples
x_reconstr, log_px_reconstr = flow(z, params=params, inverse=True, reconstruction=True)

assert jnp.allclose(x_samples, x_reconstr)
assert jnp.allclose(log_px_samples, log_px)
assert jnp.allclose(log_px, log_px_reconstr)
```

## Get started
The easiest way to use NuX is to clone this repo and install the prerequisites with the "pip install ." command.  JAX should be manually installed (see [this](https://github.com/google/jax#installation)) because GPU support is system dependent.  The NuX package available on pip is outdated and does not have much of the functionality of the current code.

NuX is in active development, so expect the API to change over time.  Contributions are welcome!
