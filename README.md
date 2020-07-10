# NuX - Normalizing Flows using JAX

## What is NuX?
NuX is a library for building [normalizing flows](https://arxiv.org/pdf/1912.02762.pdf) using [JAX](https://github.com/google/jax).

## Why use NuX?
NuX has many normalizing flow layers implemented with an easy to use interface.

```python
import nux.flows as nux
import jax
from jax import random
import jax.numpy as jnp
key = random.PRNGKey(0)

# Build a dummy dataset
x_train, x_test = jnp.ones((2, 100, 4))

# Build a simple normalizing flow
init_fun = nux.sequential(nux.Coupling(),
                          nux.ActNorm(),
                          nux.UnitGaussianPrior())

# Perform data-dependent initialization
_, flow = init_fun(key, {'x': x_train}, batched=True)

# Run data through the flow
inputs = {'x': x_test}
outputs, _ = flow.apply(flow.params, flow.state, inputs)
z, log_likelihood = outputs['x'], outputs['log_pz'] + outputs['log_det']

# Check the reconstructions
reconst, _ = flow.apply(flow.params, flow.state, {'x': z}, reverse=True)

assert jnp.allclose(x_test, reconst['x'])
```