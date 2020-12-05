
# NuX - Normalizing Flows using JAX

## What is NuX?
NuX is a library for building [normalizing flows](https://arxiv.org/pdf/1912.02762.pdf) using [JAX](https://github.com/google/jax).

## What are normalizing flows?
Normalizing flows learn a parametric model over an unknown probability density function using data.  We assume that data points are sampled i.i.d. from an unknown distribution p(x).  Normalizing flows learn a parametric approximation of the true data distribution using maximum likelihood learning.  The learned distribution can be efficiently sampled from and has a log likelihood that can be evaluated exactly.

## Why use NuX?
It is easy to build, train and evaluate normalizing flows with NuX

```python
import nux
import jax
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
key = jax.random.PRNGKey(0)

# Build a dummy dataset
x_train, x_test = jnp.ones((2, 100, 2))
train_inputs, test_inputs = {"x": x_train}, {"x": x_test}

# Build a simple normalizing flow
def create_flow():
  return nux.sequential(nux.Coupling(), nux.AffineLDU(), nux.UnitGaussianPrior())

# Perform data-dependent initialization
flow = nux.transform_flow(create_flow)
params, state = flow.init(key, train_inputs, batch_axes=(0,))

# Create our loss function
def negative_log_likelihood(params, state, key, inputs):
  outputs, updated_state = flow.apply(params, state, key, inputs)
  log_px = outputs["log_pz"] + outputs["log_det"]
  return -log_px.mean(), updated_state

# Generate the gradient function
grad_fun = jax.grad(negative_log_likelihood, has_aux=True)

# Train the flow using Optax
opt_init, opt_update = optax.adam(lr=1e-3)
opt_state = opt_init(params)

for i, key in enumerate(random.split(key, 100)):
  state, grads = grad_fun(params, state, key, train_inputs)
  updates, opt_state = opt_update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)

# Pull samples from the model and plot
shape_placeholder = {"x": jnp.zero_like(x_train)}
outputs, _ = flow.apply(params, state, shape_placeholder, sample=True)
plt.scatter(*outputs["x"].T)
```
Check out the tutorial or any of the examples to get a better idea of how to use NuX.


## Installation
For the moment, NuX only works with python 3.7.  The steps to install are:

     pip install nux
     pip install git+https://github.com/deepmind/dm-haiku

If you want GPU support for JAX, follow the intructions here https://github.com/google/jax#installation
