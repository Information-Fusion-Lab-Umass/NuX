# NoX - Normalizing Flows using JAX

## What is NoX?
NoX is a library for building Normalizing Flows using JAX.

## What are Normalizing Flows?
Normalizng Flows (http://proceedings.mlr.press/v37/rezende15.pdf) are a probabilistic modeling tool learn maximum likelihood models using invertible neural networks.  Given learn a transformation, ![f_{\theta}: \mathbb{R}^N \to \mathbb{R}^N](https://render.githubusercontent.com/render/math?math=f_%7B%5Ctheta%7D%3A%20%5Cmathbb%7BR%7D%5EN%20%5Cto%20%5Cmathbb%7BR%7D%5EN) between a nice latent variable, say ![z\sim N(0,I)](https://render.githubusercontent.com/render/math?math=z%5Csim%20N(0%2CI)), and data from an unknown probability distribution, ![x\sim p(x)](https://render.githubusercontent.com/render/math?math=x%5Csim%20p(x)) such that ![p(f(z)) \approx p(x)](https://render.githubusercontent.com/render/math?math=p(f(z))%20%5Capprox%20p(x)).  We do this using the probability change of variable formula ![p(x)=p(z)|\frac{dz}{dx}|](https://render.githubusercontent.com/render/math?math=p(x)%3Dp(z)%7C%5Cfrac%7Bdz%7D%7Bdx%7D%7C).  Properties of the Jacobian and determinant let us compose multiple functions, ![f_1, \dots, f_K](https://render.githubusercontent.com/render/math?math=f_1%2C%20%5Cdots%2C%20f_K), who are all easily invertible and have an easy to calculate Jacobian determinant, in order to build expressive transformations.

## Why use NoX?
Nox provides a simple interface for building normalizing flows.

```python
from jax import random, jit, value_and_grad
from normalizing_flows import sequential_flow, MAF, Reverse, UnitGaussianPrior
from util import TEST, TRAIN

# Build a dummy dataset
x_train, x_test = np.ones((70, 3)), np.ones((30, 3))

# Build a normalizing flow with 2 Masked Auto-Regressive Flows
flow = sequential_flow(MAF([1024], dropout=0.7), Reverse(), MAF([1024], dropout=0.7), UnitGaussianPrior())

# Initialize the flow.  This example will not condition on a variable.
key = random.PRNGKey(0)
names, output_shape, params, state = init_fun(key, input_shape=x.shape[-1], condition_shape=())

# Create the loss function and its gradients
def nll(params, state, x, **kwargs):
    log_px, z, updated_state = forward(params, state, np.zeros(x.shape[0]), x, cond=(), **kwargs)
    return -np.mean(log_px), updated_state
valgrad = jit(value_and_grad(nll, has_aux=True))

# Create the optimizer
opt_init, opt_update, get_params = optimizers.adam(0.001)
opt_state = opt_init(params)

# Train the flow
for i in range(100):
    key, *keys = random.split(key, 3)
    params = get_params(opt_state)
    (loss, state), grad = valgrad(params, state, x_train, key=keys[1], test=TRAIN)
    opt_state = opt_update(i, grad, opt_state)

# Generate samples from the model
z = random.normal(keys[1], (10, x.shape[-1]))
log_pfz, fz, _ = inverse(params, state, np.zeros(z.shape[0]), z, cond, key=keys[1], test=TEST)
```
