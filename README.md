# NoX - Normalizing Flows using JAX

## What is NoX?
NoX is a library for building Normalizing Flows using JAX.

## What are Normalizing Flows?
Normalizng Flows (http://proceedings.mlr.press/v37/rezende15.pdf) are a probabilistic modeling tool learn maximum likelihood models using invertible neural networks.  Given learn a transformation, $f_{\theta}: \mathbb{R}^N \to \mathbb{R}^N$ between a nice latent variable, say $z\sim N(0,I)$, and data from an unknown probability distribution, $x\sim p(x)$ such that $p(f(z)) \approx p(x)$.  We do this using the probability change of variable formula $p(x)=p(z)|\frac{dz}{dx}|.  Properties of the Jacobian and determinant let us compose multiple functions, $f_1, \dots, f_K$, who are all easily invertible and have an easy to calculate Jacobian determinant, in order to build expressive transformations.

## Why use NoX?
Nox provides a simple interface for building normalizing flows.

```python
from tqdm.notebook import tnrange
from jax import random, jit, value_and_grad
from normalizing_flows import sequential_flow, MAF, Reverse, UnitGaussianPrior
from util import TEST, TRAIN

# Build a dummy dataset
data = np.ones((100, 3))
x = data
x_train = data[70:]
x_test = data[:70]

# Build a normalizing flow with 2 Masked Auto-Regressive Flows
flow = sequential_flow(MAF([1024, 1024], dropout=0.7),
                       Reverse(),
                       MAF([1024, 1024], dropout=0.7),
                       UnitGaussianPrior())

# Initialize the flow.  This example will not condition on a variable.
key = random.PRNGKey(0)
input_shape = x.shape[1:]
condition_shape = (input_shape,)
cond = ()
names, output_shape, params, state = init_fun(key, input_shape, condition_shape)

# Create the loss function and its gradients
@jit
def nll(params, state, x, **kwargs):
    cond = ()
    log_px, z, updated_state = forward(params, state, np.zeros(x.shape[0]), x, cond, **kwargs)
    return -np.mean(log_px), updated_state
valgrad = jit(value_and_grad(nll, has_aux=True))

# Create the optimizer
opt_init, opt_update, get_params = optimizers.adam(0.001)
opt_update = jit(opt_update)
opt_state = opt_init(params)

# Train
batch_size = 256

pbar = tnrange(5000)
for i in pbar:
    key, *keys = random.split(key, 3)

    batch_idx = random.randint(keys[0], (batch_size,), minval=0, maxval=x_train.shape[0])
    x_batch = x_train[batch_idx,:]

    params = get_params(opt_state)
    (loss, state), grad = valgrad(params, state, x_batch, key=keys[1], test=TRAIN)
    opt_state = opt_update(i, grad, opt_state)

    pbar.set_description('Negative Log Likelihood: %5.3f'%(loss))

key, *keys = random.split(key, 3)
params = get_params(opt_state)

# Evaluate the test log likelihood
log_px, z, _ = forward(params, state, np.zeros(x_test.shape[0]), x_test, cond, key=keys[0], test=TEST)

# Generate samples from the model
z = random.normal(keys[1], (10, x.shape[-1]))
log_pfz, fz, _ = forward(params, state, np.zeros(z.shape[0]), z, cond, key=keys[1], test=TEST)
```
