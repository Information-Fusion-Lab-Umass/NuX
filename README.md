# NuX - Normalizing Flows using JAX

## What is NuX?
NuX is a library for building [normalizing flows](https://arxiv.org/pdf/1912.02762.pdf) using [JAX](https://github.com/google/jax).

## What are normalizing flows?
Normalizing flows learn a parametric model over an unknown probability density function using data.  We assume that data points are sampled i.i.d. from an unknown distribution p(x).  Normalizing flows learn a parametric approximation of the true data distribution, q(x), using maximum likelihood learning.  The resulting q(x) can be efficiently sampled from and evaluated exactly.

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

## What's implemented?
Check out the [bijective](https://github.com/Information-Fusion-Lab-Umass/NuX/blob/master/nux/flows/bijective/README.md), [injective](https://github.com/Information-Fusion-Lab-Umass/NuX/blob/master/nux/flows/injective/README.md) and [surjective](https://github.com/Information-Fusion-Lab-Umass/NuX/blob/master/nux/flows/surjective/README.md) transformations that are implemented.  Any contributions are welcome!

## How does it work?
The modularity of normalizing flows allows us to construct complex flows using code that is made simple using JAX.  

### Create complex flows
Flow layers can be imported from nux.flows and chained together sequentially using ```nux.flows.sequential```.  For more complex flows, we can split the flow using the [probability chain rule](https://en.wikipedia.org/wiki/Chain_rule_(probability)) in ```nux.flows.ChainRule``` and then run multiple flows in parallel using ```nux.flows.parallel```.

Using these basic transformations, we can easily create complex flows.  For example, a [multiscale](https://arxiv.org/pdf/1605.08803.pdf) [GLOW](https://arxiv.org/pdf/1807.03039.pdf) normalizing flow can be implemented easily:

```python
import nux.flows as nux

def multi_scale(flow, existing_flow):
    # This is implemented in nux.flows.compose
    return nux.sequential(flow,
                          nux.Squeeze(),
                          nux.ChainRule(2, factor=True),
                          nux.factored(existing_flow, nux.Identity()),
                          nux.ChainRule(2, factor=False),
                          nux.UnSqueeze())
                      
def GLOWBlock():
    return nux.sequential(nux.ActNorm(),
                          nux.OnebyOneConv(),
                          nux.Coupling(n_channels=512))

def GLOW(num_blocks=4):
    layers = [GLOWBlock() for _ in range(num_blocks)]
    return nux.sequential(*layers)

def MultiscaleGLOW(quantize_bits=3):
    flow = nux.Identity()
    flow = multi_scale(GLOW(), flow)
    flow = multi_scale(GLOW(), flow)
    flow = multi_scale(GLOW(), flow)
    flow = multi_scale(GLOW(), flow)

    flow = nux.sequential(nux.UniformDequantization(scale=2**quantize_bits),
                          nux.Logit(),
                          nux.Squeeze(), # So that the channel is divisible by 2
                          flow,
                          nux.Flatten(),
                          nux.AffineGaussianPriorDiagCov(out_dim=128)) # Use a low dimensional prior for best results!
                          
    flow_init_fun = flow # The result of creating layers is an initializer
    return flow_init_fun
```

### Initialize your flow with data
NuX initializes flows using data to infer the input and output shapes at each flow layer and to help initialize layers like [Actnorm](https://arxiv.org/pdf/1807.03039.pdf).  NuX uses dictionaries as the primary data-structure to pass data between flow layers.  

```python
import jax
import jax.numpy as jnp

flow_init_fun = MultiscaleGLOW()
key = jax.random.PRNGKey(0)
inputs = {'x': jnp.zeros(64, 32, 32, 3)} # Create a dummy dataset
outputs, flow = flow_init_fun(key, inputs, batched=True) # Must specify if the input data is batched or not
```

#### Inputs/Outputs
Each flow application expects an input dictionary.  The key 'x' should correspond to the data that is passed between flow layers.  Furthermore all elements of the input dictionary are passed to each flow layer.  For example, in classification we can pass labels to any layer using:
```python
inputs = {'x': data, 'y': labels}
```

Every flow layer returns a dictionary of outputs that contains the transformed data in key 'x' and a log likelihood contribution term.  For standard transformations, the log likelihood contribution term is under 'log_det' and 'log_pz' for priors.  Like the inputs, flow layers can also output other key value pairs.

#### Flow data structure
The second value returned by an initializer call is the [flow data structure](https://github.com/Information-Fusion-Lab-Umass/NuX/blob/master/nux/flows/base.py#L10).  This data structure contains the name of a layer and dictionaries of the input/output shapes/dims, parameters, state and apply function.  The shapes/dims aid auto-batching while the parameters and state parametrize the flow.  The difference between parameters and state is that parameters is intended to contain the parameters that will be trained with gradient descent while the state values do not (like the running statistics in batch normalization).  ```jax.tree_util.tree_map``` and ```jax.tree_util.tree_multimap``` are your friend when working with dictionaries!

The apply function is called with the parameters, state, inputs, keyword arguments and a flag that specifies which direction to run the flow:
```python
# Run the flow forwards (x -> z)
outputs, updated_state = flow.apply(flow.params, flow.state, inputs, key=key, reverse=False)
log_px = outputs['log_pz'] + outputs['log_det']

# Run the flow forwards in reverse (z -> x)
reconstr_inputs, _ = flow.apply(flow.params, flow.state, outputs, key=key, reverse=True)
```

### Use [Haiku](https://github.com/deepmind/dm-haiku) to create deep flow layers
Flow layers, like ```nux.flows.Coupling```, can use a neural network to introduce complex non-linearities.  These neural networks must constructed using [Haiku](https://github.com/deepmind/dm-haiku).  There are default networks that flow layers default to, but any Haiku network can be used.  For example, we can construct a transformation for image coupling layers as follows:
```python
class SimpleConv(hk.Module):

    def __init__(self, out_shape, n_hidden_channels, name=None):
        super().__init__(name=name)
        _, _, out_channels = out_shape
        self.out_channels = out_channels
        self.n_hidden_channels = n_hidden_channels
        self.last_channels = 2*out_channels

    def __call__(self, x, **kwargs):
        H, W, C = x.shape # NuX ensures that the input will be unbatched!

        x = hk.Conv2D(output_channels=self.n_hidden_channels,
                      kernel_shape=(3, 3),
                      stride=(1, 1))(x[None])[0]
        x = jax.nn.relu(x)
        x = hk.Conv2D(output_channels=self.n_hidden_channels,
                      kernel_shape=(1, 1),
                      stride=(1, 1))(x[None])[0]
        x = jax.nn.relu(x)
        x = hk.Conv2D(output_channels=self.last_channels,
                      kernel_shape=(3, 3),
                      stride=(1, 1),
                      w_init=hk.initializers.Constant(0),
                      b_init=hk.initializers.Constant(0))(x[None])[0]

        mu, alpha = jnp.split(x, 2, axis=-1)
        alpha = jnp.tanh(alpha)
        return mu, alpha
```
NuX handles batching internally, so every flow and network is guaranteed to be passed an unbatched input.

### Using JAX
NuX is built using JAX so all of its features can be used on a flow.

## Creating custom flow layers
Custom flows are easy to create in NuX. NuX can internally handle batch dimensions, so custom layers can be implemented assuming the input is unbatched:
```python
import jax.numpy as jnp
import jax.nn.initializers as jaxinit
import nux.flows.base as base

@base.auto_batch # Ensure that apply_fun receives unbatched inputs.
def OnebyOneConvDense(W_init=jaxinit.glorot_normal(), name='1x1conv_dense'):

    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        x = inputs['x']                  # Unpack the inputs
        W = params['W']                  # Unpack the parameters
        height, width, channel = x.shape # auto_batch ensures x is unbatched!
        
        # Compute the transformation
        if(reverse == False):
            z = jnp.einsum('ij,hwj->hwi', W, x)
        else:
            W_inv = jnp.linalg.inv(W)
            z = jnp.einsum('ij,hwj->hwi', W_inv, x)
        
        # Compute the log Jacobian determinant
        log_det = jnp.linalg.slogdet(W)[1]
        log_det *= height*width
        
        # Return the outputs and update the state if necessary.
        outputs = {'x': z, 'log_det': log_det}
        updated_state = state
        return outputs, updated_state

    def create_params_and_state(key, input_shapes):
        # The data_dependent=False flag below ensures that input_shapes is unbatched.
        height, width, channel = input_shapes['x']
        
        # Initialize the parameters
        W = W_init(key, (channel, channel))

        params, state = {'W': W}, {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state, data_dependent=False) # Helper to put everything together.
```
Under the hood, ```base.initialize``` extracts the shapes from the input to the initializer.  These shapes are passed to ```create_params_and_state``` to generate the parameters and state.  The inputs, parameters and state are then passed to ```apply_fun``` to compute the outputs.  Finally, the shapes/dimensions of the outputs are retrieved and stored.

At runtime, ```base.auto_batch``` has access to the unbatched input dimensions for each flow.  With this information, it recursively applies ```jax.vmap``` to correctly handle nested batching.

The ```data_dependent``` flag can be set to in order to pass the batched inputs to ```create_params_and_state```.  For ```Actnorm```, this looks like:
```python

@base.auto_batch
def ActNorm(log_s_init=jaxinit.zeros, b_init=jaxinit.zeros, name='act_norm'):
    multiply_by = None # We can store initialize time constants in this outer scope.

    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        if(reverse == False):
            z = (inputs['x'] - params['b'])*jnp.exp(-params['log_s'])
        else:
            z = jnp.exp(params['log_s'])*inputs['x'] + params['b']
        log_det = -params['log_s'].sum()*multiply_by
        outputs = {'x': z, 'log_det': log_det}
        return outputs, state

    def create_params_and_state(key, inputs, batch_depth):
        # The shape of x is the unbatched shape of x prepended with batch_depth dimensions.
        x = inputs['x'] 

        # Need to keep track of the dimensionality of all but the last axis in case we pass in an image.
        nonlocal multiply_by
        multiply_by = jnp.prod([s for i, s in enumerate(x.shape) if i >= batch_depth and i < len(x.shape) - 1])

        # Create the parameters using the batch of data
        axes = tuple(jnp.arange(len(x.shape) - 1))
        params = {'b'    : jnp.mean(x, axis=axes),
                  'log_s': jnp.log(jnp.std(x, axis=axes) + 1e-5)}
        state = {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state, data_dependent=True)
```

### Testing a custom flow
```nux.tests.nf_test.flow_test``` is a simple function to test the correctness of a flow.  It checks unbatched/batched/doubly-batched reconstructions by running a flow forwards then in reverse and checks the log Jacobian determinant against the brute force solution computed using ```jax.jacobian```.
```python
init_fun = Flow()
unbatched_inputs = {'x': data}
flow_test(init_fun, unbatched_inputs, key)
```

## Installation
For the moment, NuX only works with python 3.7.  The steps to install are:

     pip install nux
     pip install git+https://github.com/deepmind/dm-haiku

If you want GPU support for JAX, follow the intructions here https://github.com/google/jax#installation
