import numpy as np
import jax.numpy as jnp
from jax import random
import haiku as hk
import jax
import nux.flows.base as base

class GaussianMADE(hk.Module):

    def __init__(self, input_sel, dim, hidden_layer_sizes, reverse=False, method='sequential', key=None, name=None):
        super().__init__(name=name)

        # Store the dimensions for later
        self.dim = dim
        self.hidden_layer_sizes = hidden_layer_sizes

        """ Build the autoregressive masks """
        layer_sizes = hidden_layer_sizes + [dim]

        self.input_sel = input_sel

        # Build the hidden layer masks
        masks = []
        sel = input_sel
        prev_sel = sel

        if(method == 'random'):
            keys = random.split(key, len(layer_sizes))
            key_iter = iter(keys)

        for size in layer_sizes:

            # Choose the degrees of the next layer
            if(method == 'random'):
                sel = random.randint(next(key_iter), shape=(size,), minval=min(jnp.min(sel), dim - 1), maxval=dim)
            else:
                sel = jnp.arange(size)%max(1, dim - 1) + min(1, dim - 1)

            # Create the new mask
            mask = (prev_sel[:,None] <= sel).astype(jnp.int32)
            prev_sel = sel
            masks.append(mask)

        # Will use these at runtime to mask the linear layers
        self.masks = tuple(masks)

        # # Build the mask for the matrix between the input and output
        # self.skip_mask = (input_sel[:,None] < input_sel).astype(jnp.int32)

        # Build the output layers.  Remember that we need to output mu and sigma.  Just need
        # a triangular matrix for the masks
        self.out_mask = (prev_sel[:,None] < input_sel).astype(jnp.int32)

    def __call__(self, inputs, **kwargs):

        w_init = hk.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='truncated_normal')

        # Main autoregressive transform
        x = inputs
        layer_sizes = [self.dim] + self.hidden_layer_sizes + [self.dim]
        for i, (mask, input_size, output_size) in enumerate(zip(self.masks, layer_sizes[:-1], layer_sizes[1:])):
            w = hk.get_parameter('w_%d'%i, [input_size, output_size], jnp.float32, init=w_init)
            b = hk.get_parameter('b_%d'%i, [output_size], jnp.float32, init=jnp.zeros)
            w_masked = w*mask
            x = jnp.dot(x, w_masked) + b

        # # Skip connection # Implemented this wrong probably
        # w_skip = hk.get_parameter('w_skip', [self.dim, self.dim], jnp.float32, init=w_init)
        # x += jnp.dot(x, w_skip*self.skip_mask)

        # Split into two parameters
        w_mu = hk.get_parameter('w_mu', [self.dim, self.dim], jnp.float32, init=w_init)
        w_alpha = hk.get_parameter('w_alpha', [self.dim, self.dim], jnp.float32, init=w_init)

        mu = jnp.dot(x, w_mu*self.out_mask)
        alpha = jnp.dot(x, w_alpha*self.out_mask)
        alpha = jnp.tanh(alpha)

        return mu, alpha

################################################################################################################

@base.auto_batch
def MAF(hidden_layer_sizes,
        reverse=False,
        method='sequential',
        key=None,
        name='maf',
        **kwargs):
    # language=rst
    """
    Masked Autoregressive Flow https://arxiv.org/pdf/1705.07057.pdf
    Invertible network that enforces autoregressive property.

    :param hidden_layer_sizes: A list of the size of the feature network
    :param reverse: Whether or not to reverse the inputs
    :param method: Either 'sequential' or 'random'.  Controls how indices are assigned to nodes in each layer
    :param key: JAX random key.  Only needed in random mode
    """
    network = None
    input_sel = None

    def forward(params, state, inputs, **kwargs):
        x = inputs['x']
        network_params = params['hk_params']

        mu, alpha = network.apply(network_params, x)
        z = (x - mu)*jnp.exp(-alpha)
        log_det = -alpha.sum(axis=-1)

        outputs = {'x': z, 'log_det': log_det}
        return outputs, state

    def inverse(params, state, inputs, **kwargs):
        z = inputs['x']
        network_params = params['hk_params']

        x = jnp.zeros_like(z)
        u = z

        # For each MADE block, need to build output a dimension at a time
        def carry_body(carry, inputs):
            x, idx = carry, inputs
            mu, alpha = network.apply(network_params, x)
            w = mu + u*jnp.exp(alpha)
            x = jax.ops.index_update(x, idx, w[idx])
            return x, alpha[idx]

        indices = jnp.nonzero(input_sel == (1 + np.arange(x.shape[0])[:,None]))[1]
        x, alpha_diag = jax.lax.scan(carry_body, x, indices)
        log_det = -alpha_diag.sum(axis=-1)

        outputs = {'x': x, 'log_det': log_det}
        return outputs, state

    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        if(reverse == False):
            return forward(params, state, inputs, **kwargs)
        return inverse(params, state, inputs, **kwargs)

    def create_params_and_state(key, input_shapes):
        x_shape = input_shapes['x']
        assert len(x_shape) == 1
        dim = x_shape[0]

        # We can either assign indices randomly or sequentially.  FOR LOW DIMENSIONS USE SEQUENTIAL!!!
        nonlocal input_sel
        if(method == 'random'):
            assert key is not None
            keys = random.split(key, len(layer_sizes) + 1)
            key_iter = iter(keys)
            input_sel = random.randint(next(key_iter), shape=(dim,), minval=1, maxval=dim+1)
        else:
            # Alternate direction in consecutive layers
            input_sel = jnp.arange(1, dim + 1)
            if(reverse):
                input_sel = input_sel[::-1]

        # Create the network
        nonlocal network
        network = hk.transform(lambda x: GaussianMADE(input_sel, dim, hidden_layer_sizes, reverse, method, name)(x))

        # Initialize it.  Remember that this function expects an unbatched input
        params = {'hk_params': network.init(key, jnp.zeros(x_shape))}
        state = {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

__all__ = ['MAF']