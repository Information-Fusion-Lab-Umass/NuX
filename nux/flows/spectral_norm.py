import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
import nux.util as util
import nux.flows.base as base

__all__ = ['spectral_norm_body',
           'spectral_norm_apply',
           'spectral_norm_tree',
           'initialize_spectral_norm_u_tree',
           'check_spectral_norm',
           'spectral_norm_wrapper']

################################################################################################################

@jit
def spectral_norm_body(carry, inputs):
    W, u, v = carry

    v = W.T@u
    v *= jax.lax.rsqrt(jnp.dot(v, v) + 1e-12)

    u = W@v
    u *= jax.lax.rsqrt(jnp.dot(u, u) + 1e-12)

    return (W, u, v), inputs

@partial(jit, static_argnums=(3,))
def spectral_norm_apply(W, u, scale, n_iters):
    # v is set inside the loop, so just pass in a dummy value.
    v = jnp.zeros((W.shape[1],))
    (W, u, v), _ = jax.lax.scan(spectral_norm_body, (W, u, v), jnp.arange(n_iters))
    u = jax.lax.stop_gradient(u)
    v = jax.lax.stop_gradient(v)

    sigma = jnp.einsum('i,ij,j', u, W, v)

    # Scale coefficient to account for the fact that sigma can be an under-estimate.
    factor = jnp.where(scale < sigma, scale/sigma, 1.0)

    return W*factor, u

@partial(jit, static_argnums=(3,))
def spectral_norm_tree(pytree, u_tree, scale, n_iters):

    def apply_spectral_norm(val, u):
        # Only want to do this for matrices.
        if(val.ndim == 2):
            return spectral_norm_apply(val, u, scale, n_iters)
        return val, u

    return util.tree_multimap_multiout(apply_spectral_norm, pytree, u_tree)

def initialize_spectral_norm_u_tree(key, pytree):
    key_tree = util.key_tree_like(key, pytree)

    # Initialize the u tree
    def gen_u(key, val):
        return random.normal(key, (val.shape[0],)) if val.ndim == 2 else ()
    return jax.tree_util.tree_multimap(gen_u, key_tree, pytree)

def check_spectral_norm(pytree):

    def spectral_norm_apply(val):
        return jnp.linalg.norm(val, ord=2) if val.ndim == 2 else 0

    return jax.tree_util.tree_map(spectral_norm_apply, pytree)

################################################################################################################

def spectral_norm_wrapper(flow_init, scale=0.97, spectral_norm_iters=1, name='spectral_norm'):

    flow_apply_fun = None

    def apply_fun(params, state, inputs, *args, **kwargs):

        updated_state = {}

        # Apply spectral normalization
        params, updated_state['u_tree'] = spectral_norm_tree(params, state['u_tree'], scale, spectral_norm_iters)

        # Run the flow
        outputs, updated_state['flow_state'] = flow_apply_fun(params, state['flow_state'], inputs, *args, **kwargs)

        return outputs, updated_state

    def init_fun(key, inputs, **kwargs):
        k1, k2, k3 = random.split(key, 3)

        outputs, flow = flow_init(k1, inputs, **kwargs)

        # # Swap the apply functions
        nonlocal flow_apply_fun
        flow_apply_fun = flow.apply

        # Make sure that we keep track of the u_tree
        u_tree = initialize_spectral_norm_u_tree(k2, flow.params)

        # Initialize the parameters with some spectral norm
        params, u_tree = spectral_norm_tree(flow.params, u_tree, scale, 20)

        state = {'flow_state': flow.state, 'u_tree': u_tree}

        outputs, _ = apply_fun(params, state, inputs, key=k3, **kwargs)

        flow = base.Flow(name,
                         flow.input_shapes,
                         flow.output_shapes,
                         flow.input_ndims,
                         flow.output_ndims,
                         params,
                         state,
                         apply_fun)
        return outputs, flow

    return init_fun
