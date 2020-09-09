from functools import partial
import nux
import jax
import jax.numpy as jnp
from jax import random, vmap, jit

################################################################################################################

@partial(jit, static_argnums=(0, 1))
def scan_body(apply_fun, aggregate_fun, carry, inputs, **kwargs):
    params, state = carry
    key, inputs = inputs

    # Take a gradient step
    outputs, _ = apply_fun(params, state, inputs, key=key, reverse=False, **kwargs)
    reconstr, _ = apply_fun(params, state, outputs, key=key, reverse=True, **kwargs)

    # Aggregate the findings
    test_metrics = aggregate_fun(inputs, outputs, reconstr)

    return (params, state), test_metrics

@partial(jit, static_argnums=(0, 1))
def test_loop(apply_fun, aggregate_fun, params, state, key, inputs, **kwargs):
    """ Fast training loop using scan """

    # Fill the scan function
    body = partial(scan_body, apply_fun, aggregate_fun, **kwargs)

    # Get the inputs for the scan loop
    n_iters = inputs['x'].shape[0]
    keys = random.split(key, n_iters)

    # Run the optimizer steps
    carry = (params, state)
    inputs = (keys, inputs)
    return jax.lax.scan(body, carry, inputs)

################################################################################################################

class Tester():

    def __init__(self, apply_fun, aggregate_fun=None):
        assert aggregate_fun is not None, 'Expecting a function like "aggregate_fun(inputs, outputs, reconstr)"'
        self.fast_test_loop = partial(test_loop, apply_fun, aggregate_fun)
        self.fast_test_loop = jit(self.fast_test_loop)

    def multi_eval_step(self, key, inputs, params, state, **kwargs):
        _, test_metrics = self.fast_test_loop(params, state, key, inputs, **kwargs)
        return test_metrics
