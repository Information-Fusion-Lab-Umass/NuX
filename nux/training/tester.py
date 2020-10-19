from functools import partial
import nux
import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from typing import Optional, Mapping, Callable, Sequence, Tuple, Any
from haiku._src.typing import Params, State, PRNGKey

################################################################################################################

@partial(jit, static_argnums=(0, 1, 2))
def scan_body(apply_fun: Callable,
              aggregate_fun: Callable,
              get_reconstr: bool,
              carry: Tuple[Params, State],
              inputs: Mapping[str, jnp.ndarray],
              **kwargs
  ) -> Tuple[Tuple[Params, State], Mapping[str, float]]:
  params, state = carry
  key, inputs = inputs

  # Take a gradient step
  outputs, _ = apply_fun(params, state, key, inputs, sample=False, **kwargs)
  if get_reconstr:
    reconstr, _ = apply_fun(params, state, key, outputs, sample=True, **kwargs)

  # Aggregate the findings
  if get_reconstr:
    test_metrics = aggregate_fun(inputs, outputs, reconstr)
  else:
    test_metrics = aggregate_fun(inputs, outputs)

  return (params, state), test_metrics

@partial(jit, static_argnums=(0, 1, 2))
def test_loop(apply_fun: Callable,
              aggregate_fun: Callable,
              get_reconstr: bool,
              params: Params,
              state: State,
              key: PRNGKey,
              inputs: Mapping[str, jnp.ndarray],
              **kwargs
  ) -> Tuple[Tuple[Params, State], Mapping[str, jnp.ndarray]]:
  """ Fast training loop using scan """

  # Fill the scan function
  body = partial(scan_body, apply_fun, aggregate_fun, get_reconstr, **kwargs)

  # Get the inputs for the scan loop
  n_iters = inputs["x"].shape[0]
  keys = random.split(key, n_iters)

  # Run the optimizer steps
  carry = (params, state)
  inputs = (keys, inputs)
  return jax.lax.scan(body, carry, inputs)

################################################################################################################

class Tester():

  def __init__(self,
               apply_fun: Callable,
               aggregate_fun: Callable=None,
               get_reconstr: bool=False):
    assert aggregate_fun is not None, "Expecting a function like 'aggregate_fun(inputs, outputs, reconstr)'"
    self.fast_test_loop = partial(test_loop, apply_fun, aggregate_fun, get_reconstr)
    self.fast_test_loop = jit(self.fast_test_loop)
    self.losses = []

  def multi_eval_step(self,
                      key: PRNGKey,
                      inputs: Mapping[str, jnp.ndarray],
                      params: Params,
                      state: State,
                      **kwargs
    ) -> Mapping[str, jnp.ndarray]:
    _, test_metrics = self.fast_test_loop(params, state, key, inputs, **kwargs)
    return test_metrics
