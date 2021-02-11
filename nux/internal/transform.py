from functools import partial, wraps
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
import haiku as hk
from abc import ABC, abstractmethod
import warnings
from typing import Optional, Mapping, Type, Callable, Iterable, Any, Sequence, Union, Tuple, MutableMapping, NamedTuple, Set, TypeVar
from nux.internal.base import get_constant, new_custom_context
from nux.internal.layer import Layer
from haiku._src.typing import PRNGKey, Params, State
from haiku._src.transform import TransformedWithState, \
                                 to_prng_sequence, \
                                 check_mapping, \
                                 INIT_RNG_ERROR, \
                                 APPLY_RNG_STATE_ERROR, \
                                 APPLY_RNG_ERROR

__all__ = ["transform_flow",
           "transform_flow_from_fun"]

def transform_flow_from_fun(fun) -> TransformedWithState:

  # We will keep the expected shapes for the flow here so that
  # JAX will compile these constants
  constants = None

  def init_fn(rng: Optional[Union[PRNGKey]],
              inputs: Mapping[str, jnp.ndarray],
              batch_axes=(),
              return_initial_output=False,
              **kwargs
  ) -> Tuple[Params, State]:
    """ Initializes your function collecting parameters and state. """
    rng = to_prng_sequence(rng, err_msg=INIT_RNG_ERROR)
    with new_custom_context(rng=rng) as ctx:
      # Load the batch axes for the inputs
      Layer.batch_axes = batch_axes
      Layer._is_initializing = True

      key = hk.next_rng_key()

      # Initialize the model
      outputs = fun(inputs, key, **kwargs)

      # Unset the batch axes
      Layer.batch_axes = ()
      Layer._is_initializing = False

    nonlocal constants
    params, state, constants = ctx.collect_params(), ctx.collect_initial_state(), ctx.collect_constants()

    if return_initial_output:
      return params, state, outputs

    return params, state

  def apply_fn(params: Optional[Params],
               state: Optional[State],
               rng: Optional[Union[PRNGKey]],
               inputs,
               **kwargs
  ) -> Tuple[Any, State]:
    """ Applies your function injecting parameters and state. """
    params = check_mapping("params", params)
    state = check_mapping("state", state)

    rng = to_prng_sequence(rng, err_msg=(APPLY_RNG_STATE_ERROR if state else APPLY_RNG_ERROR))
    with new_custom_context(params=params, state=state, constants=constants, rng=rng) as ctx:
      key = hk.next_rng_key()
      out = fun(inputs, key, **kwargs)
    return out, ctx.collect_state()

  return TransformedWithState(init_fn, apply_fn)

################################################################################################################

def transform_flow(create_fun) -> TransformedWithState:

  # We will keep the expected shapes for the flow here so that
  # JAX will compile these constants
  constants = None

  def init_fn(rng: Optional[Union[PRNGKey]],
              inputs: Mapping[str, jnp.ndarray],
              batch_axes=(),
              return_initial_output=False,
              **kwargs
  ) -> Tuple[Params, State]:
    """ Initializes your function collecting parameters and state. """
    rng = to_prng_sequence(rng, err_msg=INIT_RNG_ERROR)
    with new_custom_context(rng=rng) as ctx:
      # Create the model
      model = create_fun()

      # Load the batch axes for the inputs
      Layer.batch_axes = batch_axes
      Layer._is_initializing = True

      key = hk.next_rng_key()

      # Initialize the model
      outputs = model(inputs, key, **kwargs)

      # Unset the batch axes
      Layer.batch_axes = ()
      Layer._is_initializing = False

    nonlocal constants
    params, state, constants = ctx.collect_params(), ctx.collect_initial_state(), ctx.collect_constants()

    if return_initial_output:
      return params, state, outputs

    return params, state

  def apply_fn(params: Optional[Params],
               state: Optional[State],
               rng: Optional[Union[PRNGKey]],
               inputs,
               **kwargs
  ) -> Tuple[Any, State]:
    """ Applies your function injecting parameters and state. """
    params = check_mapping("params", params)
    state = check_mapping("state", state)

    rng = to_prng_sequence(rng, err_msg=(APPLY_RNG_STATE_ERROR if state else APPLY_RNG_ERROR))
    with new_custom_context(params=params, state=state, constants=constants, rng=rng) as ctx:
      model = create_fun()
      key = hk.next_rng_key()
      out = model(inputs, key, **kwargs)
    return out, ctx.collect_state()

  return TransformedWithState(init_fn, apply_fn)

################################################################################################################
