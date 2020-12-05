import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
import chex
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, Union

OptState = NamedTuple  # Transformation states are (possibly empty) namedtuples.
Params = Any  # Parameters are arbitrary nests of `jnp.ndarrays`.
Updates = Params  # Gradient updates are of the same type as parameters.

################################################################################################################

class GradientTransformation(NamedTuple):
  """Optax transformations consists of a function pair: (initialise, update)."""
  init: Callable[  # Function used to initialise the transformation's state.
      [Params], Union[OptState, Sequence[OptState]]]
  update: Callable[  # Function used to apply a transformation.
      [Updates, OptState, Optional[Params]], Tuple[Updates, OptState]]

def _update_moment(updates, moments, decay, order):
  return jax.tree_multimap(
      lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)

def _safe_int32_increment(count):
  chex.assert_type(count, jnp.int32)
  max_int32_value = jnp.iinfo(jnp.int32).max
  one = jnp.array(1, dtype=jnp.int32)
  return jnp.where(count < max_int32_value, count + one, max_int32_value)


def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  bias_correction = 1 - decay**count
  return jax.tree_map(lambda t: t / bias_correction.astype(t.dtype), moment)

class ScaleByBeliefState(OptState):
  """State for the rescaling by AdaBelief algorithm."""
  count: jnp.ndarray  # shape=(), dtype=jnp.int32.
  mu: Updates
  nu: Updates

def scale_by_belief(b1: float=0.9,
                    b2: float=0.999,
                    eps: float=0.,
                    eps_root: float=1e-16) -> GradientTransformation:
  """ This isn't added to optax yet
  """

  def init_fn(params):
    mu = jax.tree_map(jnp.zeros_like, params)  # First moment
    s = jax.tree_map(jnp.zeros_like, params)  # Second Central moment
    return ScaleByBeliefState(count=jnp.zeros([], jnp.int32), mu=mu, nu=s)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, b1, 1)
    prediction_error = jax.tree_multimap(lambda g, m: g-m, updates, state.mu)
    nu = _update_moment(prediction_error, state.nu, b2, 2)
    count_inc = _safe_int32_increment(state.count)
    mu_hat = _bias_correction(mu, b1, count_inc)
    nu_hat = _bias_correction(nu, b2, count_inc)
    updates = jax.tree_multimap(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    return updates, ScaleByBeliefState(count=count_inc, mu=mu, nu=nu)

  return GradientTransformation(init_fn, update_fn)

################################################################################################################

def linear_warmup_lr_schedule(i, warmup=1000, lr_decay=1.0, lr=1e-4):
  return jnp.where(i < warmup,
                   lr*i/warmup,
                   lr*(lr_decay**(i - warmup)))
