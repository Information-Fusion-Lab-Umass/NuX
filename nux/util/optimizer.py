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

class AddNoiseState(OptState):
  """State for adding gradient noise. Contains a count for annealing."""
  count: jnp.ndarray
  rng_key: jnp.ndarray

def add_noise(eta: float, gamma: float, seed: int) -> GradientTransformation:
  """Add gradient noise.  This is implemented pretty poorly in Optax.  Don't want to split
  the key a bunch of times every gradient step
  References:
    [Neelakantan et al, 2014](https://arxiv.org/abs/1511.06807)
  Args:
    eta: base variance of the gaussian noise added to the gradient.
    gamma: decay exponent for annealing of the variance.
    seed: seed for random number generation.
  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return AddNoiseState(
        count=jnp.zeros([], jnp.int32), rng_key=jax.random.PRNGKey(seed))

  def update_fn(updates, state, params=None):  # pylint: disable=missing-docstring
    del params
    # num_vars = len(jax.tree_leaves(updates))
    # treedef = jax.tree_structure(updates)
    count_inc = _safe_int32_increment(state.count)
    var = eta / count_inc**gamma
    # all_keys = jax.random.split(state.rng_key, num=num_vars + 1)

    flat_updates, unflatten_updates = jax.flatten_util.ravel_pytree(updates)
    noise = random.normal(state.rng_key, flat_updates.shape)
    updates = unflatten_updates(flat_updates + var*noise)

    _, key = random.split(state.rng_key, 2)
    return updates, AddNoiseState(count=count_inc, rng_key=key)

    # noise = jax.tree_multimap(
    #     lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype),
    #     updates, jax.tree_unflatten(treedef, all_keys[1:]))
    # updates = jax.tree_multimap(
    #     lambda g, n: g + variance.astype(g.dtype) * n,
    #     updates, noise)
    # return updates, AddNoiseState(count=count_inc, rng_key=all_keys[0])

  return GradientTransformation(init_fn, update_fn)

################################################################################################################

def linear_lr(i, lr=1e-4):
  return lr

def linear_warmup_schedule(i, warmup=1000, lr_decay=1.0):
  return jnp.where(i < warmup,
                   i/warmup,
                   (lr_decay**(i - warmup)))

def linear_warmup_lr_schedule(i, warmup=1000, lr_decay=1.0, lr=1e-4):
  return jnp.where(i < warmup,
                   lr*i/warmup,
                   lr*(lr_decay**(i - warmup)))
