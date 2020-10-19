from functools import partial
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
import optax
import nux.util as util
from typing import Optional, Mapping, Callable, Sequence, Tuple, Any
from haiku._src.typing import Params, State, PRNGKey

__all__ = ["Trainer"]

################################################################################################################

@partial(jit, static_argnums=(0, 1))
def scan_body(valgrad: Callable,
              opt_update: Callable,
              carry: Tuple[Params, State, State],
              inputs: Mapping[str, jnp.ndarray]
  ) -> Tuple[Tuple[Params, State, State], Tuple[float, Mapping[str, jnp.ndarray]]]:
  params, state, opt_state = carry
  i, key, inputs = inputs

  # Take a gradient step
  (train_loss, (outputs, state)), grad = valgrad(params, state, key, inputs)

  # Update the parameters and optimizer state
  updates, opt_state = opt_update(grad, opt_state, params)
  params = jit(optax.apply_updates)(params, updates)

  return (params, state, opt_state), (train_loss, outputs)

@partial(jit, static_argnums=(0, 1))
def train_loop(valgrad: Callable,
               opt_update: Callable,
               params: Params,
               state: State,
               opt_state: State,
               key: PRNGKey,
               inputs: Mapping[str, jnp.ndarray],
               iter_numbers: jnp.ndarray
  ) -> Tuple[Tuple[Params, State, State], Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]]:
  """ Fast training loop using scan """

  # Fill the scan function
  body = partial(scan_body, valgrad, opt_update)

  # Get the inputs for the scan loop
  n_iters = iter_numbers.shape[0]
  keys = random.split(key, n_iters)

  # Run the optimizer steps
  carry = (params, state, opt_state)
  inputs = (iter_numbers, keys, inputs)
  return jax.lax.scan(body, carry, inputs)

################################################################################################################

class Trainer():
  def __init__(self, loss_fun, params, optimizer=None, clip=5.0, lr=5e-4, warmup=2000):
    self.loss_fun = loss_fun
    self.valgrad = jax.value_and_grad(self.loss_fun, has_aux=True)
    self.valgrad = jit(self.valgrad)

    if optimizer is None:
      warmup_schedule = partial(util.linear_warmup_lr_schedule, warmup=warmup, lr_decay=1.0, lr=-lr)
      opt_init, opt_update = optax.chain(optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
                                         # optax.additive_weight_decay(0.0),
                                         # optax.scale_by_trust_ratio(),
                                         optax.scale_by_schedule(warmup_schedule),
                                         optax.clip(clip))
    else:
      opt_init, opt_update = optimizer

    # Initialize the optimizer state
    self.opt_update = jit(opt_update)
    self.opt_state = opt_init(params)

    self.training_steps = 0
    self.losses = []

    self.fast_train = partial(train_loop, self.valgrad, self.opt_update)
    self.fast_train = jit(self.fast_train)

  def grad_step(self, key, inputs, params, state, **kwargs):

    inputs = (self.training_steps, key, inputs)
    carry = (params, state, self.opt_state)
    (params, state, self.opt_state), (train_loss, outputs) = scan_body(self.valgrad, self.opt_update, carry, inputs)
    self.losses.append(train_loss)
    self.training_steps += 1
    return train_loss, outputs, params, state


    # Compute the gradients
    (loss, (outputs, state)), grad = self.valgrad(params, state, key, inputs, **kwargs)
    self.losses.append(loss)

    # Take a grad step
    updates, self.opt_state = self.opt_update(grad, self.opt_state, params)
    params = jit(optax.apply_updates)(params, updates)

    self.training_steps += 1

    return loss, outputs, params, state

  def multi_grad_step(self, key, inputs, params, state):
    # Assumes that we are passing things in correctly
    n_iters = inputs['x'].shape[0]
    iter_numbers = jnp.arange(self.training_steps, self.training_steps + n_iters)
    (params, state, opt_state), (train_losses, outputs) = self.fast_train(params, state, self.opt_state, key, inputs, iter_numbers)

    self.losses.extend(list(train_losses))
    self.training_steps += n_iters
    self.opt_state = opt_state
    return (train_losses, outputs), params, state

  def save_opt_state_to_file(self, path=None):
    assert path is not None

    opt_state_path = os.path.join(path, 'opt_state.pickle')
    util.save_pytree(self.opt_state, opt_state_path, overwrite=True)

  def load_param_and_state_from_file(self, path=None):
    assert path is not None

    opt_state_path = os.path.join(path, 'opt_state.pickle')
    self.opt_state = util.load_pytree(opt_state_path)

################################################################################################################
