from functools import partial
import jax.nn.initializers as jaxinit
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
from jax.experimental import optimizers
import nux.util as util

################################################################################################################

@partial(jit, static_argnums=(0, 1, 2))
def scan_body(valgrad, opt_update, get_params, carry, inputs, clip=True):
    params, state, opt_state = carry
    i, key, inputs = inputs

    # Take a gradient step
    (train_loss, (outputs, state)), grad = valgrad(params, state, inputs, key=key)

    if(clip):
        grad = jit(optimizers.clip_grads)(grad, 5.0)

    # Update the parameters and optimizer state
    opt_state = opt_update(i, grad, opt_state)
    params = get_params(opt_state)

    return (params, state, opt_state), (train_loss, outputs)

@partial(jit, static_argnums=(0, 1, 2))
def train_loop(valgrad, opt_update, get_params, params, state, opt_state, key, inputs, iter_numbers):
    """ Fast training loop using scan """

    # Fill the scan function
    body = partial(scan_body, valgrad, opt_update, get_params)

    # Get the inputs for the scan loop
    n_iters = iter_numbers.shape[0]
    keys = random.split(key, n_iters)

    # Run the optimizer steps
    carry = (params, state, opt_state)
    inputs = (iter_numbers, keys, inputs)
    return jax.lax.scan(body, carry, inputs)

################################################################################################################

class Trainer():
    def __init__(self, loss_fun, params, clip=5.0, warmup=None, lr_decay=1.0, lr=1e-4):
        self.loss_fun = loss_fun
        self.valgrad = jit(jax.value_and_grad(self.loss_fun, has_aux=True))

        # Optionally use a learning schedule
        if(warmup is None):
            opt_init, opt_update, get_params = optimizers.adam(lr)
        else:
            schedule = partial(util.linear_warmup_lr_schedule, warmup=warmup, lr_decay=lr_decay, lr=lr)
            opt_init, opt_update, get_params = optimizers.adam(schedule)

        # Initialize the optimizer state
        self.opt_update, self.get_params = jit(opt_update), jit(get_params)
        self.opt_state = opt_init(params)

        # Gradient clipping is crucial for tough datasets!
        if(clip is not None):
            self.clip = partial(optimizers.clip_grads, max_norm=clip)
            self.clip = jit(self.clip)
        else:
            self.clip = None

        self.training_steps = 0
        self.losses = []

        self.fast_train = partial(train_loop, self.valgrad, self.opt_update, self.get_params)

    def grad_step(self, key, inputs, params, state, **kwargs):

        # Compute the gradients
        (loss, (outputs, state)), grad = self.valgrad(params, state, inputs, key=key, **kwargs)
        self.losses.append(loss)

        # Clip the gradients
        if(self.clip):
            grad = self.clip(grad)

        # Take a grad step
        self.opt_state = self.opt_update(self.training_steps, grad, self.opt_state)
        self.training_steps += 1

        params = self.get_params(self.opt_state)

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

################################################################################################################
