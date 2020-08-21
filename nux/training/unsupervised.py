from functools import partial
import jax.nn.initializers as jaxinit
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
from jax.experimental import optimizers
import nux.util as util

__all__ = ['nll_loss', 'GenerativeModel', 'ImageGenerativeModel', '_ImageMixin']

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

    return (params, state, opt_state), train_loss

@partial(jit, static_argnums=(0, 1, 2))
def train_loop(valgrad, opt_update, get_params, params, state, opt_state, key, x, iter_numbers):
    """ Fast training loop using scan """

    # Fill the scan function
    body = partial(scan_body, valgrad, opt_update, get_params)

    # Get the inputs for the scan loop
    n_iters = x.shape[0]
    inputs = {'x': x}
    keys = random.split(key, n_iters)

    # Run the optimizer steps
    carry = (params, state, opt_state)
    inputs = (iter_numbers, keys, inputs)
    return jax.lax.scan(body, carry, inputs)

################################################################################################################

@partial(jit, static_argnums=(0,))
def nll_loss(apply_fun, params, state, inputs, **kwargs):
    """ Compute the negative mean log likelihood -sum log p(x).

        Args:
            apply_fun - Application function for flow.
            params    - Trainable parameters of flow.
            state     - Non-trainable parameters of flow.
            inputs    - Dictionary containing inputs.
    """
    outputs, updated_state = apply_fun(params, state, inputs, **kwargs)
    loss = outputs.get('log_pz', 0.0) + outputs.get('log_det', 0.0)
    return -jnp.mean(loss), (outputs, updated_state)

################################################################################################################

class GenerativeModel():
    """ Convenience class for training a generative flow model.

        Args:
            flow     - A Flow object.
            clip     - How much to clip gradients.  This is crucial for stable training!
            warmup   - How much to warm up the learning rate.
            lr_decay - Learning rate decay.
            lr       - Max learning rate.
    """
    def __init__(self, flow, loss_fun=None, clip=5.0, warmup=None, lr_decay=1.0, lr=1e-4):
        self.flow = flow
        loss_fun = nll_loss if loss_fun is None else loss_fun
        self.loss_fun = partial(loss_fun, partial(flow.apply, reverse=False))
        self.valgrad = jit(jax.value_and_grad(self.loss_fun, has_aux=True))

        # Optionally use a learning schedule
        if(warmup is None):
            opt_init, opt_update, get_params = optimizers.adam(lr)
        else:
            schedule = partial(util.linear_warmup_lr_schedule, warmup=warmup, lr_decay=lr_decay, lr=lr)
            opt_init, opt_update, get_params = optimizers.adam(schedule)

        # Initialize the optimizer state
        self.opt_update, self.get_params = jit(opt_update), jit(get_params)
        self.opt_state = opt_init(flow.params)

        # Gradient clipping is crucial for tough datasets!
        if(clip is not None):
            self.clip = partial(optimizers.clip_grads, max_norm=clip)
            self.clip = jit(self.clip)
        else:
            self.clip = None

        self.training_steps = 0
        self.losses = []

        self.fast_train = partial(train_loop, self.valgrad, self.opt_update, self.get_params)

    #############################################################################

    @property
    def apply(self):
        return self.flow.apply

    #############################################################################

    @property
    def params(self):
        return self.flow.params

    @params.setter
    def params(self, val):
        self.flow.params = val

    #############################################################################

    @property
    def state(self):
        return self.flow.state

    @state.setter
    def state(self, val):
        self.flow.state = val

    #############################################################################

    def grad_step(self, key, inputs, **kwargs):

        # Compute the gradients
        (loss, (state, outputs)), grad = self.valgrad(self.flow.params, self.flow.state, inputs, key=key, **kwargs)
        self.losses.append(loss)

        # Clip the gradients
        if(self.clip):
            grad = self.clip(grad)

        # Take a grad step
        self.opt_state = self.opt_update(self.training_steps, grad, self.opt_state)
        self.training_steps += 1

        # Update the parameters and state
        self.flow.params = self.get_params(self.opt_state)
        self.flow.state = state

        return loss, outputs

    def multi_grad_step(self, key, x):
        # This function expects doubly batched inputs!!
        assert len(x.shape) - len(self.flow.input_shapes['x']) == 2

        n_iters = x.shape[0]
        iter_numbers = jnp.arange(self.training_steps, self.training_steps + n_iters)
        (params, state, opt_state), train_losses = self.fast_train(self.flow.params, self.flow.state, self.opt_state, key, x, iter_numbers)
        self.losses.extend(list(train_losses))

        self.training_steps += n_iters
        self.flow.params = params
        self.flow.state = state
        self.opt_state = opt_state
        return train_losses

    #############################################################################

    def forward_apply(self, key, inputs):
        outputs, _ = self.apply(self.params, self.state, inputs, key=key)
        return outputs

    #############################################################################

    def sample(self, key, n_samples, full_output=False):
        # dummy_z is a placeholder with the shapes we'll use when we sample in the prior.
        dummy_z = jnp.zeros((n_samples,) + self.flow.output_shapes['x'])
        outputs, _ = self.apply(self.params, self.state, {'x': dummy_z}, key=key, reverse=True, compute_base=True, prior_sample=True)
        return outputs['x'] if full_output == False else outputs

    #############################################################################

    def inverse(self, key, inputs, compute_base=False):
        outputs, _ = self.apply(self.params, self.state, inputs, key=key, reverse=True, compute_base=compute_base)
        return outputs

    #############################################################################

    def save_flow(self, path=None):
        self.flow.save_params_and_state_to_file(path=path)

    def load_flow(self, path=None):
        self.flow.load_param_and_state_from_file(path=path)

    #############################################################################

    def save_training_state(self, path=None):
        assert 0, 'Not implemented'

    def load_training_state(self, path=None):
        assert 0, 'Not implemented'

################################################################################################################

class _ImageMixin():
    """ Helper class to generate image samples easily.

        Args:
    """
    def sample(self, key, n_samples, full_output=False):
        # dummy_z is a placeholder with the shapes we'll use when we sample in the prior.
        dummy_z = jnp.zeros((n_samples,) + self.flow.output_shapes['x'])
        outputs, _ = self.apply(self.params, self.state, {'x': dummy_z}, key=key, reverse=True, compute_base=True, prior_sample=True, generate_image=True)
        return outputs['image'] if full_output == False else outputs

################################################################################################################

class ImageGenerativeModel(_ImageMixin, GenerativeModel):
    """ Generative flow model for images.

        Args:
    """
    pass
