from functools import partial
import jax.nn.initializers as jaxinit
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
from jax.experimental import optimizers
import nux.util as util
from nux.training.base import Trainer

__all__ = ['nll_loss',
           'GenerativeModel',
           'MultiLossGenerativeModel',
           'ImageGenerativeModel',
           'MultiLossImageGenerativeModel',
           '_ImageMixin']

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
        loss_fun = partial(loss_fun, self.flow.apply)
        self.trainer = Trainer(loss_fun, self.flow.params, clip=clip, warmup=warmup, lr_decay=lr_decay, lr=lr)

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
        loss, outputs, params, state = self.trainer.grad_step(key, inputs, self.params, self.state, **kwargs)
        self.flow.params = params
        self.flow.state = state
        return loss, outputs

    def multi_grad_step(self, key, inputs):
        x = inputs['x']

        # This function expects doubly batched inputs!!
        assert len(x.shape) - len(self.flow.input_shapes['x']) == 2

        (train_losses, outputs), params, state = self.trainer.multi_grad_step(key, inputs, self.flow.params, self.flow.state)
        self.flow.params = params
        self.flow.state = state
        return train_losses, outputs

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

class MultiLossGenerativeModel(GenerativeModel):
    """ Helper class in case we want multiple training objectives

        Args:
    """
    def __init__(self, flow, loss_funs=None, **kwargs):
        self.flow = flow

        # Take each of the loss functions
        assert loss_funs is not None
        assert isinstance(loss_funs, dict)

        fill = lambda x: partial(x, self.flow.apply)
        self.trainers = dict([(key, Trainer(fill(fun), self.flow.params, **kwargs)) for key, fun in loss_funs.items()])

    def grad_step(self, trainer_key, key, inputs, **kwargs):
        loss, outputs, params, state = self.trainers[trainer_key].grad_step(key, inputs, self.params, self.state, **kwargs)
        self.flow.params = params
        self.flow.state = state
        return loss, outputs

    def multi_grad_step(self, trainer_key, key, inputs):
        x = inputs['x']

        # This function expects doubly batched inputs!!
        assert len(x.shape) - len(self.flow.input_shapes['x']) == 2

        (train_losses, outputs), params, state = self.trainers[trainer_key].multi_grad_step(key, inputs, self.flow.params, self.flow.state)
        self.flow.params = params
        self.flow.state = state
        return train_losses, outputs

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

    def sample2(self, key, n_samples, full_output=False):
        # dummy_z is a placeholder with the shapes we'll use when we sample in the prior.
        z = random.normal(key, (n_samples,) + self.flow.output_shapes['x'])
        outputs, _ = self.apply(self.params, self.state, {'x': z}, reverse=True, compute_base=True, generate_image=True)
        return outputs['image'] if full_output == False else outputs

################################################################################################################

class ImageGenerativeModel(_ImageMixin, GenerativeModel):
    """ Generative flow model for images.

        Args:
    """
    pass

class MultiLossImageGenerativeModel(_ImageMixin, MultiLossGenerativeModel):
    """ Generative flow model for images.

        Args:
    """
    pass
