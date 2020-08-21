from functools import partial
import jax.nn.initializers as jaxinit
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
from jax.experimental import optimizers
from nux.training import unsupervised

__all__ = ['classification_loss',
           'SupervisedGenerativeModel',
           'SupervisedImageGenerativeModel']

################################################################################################################

@partial(jit, static_argnums=(0, 1, 2))
def train_body(valgrad, opt_update, get_params, carry, inputs, clip=True):
    params, state, opt_state = carry
    i, key, inputs = inputs

    # Take a gradient step
    (train_loss, (outputs, state)), grad = valgrad(params, state, inputs, beta=0.0, key=key)

    # Get the accuracy
    acc = jnp.mean(outputs['prediction'] == inputs['y'])

    if(clip):
        grad = jit(optimizers.clip_grads)(grad, 5.0)

    # Update the parameters and optimizer state
    opt_state = opt_update(i, grad, opt_state)
    params = get_params(opt_state)

    return (params, state, opt_state), (train_loss, acc)

@partial(jit, static_argnums=(0, 1, 2, 3))
def train_loop(valgrad, opt_update, get_params, batch_size, params, state, opt_state, key, x, y, iter_numbers):
    """ Fast training loop using scan """
    body = partial(train_body, valgrad, opt_update, get_params)

    # Get the inputs for the scan loop
    n_iters = x.shape[0]
    inputs = {'x': x, 'y': y}
    keys = random.split(key, n_iters)

    # Run the optimizer steps
    carry = (params, state, opt_state)
    inputs = (iter_numbers, keys, inputs)
    return jax.lax.scan(body, carry, inputs)

################################################################################################################

@partial(jit, static_argnums=(0,))
def test_body(apply_fun, carry, inputs):
    params, state = carry
    key, inputs = inputs

    # Get the number of correct predictions
    outputs, _ = apply_fun(params, state, inputs, key=key)
    correct = jnp.sum(outputs['prediction'] == inputs['y'])
    return (params, state), correct

@partial(jit, static_argnums=(0,))
def test_loop(apply_fun, params, state, key, x, y):
    """ Fast training loop using scan """
    body = partial(test_body, apply_fun)

    # Reshape the test dataset
    n_iters = x.shape[0]
    inputs = {'x': x, 'y': y}
    keys = random.split(key, n_iters)

    # Run the optimizer steps
    carry = (params, state)
    inputs = (keys, inputs)
    return jax.lax.scan(body, carry, inputs)

################################################################################################################

@partial(jit, static_argnums=(0,))
def classification_loss(apply_fun, params, state, inputs, beta, **kwargs):
    """ Compute the negative mean log joint likelihood -sum log p(y,x).
        Assumes that the flow ends with a softmax!!!!

        Args:
            apply_fun - Application function for flow.
            params    - Trainable parameters of flow.
            state     - Non-trainable parameters of flow.
            inputs    - Dictionary containing inputs.
            beta      - How much to weight the log likelihood contribution.
    """
    # Compute log p(x,y)
    nll, (outputs, updated_state) = unsupervised.nll_loss(apply_fun, params, state, inputs, **kwargs)
    assert 'prediction' in outputs.keys()
    return nll, (outputs, updated_state)

################################################################################################################

class SupervisedGenerativeModel(unsupervised.GenerativeModel):
    """ Convenience class for training a supervised generative flow model.

        Args:
            flow     - A Flow object.
            clip     - How much to clip gradients.  This is crucial for stable training!
            warmup   - How much to warm up the learning rate.
            lr_decay - Learning rate decay.
            lr       - Max learning rate.
            beta      - How much to weight the log likelihood contribution.
    """
    def __init__(self, flow, loss_fun=None, clip=5.0, warmup=None, lr_decay=1.0, lr=1e-4, beta=1.0, batch_size=32):
        super().__init__(flow,
                         clip=clip,
                         warmup=warmup,
                         lr_decay=lr_decay,
                         lr=lr)

        loss_fun = classification_loss if loss_fun is None else loss_fun
        self.loss_fun = partial(loss_fun, partial(flow.apply, reverse=False))
        self.valgrad = jit(jax.value_and_grad(self.loss_fun, has_aux=True))

        # In case we want to weight classification loss more
        self.beta = beta

        self.losses = []
        self.train_accs = []
        self.test_accs = []

        self.fast_train = partial(train_loop, self.valgrad, self.opt_update, self.get_params, batch_size)
        self.fast_test = partial(test_loop, self.flow.apply)

    #############################################################################

    def grad_step(self, key, inputs, beta=None, **kwargs):

        beta = beta if beta is not None else self.beta

        # Take a gradient step
        loss, outputs = super().grad_step(key, inputs, beta=beta, **kwargs)
        self.losses.append(loss)

        # Compute the training accuracy
        train_acc = jnp.mean(outputs['prediction'] == inputs['y'])
        self.train_accs.append(train_acc)

        return loss, train_acc, outputs

    def multi_grad_step(self, key, x, y):
        # This function expects doubly batched inputs!!
        assert len(x.shape) - len(self.flow.input_shapes['x']) == 2
        assert len(y.shape) == 2

        n_iters = x.shape[0]
        iter_numbers = jnp.arange(self.training_steps, self.training_steps + n_iters)
        (params, state, opt_state), (train_losses, train_accs) = self.fast_train(self.flow.params, self.flow.state, self.opt_state, key, x, y, iter_numbers)
        self.losses.extend(list(train_losses))
        self.train_accs.append(train_accs)

        self.training_steps += n_iters
        self.flow.params = params
        self.flow.state = state
        self.opt_state = opt_state
        return train_losses, train_accs

    #############################################################################

    def predict(self, key, x):
        inputs = {'x': x}
        outputs, _ = self.flow.apply(self.flow.params, self.flow.state, inputs, key=key)
        return outputs

    #############################################################################

    def eval_test_set(self, key, x, y):
        """ Compute the full test set accuracy """
        # This function expects doubly batched inputs!!
        assert len(x.shape) - len(self.flow.input_shapes['x']) == 2
        assert len(y.shape) == 2

        _, n_correct = self.fast_test(self.params, self.state, key, x, y)
        total_correct = n_correct.sum()
        test_acc_mean = total_correct/jnp.prod(x.shape[:2])
        self.test_accs.append(test_acc_mean)
        return test_acc_mean

################################################################################################################

class SupervisedImageGenerativeModel(unsupervised._ImageMixin, SupervisedGenerativeModel):
    pass