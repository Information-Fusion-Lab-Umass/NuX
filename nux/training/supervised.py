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
    # Compute log p(x)
    nll, (updated_state, outputs) = unsupervised.nll_loss(apply_fun, params, state, inputs, **kwargs)
    loss = nll

    l2 = jnp.linalg.norm(jax.flatten_util.ravel_pytree(params)[0])

    # Compute log p(y|x)
    labels = inputs['y']
    log_pygx = jnp.log(outputs['x'][jnp.arange(labels.shape[0]), labels])
    loss = nll - beta*jnp.mean(log_pygx)

    return loss + 0.1*l2, (updated_state, outputs)

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
    def __init__(self, flow, loss_fun=None, clip=5.0, warmup=None, lr_decay=1.0, lr=1e-4, beta=1.0):
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

    #############################################################################

    def predict(self, key, x):
        inputs = {'x': x}
        outputs, _ = self.flow.apply(self.flow.params, self.flow.state, inputs, key=key)
        return outputs

    #############################################################################

    def test_acc(self, key, x, y, batch_size=256, pbar_fun=()):
        """ Compute the full test set accuracy.

            Args:
                key        - JAX random key.
                x          - The test data.
                y          - The test labels.
                batch_size - How large the batches should be.
                pbar_fun   - Can pass in a tqdm progress bar.
        """
        n_test = x.shape[0]
        n_correct = 0

        # Partition the test set
        n_chunks = jnp.ceil(n_test//batch_size).astype(jnp.int32)
        indices = jnp.ceil(jnp.linspace(0, 1, n_chunks)*n_test).astype(jnp.int32)
        test_batch_indices = list(zip(indices, indices[1:]))

        # Compute the test accuracy
        for low, high in pbar_fun(test_batch_indices):
            inputs = {'x': x[low:high], 'y': y[low:high]}
            outputs = self.predict(key, inputs['x'])
            correct = jnp.sum(outputs['prediction'] == inputs['y'])
            n_correct += correct

        test_acc_mean = n_correct/n_test
        self.test_accs.append(test_acc_mean)

################################################################################################################

class SupervisedImageGenerativeModel(unsupervised._ImageMixin, SupervisedGenerativeModel):
    pass