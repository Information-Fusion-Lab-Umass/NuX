import jax
import jax.numpy as jnp
import jax.nn.initializers as jaxinit
import nux.util as util
from jax import random, vmap
from functools import partial
import nux.flows.base as base
from jax.scipy.special import logsumexp

@base.auto_batch
def SoftmaxPP(tau=0.4, log_delta=0.0, name='softmax_pp'):
    """
    Softmax++ from IGR paper
    https://arxiv.org/pdf/1912.09588.pdf
    """
    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        x = inputs['x']

        if(reverse == False):
            # Go from x to p
            xp = jnp.hstack([x/tau, log_delta])
            z = jax.nn.softmax(xp)[:-1]

            y = x
        else:
            # Go from p to x
            z = log_delta + jnp.log(x) - jnp.log(1.0 - x.sum())
            z *= tau

            y = z

        Km1 = x.shape[0]
        yt = y/tau
        log_s = logsumexp(jnp.hstack([yt, log_delta]))
        log_det = log_delta - Km1*jnp.log(tau) - (Km1 + 1)*log_s + yt.sum()

        outputs = {'x': z, 'log_det': log_det}
        return outputs, state

    def create_params_and_state(key, input_shapes):
        params, state = {}, {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

@base.auto_batch
def CategoricalClassifier(n_classes, name='categorical_classifier'):

    def apply_fun(params, state, inputs, reverse=False, **kwargs):

        x = inputs['x']

        if(reverse == False):
            # Get the labels
            y = inputs['y']

            # Fill in the last index
            p = jnp.hstack([x, 1.0 - x.sum()])

            # Compute log p(y|p)
            log_det = jnp.log(p[y])
            z = jnp.argmax(p)
        else:
            y = inputs['y']

            p = (y == jnp.arange(n_classes))*7
            key = kwargs.pop('key', None)
            assert key is not None
            p += random.normal(key, p.shape)
            p = jax.nn.softmax(p)
            z = p[:-1]
            log_det = 0.0

        outputs = {'x': z, 'log_det': log_det}
        return outputs, state

    def create_params_and_state(key, input_shapes):
        params, state = {}, {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

@base.auto_batch
def AuxilliarySoftmaxPP(n_classes, tau=1.0, log_delta=0.0, name='softmax_pp'):
    """
    Softmax++ from IGR paper
    https://arxiv.org/pdf/1912.09588.pdf
    """
    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        x = inputs['x']
        x, r = x[:n_classes-1], x[n_classes-1:]

        if(reverse == False):

            # Go from x to p
            xp = jnp.hstack([x/tau, log_delta])
            z = jax.nn.softmax(xp)[:-1]

            y = x
        else:
            # Go from p to x
            z = log_delta + jnp.log(x) - jnp.log(1.0 - x.sum())
            z *= tau

            y = z

        yt = y/tau
        log_s = logsumexp(jnp.hstack([yt, log_delta]))
        log_det = log_delta - (n_classes - 1)*jnp.log(tau) - n_classes*log_s + yt.sum()

        outputs = {'x': jnp.hstack([z, r]), 'log_det': log_det}
        return outputs, state

    def create_params_and_state(key, input_shapes):
        params, state = {}, {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

@base.auto_batch
def AuxilliaryCategoricalClassifier(n_classes, name='aux_categorical_classifier'):

    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        x = inputs['x']
        x, r = x[:n_classes-1], x[n_classes-1:]

        if(reverse == False):
            # Get the labels
            y = inputs['y']

            # Fill in the last index
            p = jnp.hstack([x, 1.0 - x.sum()])

            # Compute log p(y|p)
            log_det = jnp.log(p[y])
            pred = jnp.argmax(p)

            # Compute the likelihood of the auxiliary vector
            log_det += -0.5*jnp.sum(r**2, axis=-1) + -0.5*r.shape[0]*jnp.log(2*jnp.pi)

            z = jnp.hstack([x, r])
        else:
            # Generate the one hot vector
            y = inputs['y']
            p = (y == jnp.arange(n_classes))*7
            p = jax.nn.softmax(p)
            z = jnp.hstack([p[:-1], r])

            log_det = 0.0
            pred = y

        outputs = {'x': z, 'log_det': log_det, 'pred': pred}
        return outputs, state

    def create_params_and_state(key, input_shapes):
        params, state = {}, {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)


__all__ = ['SoftmaxPP',
           'CategoricalClassifier',
           'AuxilliarySoftmaxPP',
           'AuxilliaryCategoricalClassifier']