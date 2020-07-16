import jax
import jax.numpy as jnp
import jax.nn.initializers as jaxinit
import nux.util as util
from jax import random, vmap
from functools import partial
import nux.flows
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
        log_det = log_delta - Km1*jnp.log(tau) - (Km1 + 1)*log_s + yt.sum() # This is correct.

        outputs = {'x': z, 'log_det': log_det}
        return outputs, state

    def create_params_and_state(key, input_shapes):
        params, state = {}, {}
        return params, state

    return base.initialize(name, apply_fun, create_params_and_state)

__all__ = ['SoftmaxPP']