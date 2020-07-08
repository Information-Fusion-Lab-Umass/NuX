import jax.nn.initializers as jaxinit
import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from functools import partial, reduce
from jax.tree_util import tree_map
from jax.scipy.special import logsumexp
import nux.util as util
import nux.flows.base as base

@base.auto_batch
def importance_weighted(nif, prior, n_importance_samples=64, name='importance_weight'):
    # language=rst
    """
    Use importance weighting for gradient estimation.
    """
    nif_apply, prior_apply = None, None

    def _forward(nif_outputs, prior_outputs, batch_depth=1):

        prior_outputs['log_px'] = prior_outputs['log_pz'] + prior_outputs['log_det']

        # Compute the importance weights
        assert prior_outputs['log_px'].ndim == 1
        w = prior_outputs['log_px'] - logsumexp(prior_outputs['log_px'])
        w = jnp.exp(w)

        # Don't pass a gradient through this!
        w = jax.lax.stop_gradient(w)

        # Compute the objective.  This is an estimate of log p(x) - H[p(z|x)]
        assert nif_outputs['log_pxgz'].ndim == 1
        objective = jnp.sum(w*(nif_outputs['log_pxgz'] + prior_outputs['log_px']))

        # Compute the upper bound on the log likelihood log_p(x) + KL[p(z|x)||q(z|x)]
        assert nif_outputs['log_qzgx'].ndim == 1
        log_px_upper_bound = jnp.sum(w*prior_outputs['log_px']) + jnp.mean(nif_outputs['log_det'])

        # Compute the lower bound on the log likelihood log_p(x) - KL[q(z|x)||p(z|x)]
        assert nif_outputs['log_det'].ndim == 1
        log_px_lower_bound = jnp.mean(prior_outputs['log_px'] + nif_outputs['log_det'])

        # Compute the estimate of the log likelihood
        log_px_estimate = logsumexp(prior_outputs['log_px']) - jnp.log(n_importance_samples)
        log_px_estimate += jnp.mean(nif_outputs['log_det']) # This should have the same value in every index!

        # Compute the Jensen-Shannon divergence estimate
        js_div = 0.5*(log_px_upper_bound - log_px_lower_bound)

        # To make things consistent, just use the average of z
        zs = prior_outputs['x']
        z = jnp.mean(zs, axis=0)

        outputs = {'x': z,
                   'log_det': objective,
                   'log_px_est': log_px_estimate,
                   'log_px_upper_est': log_px_upper_bound,
                   'log_px_lower': log_px_lower_bound,
                   'js_div': js_div}

        return outputs

    def wrapped_nif(params, state, inputs, reverse, key, **kwargs):
        nif_outputs, updated_nif_state = nif_apply(params['nif'], state['nif'], inputs, reverse=False, key=key, **kwargs)
        return nif_outputs, updated_nif_state

    def forward(params, state, inputs, key=None, **kwargs):
        assert key is not None
        k1, k2 = random.split(key, 2)

        # Sample a bunch from the stochastic inverse
        keys = random.split(k1, n_importance_samples)
        nif_outputs, updated_nif_state = vmap(partial(wrapped_nif, params, state, inputs, False, **kwargs))(keys)
        updated_nif_state = jax.tree_util.tree_map(lambda x: x.mean(axis=0), updated_nif_state)

        # Pass the samples through the prior
        prior_outputs, updated_prior_state = prior_apply(params['prior'], state['prior'], nif_outputs, reverse=False, key=k2, **kwargs)

        outputs = _forward(nif_outputs, prior_outputs)

        updated_state = {'nif': updated_nif_state, 'prior': updated_prior_state}
        return outputs, updated_state

    def inverse(params, state, inputs, key=None, **kwargs):
        k1, k2 = random.split(key, 2) if key is not None else (None, None)
        outputs, updated_prior_state = prior_apply(params['prior'], state['prior'], inputs, reverse=True, key=k1, **kwargs)
        outputs, updated_nif_state = nif_apply(params['nif'], state['nif'], outputs, reverse=True, key=k2, **kwargs)

        updated_state = {'nif': updated_nif_state, 'prior': updated_prior_state}
        return outputs, updated_state

    def apply_fun(params, state, inputs, reverse=False, **kwargs):
        if(reverse == False):
            return forward(params, state, inputs, **kwargs)
        return inverse(params, state, inputs, **kwargs)

    def init_fun(key, inputs, batched=False, batch_depth=1, **kwargs):
        k1, k2 = random.split(key, 2)

        # Add an extra value to batch depth for the importance samples
        if(batched == True):
            batch_depth += 1
        else:
            batched = True
            batch_depth = 1

        # We can't vmap on the key because of some jax issues, so instead use auto-batching
        if(batched == True):
            batched_inputs = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[:,None,...], (x.shape[0], n_importance_samples) + x.shape[1:]), inputs)
        else:
            batched_inputs = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n_importance_samples,) + x.shape), inputs)

        # Pass the inputs through the nif and the prior
        nif_outputs, ni_flow = nif(k1, batched_inputs, batched=batched, batch_depth=batch_depth, **kwargs)
        prior_outputs, prior_flow = prior(k2, nif_outputs, batched=batched, batch_depth=batch_depth, **kwargs)

        # Pass the inputs to forward
        vmapped_fun = _forward
        for i in range(batch_depth - 1):
            vmapped_fun = vmap(vmapped_fun)
        outputs = vmapped_fun(nif_outputs, prior_outputs)

        # Compile the parameters
        nonlocal nif_apply, prior_apply
        nif_apply, prior_apply = ni_flow.apply, prior_flow.apply

        params = {'nif': ni_flow.params, 'prior': prior_flow.params}
        state = {'nif': ni_flow.state, 'prior': prior_flow.state}

        return outputs, base.Flow(name, ni_flow.input_shapes, prior_flow.output_shapes, ni_flow.input_ndims, prior_flow.output_ndims, params, state, apply_fun)

    return init_fun

__all__ = ['importance_weighted']
