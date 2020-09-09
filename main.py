import nux.flows as flows
from nux.tests.nf_test import standard_layer_tests, image_layer_test, unit_test, flow_test
from nux.tests.nif_test import nif_test
import jax.numpy as jnp
from debug import *

import nux.flows as nux
import jax
from jax import jit, vmap, random
from functools import partial
import nux.util as util

from jax.scipy.special import logsumexp

# from nux.flows.surjective.affine import TallDense
# from nux.flows.surjective.softmax import SoftmaxSurjection

import haiku as hk

if(__name__ == '__main__'):

    from nux.flows.injective.manifold_affine import TallMVP

    key = random.PRNGKey(0)
    data = random.normal(key, (10, 4))
    flow_init = TallMVP(2)

    inputs = {'x': data}
    outputs, flow = flow_init(key, inputs, batched=True)

    if(False):

        from jax import random, grad, vmap
        from jax.tree_util import tree_map

        def fun(x, d):
            return (x*d['x']).sum()

        ####################################

        @jax.custom_vjp
        def custom_fun(x, d):
            return (x*d['x']).sum()

        def forward(x, d):
            return (x*d['x']).sum(), (d['x'], {'x': x})

        def backward(res, g):
            chain_rule = lambda dx: tree_map(lambda x: x*g, dx)
            return chain_rule(res[0]), chain_rule(res[1])

        custom_fun.defvjp(forward, backward)

        ####################################

        def loss_fun(x, d):
            return fun(x, d).sum()

        def custom_loss_fun(x, d):
            return custom_fun(x, d).sum()

        def vmapped_loss_fun(x, d):
            return vmap(fun)(x, d).sum()

        def vmapped_custom_loss_fun(x, d):
            return vmap(custom_fun)(x, d).sum()

        ####################################

        x = jnp.ones((10, 3))
        d = {'x': jnp.ones((10, 3))}
        scan_range = jnp.arange(4)

        ####################################

        def scan_body(f, carry, inputs):
            x, d = carry
            return carry, f(x, d)

        scan_apply = lambda f: jax.lax.scan(partial(scan_body, f), (x, d), scan_range)

        ####################################

        # Loss function - Pass
        loss_fun(x, d)
        custom_loss_fun(x, d)

        # Grad of loss function - Pass
        jax.grad(loss_fun)(x, d)
        jax.grad(custom_loss_fun)(x, d)

        # Vmapped loss function - Pass
        vmapped_loss_fun(x, d)
        vmapped_custom_loss_fun(x, d)

        # Grad of vmapped loss function - Pass
        jax.grad(vmapped_loss_fun)(x, d)
        jax.grad(vmapped_custom_loss_fun)(x, d)

        # Scanned loss function - Pass
        scan_apply(loss_fun)
        scan_apply(custom_loss_fun)

        # Scanned grad of loss function - Pass
        scan_apply(jax.grad(loss_fun))
        scan_apply(jax.grad(custom_loss_fun))

        # Scanned vmapped loss function - Pass
        scan_apply(vmapped_loss_fun)
        scan_apply(vmapped_custom_loss_fun)

        # Scanned grad of vmapped loss function

        # Pass
        scan_apply(jax.grad(vmapped_loss_fun))

        # Fails!!!
        scan_apply(jax.grad(vmapped_custom_loss_fun))

        assert 0

    ################################################################################

    # standard_layer_tests()
    # image_layer_test()
    # nif_test()

    import nux.flows.bijective.residual as residual
    import nux.flows.spectral_norm as sn
    key = random.PRNGKey(0)

    data = random.normal(key, (10, 2))
    # flow_init = nux.ResidualFlow()

    # inputs = {'x': data}
    # outputs, flow = flow_init(key, inputs, batched=True)

    # reconstr, _ = flow.apply(flow.params, flow.state, outputs, reverse=True, key=key)

    x = data[0]

    if(False):
        residual_network = hk.transform(lambda x, **kwargs: util.SimpleMLP(x.shape, hidden_layer_sizes=[4], is_additive=True)(x, **kwargs), apply_rng=True)
        residual_params = residual_network.init(key, x)

        key_tree = util.key_tree_like(key, residual_params)
        residual_params = jax.tree_util.tree_multimap(lambda key, x: 4*random.normal(key, x.shape), key_tree, residual_params)

        u_tree = sn.initialize_spectral_norm_u_tree(key, residual_params)

        # Spectral normaliation
        residual_params, u_tree = sn.spectral_norm_tree(residual_params, u_tree, 0.9, 50)
        norms = sn.check_spectral_norm(residual_params)

        # Compute the true jacobian
        def true_jacobian(x):
            gx = residual_network.apply(residual_params, None, x)
            return x + gx
        J = jax.jacobian(true_jacobian)(x)

        # Run the network
        filled_residual = partial(residual_network.apply, residual_params, None)
        gx, residual_vjp = jax.vjp(filled_residual, x)

        roulette_key, trace_key = random.split(key, 2)

    ################################################################################
    """ Test the log det estimate """
    if(False):
        # roulette_keys = random.split(roulette_key, 10)
        # trace_keys = random.split(trace_key, 10)
        # fun = partial(residual.log_det_estimate, residual_vjp, x.shape)
        # log_det = vmap(vmap(fun, in_axes=(0, None)), in_axes=(None, 0))(trace_keys, roulette_keys)

        # trace_keys = random.split(trace_key, 10000)
        # fun = partial(residual.log_det_estimate, residual_vjp, x.shape)
        # log_det, (v, summed_vjp) = vmap(fun, in_axes=(0, None))(trace_keys, roulette_key)

        log_det, (v, summed_vjp) = residual.log_det_estimate(residual_vjp, x.shape, trace_key, roulette_key)

        true_logdet = jnp.linalg.slogdet(J)[1]

    ################################################################################
    """ Test the log det gradient estimate """
    if(False):
        def true_jacobian_grad(residual_params, x):
            def true_jacobian(x):
                gx = residual_network.apply(residual_params, None, x)
                return x + gx
            J = jax.jacobian(true_jacobian)(x)
            return jnp.linalg.slogdet(J)[1]

        true_dLdtheta, true_dLdx = jax.grad(true_jacobian_grad, argnums=(0, 1))(residual_params, x)

        @jit
        def estimate_grads(key):

            def compare(residual_params, x):
                return residual.residual_flow_logdet(residual_network, key, residual_params, x)

            dLdtheta, dLdx = jax.grad(compare, argnums=(0, 1))(residual_params, x)
            return dLdtheta, dLdx

        keys = random.split(key, 10000)
        dLdtheta, dLdx = vmap(estimate_grads)(keys)

        dLdtheta_mean = jax.tree_util.tree_map(lambda x: x.mean(axis=0), dLdtheta)
        dLdx_mean = jax.tree_util.tree_map(lambda x: x.mean(axis=0), dLdx)

        error = jax.tree_util.tree_multimap(lambda x, y: jnp.linalg.norm(x - y), true_dLdtheta, dLdtheta_mean)

    ################################################################################
    """ Check that we can actually maximize the log det with the grad estimate """

    if(False):
        def loss_fun(residual_params):
            log_det = residual.residual_flow_logdet(residual_network, key, residual_params, x)
            return log_det**2

        def train_body(valgrad, carry, inputs):
            key = inputs
            residual_params = carry

            log_det, dLdtheta = valgrad(residual_params)
            residual_params = jax.tree_util.tree_multimap(lambda x, g: x + 0.001*g, residual_params, dLdtheta)

            return residual_params, log_det

        valgrad = jax.value_and_grad(loss_fun)
        valgrad = jit(valgrad)

        keys = random.split(key, 200)
        residual_params, log_dets = jax.lax.scan(partial(train_body, valgrad), residual_params, keys)

    ################################################################################

    if(False):
        data = random.normal(key, (10, 2))
        flow_init = nux.ResidualFlow(hidden_layer_sizes=[2])

        inputs = {'x': data}
        outputs, flow = flow_init(key, inputs, batched=True)

        def loss_fun(params):
            outputs, _ = flow.apply(params, flow.state, inputs, key=key)
            return outputs['log_det'].sum()

        def train_body(valgrad, carry, inputs):
            key = inputs
            params = carry

            log_det, dLdtheta = valgrad(params)
            # params = jax.tree_util.tree_multimap(lambda x, g: x + 0.001*g, params, dLdtheta)

            return params, log_det

        valgrad = jax.value_and_grad(loss_fun)
        # valgrad = jit(valgrad)

        keys = random.split(key, 200)
        params, log_dets = jax.lax.scan(partial(train_body, valgrad), flow.params, keys)

        assert 0