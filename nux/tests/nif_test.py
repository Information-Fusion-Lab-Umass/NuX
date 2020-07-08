import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.flatten_util import ravel_pytree
from functools import partial
import nux.util as util
import jax.tree_util as tree_util
from jax.scipy.special import logsumexp
import nux.flows as nux

def injective_flow_test(layer, inputs, key):
    # language=rst
    """
    Check that the injective flow is valid
    """
    init_fun = layer
    input_shapes = util.tree_shapes(inputs)

    # Initialize the flow
    inputs_batched = tree_util.tree_map(lambda x: jnp.broadcast_to(x[None], (8,) + x.shape), inputs)
    _, flow = init_fun(key, inputs, batched=False)
    _, flow_batched = init_fun(key, inputs_batched, batched=True)

    # Ensure that the parameters are the same
    params_structure, state_structure = tree_util.tree_structure(flow.params), tree_util.tree_structure(flow.state)
    params_structure_batched, state_structure_batched = tree_util.tree_structure(flow_batched.params), tree_util.tree_structure(flow_batched.state)

    assert params_structure == params_structure_batched
    assert state_structure == state_structure_batched
    print('Passed parameter and state construction tests')

    # For the reconstruction tests, need a value thats on the manifold
    # Make sure the reconstructions are correct
    outputs, _ = flow.apply(flow.params, flow.state, inputs, test=util.TEST, key=None)
    reconstr, _ = flow.apply(flow.params, flow.state, outputs, reverse=True, test=util.TEST, key=None)
    inputs_on_manifold = {'x': reconstr['x']}
    inputs_on_manifold_batched = tree_util.tree_map(lambda x: jnp.broadcast_to(x[None], (8,) + x.shape), inputs_on_manifold)

    # Make sure the reconstructions are correct
    outputs, _ = flow.apply(flow.params, flow.state, inputs_on_manifold, test=util.TEST, key=None)
    reconstr, _ = flow.apply(flow.params, flow.state, outputs, reverse=True, test=util.TEST, key=None)

    assert jnp.allclose(inputs_on_manifold['x'], reconstr['x'], atol=1e-04)
    assert jnp.allclose(outputs['log_det'], reconstr['log_det'], atol=1e-04)
    print('Passed reconstruction tests')

    # Make sure the batched reconstructions are correct
    batched_outputs, _ = flow.apply(flow.params, flow.state, inputs_on_manifold_batched, test=util.TEST, key=None)
    batched_reconstr, _ = flow.apply(flow.params, flow.state, batched_outputs, reverse=True, test=util.TEST, key=None)

    assert jnp.allclose(inputs_on_manifold_batched['x'], batched_reconstr['x'], atol=1e-04)
    assert jnp.allclose(batched_outputs['log_det'], batched_reconstr['log_det'], atol=1e-04)
    print('Passed batched reconstruction tests')

################################################################################################################

def noisy_injective_flow_test(layer, inputs, key, n_keys=256, n_z=256):
    # language=rst
    """
    Check the we can estimate the marginal correctly.
    log p(x) = log E_{p(z)}[p(x|z)] = log E_{q(z|x)}[p(z)] + log int p(x|z)dz
    """
    # Initialize the nif
    outputs, flow = layer(key, inputs, batched=False)

    # Check that log q(z|x) = log p(x|z) - log int p(x|z)dz
    z = outputs['x']
    log_hx = outputs['log_det']
    log_pxgz = outputs['log_pxgz']
    log_qzgx = outputs['log_qzgx']
    assert jnp.allclose(log_qzgx, log_pxgz - log_hx)

    # Get an esimate of E_{p(z)}[p(x|z)]
    def inverse_estimate(zs, key):
        inpts = {'x': zs, 'target_x': inputs['x']}

        fun = partial(flow.apply, flow.params, flow.state, key=None, reverse=True, forward_monte_carlo=True)
        outputs, _ = vmap(fun, in_axes=({'x': 0, 'target_x': None},))(inpts)

        return logsumexp(outputs['log_pxgz']) - jnp.log(outputs['log_pxgz'].size)

    keys = random.split(key, n_keys)
    zs = random.normal(key, (n_keys, n_z) + flow.output_shapes['x'])
    inverse_estimates = vmap(inverse_estimate)(zs, keys)
    inverse_mean, inverse_std = inverse_estimates.mean(), inverse_estimates.std()
    print('log E_{p(z)}[p(x|z)] ~= %5.3f +- %5.3f'%(inverse_mean, inverse_std))

    # Get an estimate of log E_{q(z|x)}[p(z)] + log int p(x|z)dz
    def forward_estimate(key, n_importance_samples=n_z):
        def fun(key):
            outputs, _ = flow.apply(flow.params, flow.state, inputs, key=key)
            return outputs

        keys = random.split(key, n_importance_samples)
        outputs = vmap(fun)(keys)
        return logsumexp(outputs['log_pz'] + outputs['log_det']) - jnp.log(outputs['log_det'].size)

    keys = random.split(key, n_keys)
    forward_estimates = vmap(forward_estimate)(keys)
    forward_mean, forward_std = forward_estimates.mean(), forward_estimates.std()
    print('log E_{q(z|x)}[p(z)] + log int p(x|z)dz ~= %5.3f +- %5.3f'%(forward_mean, forward_std))

################################################################################################################

def nif_test():
    # language=rst
    """
    Check that a prior is correct and works with MCMC
    """
    key = random.PRNGKey(0)
    x = random.normal(key, (4, 4, 2))
    # x = jax.nn.softmax(x)
    # x = random.normal(key, (3,))
    inputs = {'x': x}

    # flow = nux.sequential(nux.TallAffineDiagCov(2),
    #                       nux.UnitGaussianPrior())

    # flow = nux.sequential(nux.CouplingTallAffineDiagCov(2, kind='blah', hidden_layer_sizes=[16]),
    #                       nux.UnitGaussianPrior())

    # flow = nux.sequential(nux.UpSample(),
    #                       nux.UnitGaussianPrior())


    flow = nux.sequential(nux.CouplingUpSample(n_channels=2),
                          nux.UnitGaussianPrior())


    # injective_flow_test(flow, inputs, key)
    noisy_injective_flow_test(flow, inputs, key)








    # # x = random.normal(key, (10,))
    # x = random.normal(key, (5, 10))
    # inputs = {'x': x}

    # layers = [nux.Coupling(), nux.ActNorm(), nux.Reverse()]*1
    # prior = nux.sequential(*layers, nux.UnitGaussianPrior())
    # flow = nux.importance_weighted(nux.TallAffineDiagCov(6), prior)

    # flow = nux.sequential(nux.Debug('c'),
    #                       flow,
    #                       nux.Debug('i'))

    # outputs, flow = flow(key, inputs, batched=True)

    # outputs, _ = flow.apply(flow.params, flow.state, inputs, key=key)
    # flow.apply(flow.params, flow.state, outputs, key=key, reverse=True)
