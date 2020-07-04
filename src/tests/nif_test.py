import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.flatten_util import ravel_pytree
from functools import partial
import src.util as util
import jax.tree_util as tree_util
from jax.scipy.special import logsumexp
import src.flows as nux

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
    inputs_on_manifold = reconstr
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

def noisy_injective_flow_test(layer, inputs, key, n_keys=128, n_z=128):
    # language=rst
    """
    Check the we can estimate the marginal correctly.
    log p(x) = log E_{p(z)}[p(x|z)] = log E_{q(z|x)}[p(z)] + log int p(x|z)dz
    """
    # Initialize the nif
    _, flow = layer(key, inputs, batched=False)

    # Get an esimate of E_{p(z)}[p(x|z)]
    def inverse_estimate(zs, key):
        # outputs, _ = flow.apply(flow.params, flow.state, {'x': zs}, target=inputs['x'], compute_likelihood=True)
        outputs, _ = flow.apply(flow.params, flow.state, {'x': zs}, key=None, reverse=True, injected_x=inputs['x'])
        return logsumexp(outputs['log_det']) - jnp.log(outputs['log_det'].shape[0])

    keys = random.split(key, n_keys)
    zs = random.normal(key, (n_keys, n_z) + flow.output_shapes['x'])
    inverse_estimates = vmap(inverse_estimate)(zs, keys)
    inverse_mean, inverse_std = inverse_estimates.mean(), inverse_estimates.std()
    print('%5.3f +- %5.3f'%(inverse_mean, inverse_std))

    # Get an estimate of log E_{q(z|x)}[p(z)] + log int p(x|z)dz
    def forward_estimate(key):
        outputs, _ = flow.apply(flow.params, flow.state, inputs, n_importance_samples=32, key=key)
        return logsumexp(outputs['log_det']) - jnp.log(outputs['log_det'].shape[0])

    keys = random.split(key, n_keys*n_z)
    forward_estimates = vmap(forward_estimate)(keys)
    forward_mean, forward_std = forward_estimates.mean(), forward_estimates.std()
    print('%5.3f +- %5.3f'%(forward_mean, forward_std))

################################################################################################################

def nif_test():
    # language=rst
    """
    Check that a prior is correct and works with MCMC
    """
    key = random.PRNGKey(0)
    # x = random.normal(key, (3, 4, 6, 8))
    # x = jax.nn.softmax(x)
    x = random.normal(key, (5, 10))
    inputs = {'x': x}

    # flow = nux.sequential(nux.TallAffineDiagCov(2),
    #                       nux.Debug(),
    #                       nux.UnitGaussianPrior())

    # # injective_flow_test(flow, inputs, key)
    # noisy_injective_flow_test(flow, inputs, key)


    layers = [nux.Coupling(), nux.ActNorm(), nux.Reverse()]*1
    prior = nux.sequential(*layers, nux.UnitGaussianPrior())
    flow = nux.importance_weighted(nux.TallAffineDiagCov(6), prior)

    flow = nux.sequential(nux.Debug('c'),
                          flow,
                          nux.Debug('i'))

    outputs, flow = flow(key, inputs, batched=True)

    outputs, _ = flow.apply(flow.params, flow.state, inputs, key=key)
    flow.apply(flow.params, flow.state, outputs, key=key, reverse=True)
