import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree
from functools import partial
import nux.util as util
import jax.tree_util as tree_util

import nux.flows.bijective.affine as affine
import nux.flows.bijective.nonlinearities as nonlinearities
import nux.flows.bijective.reshape as reshape
import nux.flows.bijective.normalization as normalization
import nux.flows.bijective.conv as conv
import nux.flows.compose as compose
import nux.flows.bijective.maf as maf
import nux.flows.bijective.coupling as coupling
import nux.flows.bijective.spline as spline
import nux.flows.base as base
import nux.flows.bijective.igr as igr

def flow_test(layer, inputs, key):
    # language=rst
    """
    Test if a flow implementation is correct.  Checks if the forward and inverse functions are consistent and
    compares the jacobian determinant calculation against an autograd calculation.
    """
    init_fun = layer
    input_shapes = util.tree_shapes(inputs)

    # Initialize the flow
    inputs_batched = tree_util.tree_map(lambda x: jnp.broadcast_to(x[None], (3,) + x.shape), inputs)
    inputs_doubly_batched = tree_util.tree_map(lambda x: jnp.broadcast_to(x[None], (3,) + x.shape), inputs_batched)
    _, flow = init_fun(key, inputs, batched=False)
    _, flow_batched = init_fun(key, inputs_batched, batched=True)
    _, flow_doubly_batched = init_fun(key, inputs_doubly_batched, batched=True, batch_depth=2)

    # Ensure that the parameters are the same
    params_structure, state_structure = tree_util.tree_structure(flow.params), tree_util.tree_structure(flow.state)
    params_structure_batched, state_structure_batched = tree_util.tree_structure(flow_batched.params), tree_util.tree_structure(flow_batched.state)
    params_structure_doubly_batched, state_structure_doubly_batched = tree_util.tree_structure(flow_doubly_batched.params), tree_util.tree_structure(flow_doubly_batched.state)

    assert params_structure == params_structure_batched
    assert state_structure == state_structure_batched
    assert params_structure == params_structure_doubly_batched
    assert state_structure == state_structure_doubly_batched
    print('Passed parameter and state construction tests')

    # Make sure the reconstructions are correct
    outputs, _ = flow.apply(flow.params, flow.state, inputs, test=util.TEST)
    reconstr, _ = flow.apply(flow.params, flow.state, outputs, reverse=True, test=util.TEST)

    assert jnp.allclose(inputs['x'], reconstr['x'], atol=1e-04)
    assert jnp.allclose(outputs['log_det'], reconstr['log_det'], atol=1e-04)
    print('Passed reconstruction tests')

    # Make sure the batched reconstructions are correct
    batched_outputs, _ = flow.apply(flow.params, flow.state, inputs_batched, test=util.TEST)
    batched_reconstr, _ = flow.apply(flow.params, flow.state, batched_outputs, reverse=True, test=util.TEST)

    assert jnp.allclose(inputs_batched['x'], batched_reconstr['x'], atol=1e-04)
    assert jnp.allclose(batched_outputs['log_det'], batched_reconstr['log_det'], atol=1e-04)
    print('Passed batched reconstruction tests')

    # Make sure the doubly batched reconstructions are correct
    doubly_batched_outputs, _ = flow.apply(flow.params, flow.state, inputs_doubly_batched, test=util.TEST)
    doubly_batched_reconstr, _ = flow.apply(flow.params, flow.state, doubly_batched_outputs, reverse=True, test=util.TEST)

    assert jnp.allclose(inputs_doubly_batched['x'], doubly_batched_reconstr['x'], atol=1e-04)
    assert jnp.allclose(doubly_batched_outputs['log_det'], doubly_batched_reconstr['log_det'], atol=1e-04)
    print('Passed doubly batched reconstruction tests')

    # Make sure that the log det terms are correct
    def z_from_x(unflatten, x_flat):
        x = unflatten(x_flat)
        outputs, _ = flow.apply(flow.params, flow.state, {'x': x}, test=util.TEST)
        return ravel_pytree(outputs['x'])[0]

    def single_elt_logdet(x):
        x_flat, unflatten = ravel_pytree(x)
        jac = jax.jacobian(partial(z_from_x, unflatten))(x_flat)
        return 0.5*jnp.linalg.slogdet(jac.T@jac)[1]

    actual_log_det = single_elt_logdet(inputs['x'])
    assert jnp.allclose(actual_log_det, outputs['log_det'], atol=1e-04), 'actual_log_det: %5.3f, outputs["log_det"]: %5.3f'%(actual_log_det, outputs['log_det'])
    print('Passed log det tests')

def standard_layer_tests():
    layers = [affine.AffineLDU,
              partial(affine.AffineSVD, n_householders=4),
              affine.AffineDense,
              affine.Affine,
              nonlinearities.LeakyReLU,
              nonlinearities.Sigmoid,
              nonlinearities.Logit,
              reshape.Reverse,
              normalization.ActNorm,
              # normalization.BatchNorm,
              affine.Identity,
              partial(maf.MAF, [1024, 1024]),
              coupling.Coupling,
              partial(spline.NeuralSpline, 4)]

    key = random.PRNGKey(0)
    x = random.normal(key, (10,))
    x = jax.nn.softmax(x)
    inputs = {'x': x}

    for layer in layers:
        print()
        print(layer)
        flow_test(layer(), inputs, key)

def image_layer_test():
    layers = [affine.OnebyOneConvLDU,
              affine.OnebyOneConv,
              affine.OnebyOneConvLAX,
              affine.LocalDense,
              partial(conv.CircularConv, filter_size=(2, 2)),
              reshape.Squeeze,
              reshape.UnSqueeze,
              reshape.Flatten,
              partial(reshape.Transpose, axis_order=(1, 0, 2)),
              partial(reshape.Reshape, shape=(2, -1)),
              coupling.Coupling]

    key = random.PRNGKey(0)
    x_img = random.normal(key, (6, 2, 4))
    inputs = {'x': x_img}

    for layer in layers:
        print()
        print(layer)
        flow_test(layer(), inputs, key)

def unit_test():
    key = random.PRNGKey(0)
    # x = random.normal(key, (4, 6, 8))
    x = random.normal(key, (10,))
    x = jax.nn.softmax(x)[:-1]
    inputs = {'x': x}

    # flow = affine.AffineDense()
    # flow = normalization.ActNorm()

    # flow = compose.sequential(affine.AffineDense())
    # flow = compose.sequential(normalization.ActNorm())

    # flow = compose.sequential(affine.AffineDense(),
    #                   normalization.ActNorm())

    # flow = compose.sequential(normalization.ActNorm(),
    #                   affine.AffineDense())

    # for flow in [flow1, flow2, flow3, flow4, flow5, flow6]:
    #     flow_test(flow, inputs, key)

    # flow = compose.sequential(compose.ChainRule(2, factor=True),
    #                   base.Debug('a'),
    #                   compose.ChainRule(2, factor=False))

    # flow = compose.sequential(compose.ChainRule(2, factor=True),
    #                   compose.ChainRule(2, factor=False),
    #                   compose.sequential(affine.AffineDense(),
    #                              normalization.ActNorm()),
    #                   compose.sequential(affine.AffineDense(),
    #                              normalization.ActNorm()))

    # flow = compose.sequential(compose.ChainRule(2, factor=True),
    #                   compose.factored(base.Debug('a'),
    #                            base.Debug('b')),
    #                   compose.ChainRule(2, factor=False))

    # flow = compose.sequential(compose.ChainRule(2, factor=True),
    #                   compose.factored(affine.AffineDense(),
    #                            affine.AffineDense()),
    #                   compose.ChainRule(2, factor=False))

    # flow = compose.sequential(compose.ChainRule(2, factor=True),
    #                           compose.factored(normalization.ActNorm(),
    #                                            affine.AffineDense()),
    #                           compose.ChainRule(2, factor=False))

    # flow = compose.sequential(compose.ChainRule(2, factor=True),
    #                           compose.factored(normalization.ActNorm(),
    #                                            normalization.ActNorm()),
    #                           compose.ChainRule(2, factor=False))

    # flow = compose.sequential(compose.ChainRule(2, factor=True),
    #                           compose.factored(compose.sequential(base.Debug('a'),
    #                                                               compose.ChainRule(2, factor=True),
    #                                                               compose.factored(base.Debug('b'),
    #                                                                                normalization.ActNorm()),
    #                                                               compose.ChainRule(2, factor=False)),
    #                                            base.Debug('c')),
    #                           compose.ChainRule(2, factor=False))

    # flow = compose.sequential(compose.ChainRule(2, factor=True),
    #                           compose.factored(compose.sequential(affine.Identity(),
    #                                                               compose.ChainRule(2, factor=True),
    #                                                               compose.factored(affine.Identity(),
    #                                                                                normalization.ActNorm()),
    #                                                               compose.ChainRule(2, factor=False)),
    #                                            affine.Identity()),
    #                           compose.ChainRule(2, factor=False))

    # flow = compose.sequential(reshape.Squeeze(),
    #                           reshape.UnSqueeze())

    # flow = compose.sequential(compose.ChainRule(2, factor=True),
    #                           compose.factored(normalization.ActNorm(),
    #                                            affine.LocalDense()),
    #                           compose.ChainRule(2, factor=False))

    # flow = nonlinearities.Logit()

    # flow = compose.sequential(compose.ChainRule(2, factor=True),
    #                           compose.ChainRule(2, factor=False))

    # flow = compose.sequential(compose.ChainRule(2, factor=True),
    #                           compose.factored(affine.OnebyOneConvLAX(),
    #                                            base.Debug('b')),
    #                           compose.ChainRule(2, factor=False))

    # flow = compose.sequential(affine.OnebyOneConvLAX())

    # flow = igr.SoftmaxPP()

    flow_test(flow, inputs, key)