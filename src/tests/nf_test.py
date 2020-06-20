import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree
from functools import partial
import src.util as util
import jax.tree_util as tree_util

import src.flows.affine as affine
import src.flows.nonlinearities as nonlinearities
import src.flows.reshape as reshape
import src.flows.dequantize as dequantize
import src.flows.normalization as normalization
import src.flows.conv as conv
import src.flows.compose as compose
import src.flows.maf as maf
import src.flows.coupling as coupling
import src.flows.spline as spline
import src.flows.basic as basic
import src.flows.base as base

def flow_test(layer, inputs, key):
    # language=rst
    """
    Test if a flow implementation is correct.  Checks if the forward and inverse functions are consistent and
    compares the jacobian determinant calculation against an autograd calculation.

    :param flow: A normalizing flow
    :param x: A batched input
    :param key: JAX random key
    """
    init_fun = layer
    input_shapes = util.tree_shapes(inputs)

    # Initialize the flow
    inputs_batched = tree_util.tree_map(lambda x: jnp.broadcast_to(x[None], (8,) + x.shape), inputs)
    _, flow = init_fun(key, inputs, batched=False)
    _, flow_batched = init_fun(key, inputs_batched, batched=True)

    # _, flow = init_fun(key, inputs_batched)
    params_structure_ddi = tree_util.tree_structure(flow.params)
    state_structure_ddi = tree_util.tree_structure(flow.state)

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
    assert jnp.allclose(actual_log_det, outputs['log_det'], atol=1e-04)
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
              dequantize.UniformDequantization,
              normalization.ActNorm,
              # normalization.BatchNorm,
              basic.Identity,
              partial(maf.MAF, [1024, 1024]),
              coupling.Coupling]

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
    x = random.normal(key, (4, 6, 8))
    # x = random.normal(key, (10,))
    x = jax.nn.softmax(x)
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
    #                   compose.factored(compose.sequential(base.Debug('a'),
    #                                       compose.ChainRule(2, factor=True),
    #                                       compose.factored(base.Debug('b'),
    #                                                ActNorm()),
    #                                       compose.ChainRule(2, factor=False)),
    #                            base.Debug('c')),
    #                   compose.ChainRule(2, factor=False))

    # flow = compose.sequential(reshape.Squeeze(),
    #                           reshape.UnSqueeze())

    # flow = compose.sequential(compose.ChainRule(2, factor=True),
    #                           compose.factored(normalization.ActNorm(),
    #                                            affine.LocalDense()),
    #                           compose.ChainRule(2, factor=False))

    # flow = nonlinearities.Logit()

    flow = compose.sequential(compose.ChainRule(2, factor=True),
                              compose.ChainRule(2, factor=False))

    flow = compose.sequential(compose.ChainRule(2, factor=True),
                              compose.factored(affine.OnebyOneConvLAX(),
                                               base.Debug('b')),
                              compose.ChainRule(2, factor=False))

    # flow = compose.sequential(affine.OnebyOneConvLAX())

    flow_test(flow, inputs, key)