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

def flow_test(layer, inputs, key):
    # language=rst
    """
    Test if a flow implementation is correct.  Checks if the forward and inverse functions are consistent and
    compares the jacobian determinant calculation against an autograd calculation.

    :param flow: A normalizing flow
    :param x: A batched input
    :param key: JAX random key
    """
    init_fun, data_dependent_init_fun = layer

    input_shapes = util.tree_shapes(inputs)

    # Initialize the flow
    flow = init_fun(key, input_shapes)
    params_structure = tree_util.tree_structure(flow.params)
    state_structure = tree_util.tree_structure(flow.state)

    # Ensure that data dependent init produces the same output shapes
    _, flow_ddi = data_dependent_init_fun(key, inputs)
    params_structure_ddi = tree_util.tree_structure(flow_ddi.params)
    state_structure_ddi = tree_util.tree_structure(flow_ddi.state)

    assert params_structure == params_structure_ddi
    assert state_structure == state_structure_ddi

    outputs, _ = flow.forward(flow.params, flow.state, inputs)
    reconstr, _ = flow.inverse(flow.params, flow.state, outputs)

    assert jnp.allclose(inputs['x'], reconstr['x'], atol=1e-04)
    assert jnp.allclose(outputs['log_det'], reconstr['log_det'], atol=1e-04)

    # Ensure that it works with batches
    inputs_batched = tree_util.tree_map(lambda x: x[None], inputs)
    flow.forward(flow.params, flow.state, inputs_batched)

    # Make sure that the log det terms are correct
    def z_from_x(unflatten, x_flat):
        x = unflatten(x_flat)
        outputs, _ = flow.forward(flow.params, flow.state, {'x': x})
        return ravel_pytree(outputs['x'])[0]

    def single_elt_logdet(x):
        x_flat, unflatten = ravel_pytree(x)
        jac = jax.jacobian(partial(z_from_x, unflatten))(x_flat)
        return 0.5*jnp.linalg.slogdet(jac.T@jac)[1]

    actual_log_det = single_elt_logdet(inputs['x'])
    assert jnp.allclose(actual_log_det, outputs['log_det'], atol=1e-04)

def standard_layer_tests():
    layers = [affine.AffineLDU,
              partial(affine.AffineSVD, n_householders=4),
              affine.AffineDense,
              affine.Affine,
              nonlinearities.LeakyReLU,
              nonlinearities.Sigmoid,
              nonlinearities.Logit,
              reshape.Flatten,
              reshape.Reverse,
              dequantize.UniformDequantization,
              normalization.ActNorm,
              normalization.BatchNorm,
              compose.Identity,
              compose.ReverseInputs,
              partial(maf.MAF, [1024, 1024]),
              coupling.Coupling]

    key = random.PRNGKey(0)
    x = random.normal(key, (10,))
    x = jax.nn.softmax(x)

    for layer in layers:
        print()
        print(layer)
        flow_test(layer(), x, key)

def image_layer_test():
    layers = [affine.OnebyOneConvLDU,
              affine.OnebyOneConv,
              affine.OnebyOneConvLAX,
              affine.LocalDense,
              partial(conv.CircularConv, filter_size=(2, 2)),
              reshape.Squeeze,
              reshape.UnSqueeze,
              partial(reshape.Transpose, axis_order=(1, 0, 2)),
              partial(reshape.Reshape, shape=(2, -1)),
              coupling.Coupling]

    key = random.PRNGKey(0)
    x_img = random.normal(key, (6, 2, 4))

    for layer in layers:
        print()
        print(layer)
        flow_test(layer(), x_img, key)

def unit_test():
    key = random.PRNGKey(0)
    x = random.normal(key, (6, 2, 4))
    #x = random.normal(key, (10,))
    inputs = {'x': x}

    layer = affine.LocalDense()
    flow_test(layer, inputs, key)