import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree
from functools import partial
import src.util as util

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

def flow_test(flow, x, key):
    # language=rst
    """
    Test if a flow implementation is correct.  Checks if the forward and inverse functions are consistent and
    compares the jacobian determinant calculation against an autograd calculation.

    :param flow: A normalizing flow
    :param x: A batched input
    :param key: JAX random key
    """
    # Initialize the flow with conditioning.
    init_fun, forward, inverse = flow
    names, output_shape, params, state = init_fun(key, x.shape)

    # Make sure that the forwards and inverse functions are consistent
    log_det1, z, updated_state = forward(params, state, x, key=key)
    log_det2, fz, updated_state = inverse(params, state, z, key=key)

    x_diff = jnp.linalg.norm(x - fz)
    log_det_diff = jnp.linalg.norm(log_det1 - log_det2)
    print('Transform consistency diffs: x_diff: %5.3f, log_det_diff: %5.3f'%(x_diff, log_det_diff))

    # Make sure that the log det terms are correct
    def z_from_x(unflatten, x_flat):
        x = unflatten(x_flat)
        z = forward(params, state, x, key=key)[1]
        return ravel_pytree(z)[0]

    def single_elt_logdet(x):
        x_flat, unflatten = ravel_pytree(x)
        jac = jax.jacobian(partial(z_from_x, unflatten))(x_flat)
        return jnp.linalg.slogdet(jac)[1]
        # return 0.5*jnp.linalg.slogdet(jac.T@jac)[1]

    actual_log_det = single_elt_logdet(x)

    print('actual_log_det', actual_log_det)
    print('log_det', log_det1)

    log_det_diff = jnp.linalg.norm(log_det1 - actual_log_det)
    print('Log det diff: %5.3f'%(log_det_diff))

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
    # x = random.normal(key, (6, 2, 4))
    x = random.normal(key, (10,))
    layer = spline.NeuralSpline(4)
    flow_test(layer, x, key)