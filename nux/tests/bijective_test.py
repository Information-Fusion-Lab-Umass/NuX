import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree
from functools import partial
import nux.util as util
import jax.tree_util as tree_util

import nux

def reconstruction_test(create_fun, inputs, rng, batch_axes):
  flow = nux.transform_flow(create_fun)

  # Initialize the flow
  params, state = flow.init(rng, inputs, batch_axes=batch_axes)

  # Make sure that reconstructions are correct
  outputs, _ = flow.apply(params, state, rng, inputs)

  inputs_for_reconstr = inputs.copy()
  inputs_for_reconstr.update(outputs) # We might have condition variables in inputs!
  reconstr, _ = flow.apply(params, state, rng, inputs_for_reconstr, sample=True)
  assert jnp.allclose(inputs["x"], reconstr["x"], atol=1e-4)
  print("Passed reconstruction tests")

def log_det_test(create_fun, inputs, rng):
  flow = nux.transform_flow(create_fun)

  # Initialize the flow
  params, state = flow.init(rng, inputs)
  outputs, _ = flow.apply(params, state, rng, inputs)

  # Make sure that the log det terms are correct
  def z_from_x(unflatten, x_flat):
    x = unflatten(x_flat)
    flow_inputs = inputs.copy()
    flow_inputs["x"] = x
    outputs, _ = flow.apply(params, state, rng, flow_inputs)
    return ravel_pytree(outputs["x"])[0]

  def single_elt_logdet(x):
    x_flat, unflatten = ravel_pytree(x)
    jac = jax.jacobian(partial(z_from_x, unflatten))(x_flat)
    return jnp.linalg.slogdet(jac)[1]

  actual_log_det = single_elt_logdet(inputs["x"])
  if jnp.allclose(actual_log_det, outputs["log_det"], atol=1e-04) == False:
    print(f"actual_log_det: {actual_log_det:.3f}, outputs['log_det']: {outputs['log_det']:.3f}")
    assert 0
  print("Passed log det tests")

def flow_test(create_fun, inputs, rng):
  """
  Test if a flow implementation is correct.  Checks if the forward and inverse functions are consistent and
  compares the jacobian determinant calculation against an autograd calculation.
  """
  # Initialize the flow
  inputs_batched = tree_util.tree_map(lambda x: jnp.broadcast_to(x[None], (3,) + x.shape), inputs)
  inputs_doubly_batched = tree_util.tree_map(lambda x: jnp.broadcast_to(x[None], (3,) + x.shape), inputs_batched)

  reconstruction_test(create_fun, inputs, rng, batch_axes=())
  reconstruction_test(create_fun, inputs_batched, rng, batch_axes=(0,))
  reconstruction_test(create_fun, inputs_doubly_batched, rng, batch_axes=(0, 1))

  log_det_test(create_fun, inputs, rng)
