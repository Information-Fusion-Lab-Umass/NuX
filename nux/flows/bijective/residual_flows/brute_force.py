import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap, jit
from functools import partial
from typing import Optional, Mapping, Callable, Sequence
from nux.internal.base import CustomFrame
from haiku._src.typing import PRNGKey
import nux.internal.functional as functional
import haiku._src.base as hk_base

from nux.flows.bijective.residual_flows.power_series import neumann_jacobian_terms

################################################################################################################

def log_det_estimate(apply_fun, params, state, x, rng, batch_info):
  x_shape, batch_shape = batch_info
  assert len(x_shape) == 1, "Not going to implement this for images"

  gx, state = apply_fun(params, state, x, rng)
  z = x + gx

  # Compute the jacobian of the transform
  jac_fun = jax.jacobian(lambda x: apply_fun(params, state, x[None], rng)[0][0])
  vmap_trace = jnp.trace
  for i in range(len(batch_shape)):
    jac_fun = vmap(jac_fun)
    vmap_trace = vmap(vmap_trace)

  J = jac_fun(x)

  # Generate the terms of the neumann series
  terms = neumann_jacobian_terms(J, rng, n_terms=10, n_exact=10)

  # Rescale the terms and sum over k (starting at k=1)
  cut_terms = terms[1:]
  log_det_coeff = -1/(1 + jnp.arange(cut_terms.shape[0]))
  log_det_coeff = util.broadcast_to_first_axis(log_det_coeff, cut_terms.ndim)
  log_det_terms = log_det_coeff*cut_terms
  summed_log_det_terms = log_det_terms.sum(axis=0)

  # Compute the log det
  log_det = vmap_trace(summed_log_det_terms)

  return z, log_det, terms, state

@partial(jax.custom_vjp, nondiff_argnums=(0,))
def res_flow_estimate(apply_fun, params, state, x, rng, batch_info):
  z, log_det, state = log_det_estimate(apply_fun, params, state, x, rng, batch_info)
  return z, log_det, state

def estimate_fwd(apply_fun, params, state, x, rng, batch_info):
  z, log_det, terms, state = log_det_estimate(apply_fun, params, state, x, rng, batch_info)

  # Accumulate the terms we need for the gradient
  summed_terms_for_grad = terms.sum(axis=0)

  x_shape, batch_shape = batch_info
  sum_axes = util.last_axes(x_shape)

  # Compute dlogdet(I + J(x;theta))/dtheta
  def jjp(params, unbatched_x, unbatched_summed_terms):
    jac_fun = jax.jacobian(lambda x: apply_fun(params, state, x[None], rng)[0][0])
    J = jac_fun(unbatched_x)
    return jnp.trace(unbatched_summed_terms@J)

  vmapped_grad_vjvp = jax.grad(jjp, argnums=(0, 1))
  for i in range(len(batch_shape)):
    vmapped_grad_vjvp = jax.vmap(vmapped_grad_vjvp, in_axes=(None, 0, 0))

  dlogdet_dtheta, dlogdet_dx = vmapped_grad_vjvp(params, x, summed_terms_for_grad)

  ctx = x, params, state, rng, batch_info, dlogdet_dtheta, dlogdet_dx
  return (z, log_det, state), ctx

def estimate_bwd(apply_fun, ctx, g):
  dLdz, dLdlogdet, _ = g
  x, params, state, rng, batch_info, dlogdet_dtheta, dlogdet_dx = ctx
  x_shape, batch_shape = batch_info
  batch_axes = tuple(range(len(batch_shape)))

  dLdtheta = jax.tree_util.tree_map(lambda x: util.broadcast_to_first_axis(dLdlogdet, x.ndim)*x, dlogdet_dtheta)
  dLdx     = jax.tree_util.tree_map(lambda x: util.broadcast_to_first_axis(dLdlogdet, x.ndim)*x, dlogdet_dx)

  # Reduce over the batch axes
  if len(batch_axes) > 0:
    dLdtheta = jax.tree_map(lambda x: x.sum(axis=batch_axes), dLdtheta)

  with hk_base.frame_stack(CustomFrame.create_from_params_and_state(params, state)):

    # Compute the partial derivatives wrt x
    _, vjp_fun = jax.vjp(lambda params, x: x + apply_fun(params, state, x, rng)[0], params, x, has_aux=False)
    dtheta, dx = vjp_fun(dLdz)

    # Combine the partial derivatives
    dLdtheta = jax.tree_multimap(lambda x, y: x + y, dLdtheta, dtheta)
    dLdx     = jax.tree_multimap(lambda x, y: x + y, dLdx, dx)

    return dLdtheta, None, dLdx, None, None

res_flow_estimate.defvjp(estimate_fwd, estimate_bwd)
