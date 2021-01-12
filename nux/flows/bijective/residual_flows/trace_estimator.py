import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap, jit
from functools import partial
from typing import Optional, Mapping, Callable, Sequence
from nux.internal.base import CustomFrame
from haiku._src.typing import PRNGKey
import haiku._src.base as hk_base
from nux.flows.bijective.residual_flows.power_series import unbiased_neumann_vjp_terms

################################################################################################################

def log_det_sliced_estimate(apply_fun,
                            params,
                            state,
                            x,
                            rng,
                            batch_info):
  trace_key, roulette_key = random.split(rng, 2)

  # Evaluate the flow and get the vjp function
  gx, state = apply_fun(params, state, x, rng, update_params=True)
  _, vjp_fun, _ = jax.vjp(lambda x: apply_fun(params, state, x, rng, update_params=False), x, has_aux=True)
  z = x + gx

  # Generate the probe vector for the trace estimate
  v = random.normal(trace_key, x.shape)

  # Get all of the terms we need for the log det and gradient estimates
  terms = unbiased_neumann_vjp_terms(vjp_fun, v, roulette_key, n_terms=7, n_exact=4)

  # Rescale the terms and sum over k (starting at k=1)
  cut_terms = terms[1:]
  log_det_coeff = -1/(1 + jnp.arange(cut_terms.shape[0]))
  log_det_coeff = util.broadcast_to_first_axis(log_det_coeff, cut_terms.ndim)
  log_det_terms = log_det_coeff*cut_terms
  summed_log_det_terms = log_det_terms.sum(axis=0)

  # Compute the log det
  x_shape, batch_shape = batch_info
  log_det = jnp.sum(summed_log_det_terms*v, axis=util.last_axes(x_shape))

  return z, log_det, v, terms, state

@partial(jax.custom_vjp, nondiff_argnums=(0,))
def res_flow_sliced_estimate(apply_fun, params, state, x, rng, batch_info):
  z, log_det, _, _, state = log_det_sliced_estimate(apply_fun, params, state, x, rng, batch_info)
  return z, log_det, state

def sliced_estimate_fwd(apply_fun, params, state, x, rng, batch_info):
  z, log_det, v, terms, state = log_det_sliced_estimate(apply_fun, params, state, x, rng, batch_info)

  # Accumulate the terms we need for the gradient
  summed_terms_for_grad = terms.sum(axis=0)

  x_shape, batch_shape = batch_info
  batch_dim = len(batch_shape)
  sum_axes = util.last_axes(x_shape)

  # Compute dlogdet(I + J(x;theta))/dtheta
  def vjvp(params, unbatched_x, unbatched_summed_terms, unbatched_v):
    # Remember that apply_fun is autobatched and can automatically pad leading dims!
    if batch_dim > 0:
      _, vjp_fun, _ = jax.vjp(lambda x: apply_fun(params, state, x[None], rng, update_params=False), unbatched_x, has_aux=True)
      w, = vjp_fun(unbatched_summed_terms[None])
    else:
      _, vjp_fun, _ = jax.vjp(lambda x: apply_fun(params, state, x, rng, update_params=False), unbatched_x, has_aux=True)
      w, = vjp_fun(unbatched_summed_terms)
    return jnp.sum(w*unbatched_v)

  # vmap over the batch dimensions
  vmapped_vjvp = jax.grad(vjvp, argnums=(0, 1))
  for i in range(batch_dim):
    vmapped_vjvp = jax.vmap(vmapped_vjvp, in_axes=(None, 0, 0, 0))

  # Compute the vector Jacobian vector products to get the gradient estimate terms.
  dlogdet_dtheta, dlogdet_dx = vmapped_vjvp(params, x, summed_terms_for_grad, v)

  # Store off everything for the backward pass
  ctx = x, params, state, rng, batch_info, dlogdet_dtheta, dlogdet_dx
  return (z, log_det, state), ctx

def sliced_estimate_bwd(apply_fun, ctx, g):
  dLdz, dLdlogdet, _ = g
  x, params, state, rng, batch_info, dlogdet_dtheta, dlogdet_dx = ctx
  x_shape, batch_shape = batch_info
  batch_dim = len(batch_shape)
  batch_axes = tuple(range(batch_dim))

  if batch_dim > 0:
    def multiply_by_val(x):
      return util.broadcast_to_first_axis(dLdlogdet, x.ndim)*x
  else:
    def multiply_by_val(x):
      return dLdlogdet*x

  # Get the gradients wrt x and theta using the terms from the forward step
  dLdtheta = jax.tree_util.tree_map(multiply_by_val, dlogdet_dtheta)
  dLdx     = jax.tree_util.tree_map(multiply_by_val, dlogdet_dx)

  # Reduce over the batch axes
  if batch_dim > 0:
    dLdtheta = jax.tree_map(lambda x: x.sum(axis=batch_axes), dLdtheta)

  # Open up a new frame so that apply_fun can retrieve the parameters
  with hk_base.frame_stack(CustomFrame.create_from_params_and_state(params, state)):

    # Add in the partial derivatives wrt x
    _, vjp_fun = jax.vjp(lambda params, x: x + apply_fun(params, state, x, rng, update_params=False)[0], params, x, has_aux=False)
    dtheta, dx = vjp_fun(dLdz)

    # Combine the partial derivatives
    dLdtheta = jax.tree_multimap(lambda x, y: x + y, dLdtheta, dtheta)
    dLdx     = jax.tree_multimap(lambda x, y: x + y, dLdx, dx)

    return dLdtheta, None, dLdx, None, None

res_flow_sliced_estimate.defvjp(sliced_estimate_fwd, sliced_estimate_bwd)