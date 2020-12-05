import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap, jit
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence
from nux.internal.layer import Layer
from nux.internal.base import CustomFrame
from haiku._src.typing import PRNGKey
import nux.networks as net
from nux.internal.functional import make_functional_modules, make_functional_modules_with_fixed_state

import nux.internal.functional as functional
import haiku._src.base as hk_base

################################################################################################################

def roulette_coefficients(k_start, k_end, key):
  # Compute the roulette coefficients using a geometric distribution
  k = jnp.arange(k_start, k_end)
  p = 0.5
  u = random.uniform(key, (1,))[0]
  N = jnp.floor(jnp.log(u)/jnp.log(1 - p)) + 1
  p_N_geq_k = (1 - p)**k

  # Zero out the terms that are over N
  roulette_coeff = jnp.where(k > N, 0.0, 1/p_N_geq_k)

  return roulette_coeff

def unbiased_neumann_coefficients(key, n_terms, n_exact):
  # Compute the standard scaling terms for the neumann series
  k = jnp.arange(0, n_terms)
  coeff = (-1)**k

  # Compute the roulette scaling terms
  roulette_coeff = roulette_coefficients(0, n_terms - n_exact, key)

  # We don't want to apply this to the exact terms
  roulette_coeff = jnp.hstack([jnp.ones(n_exact), roulette_coeff])

  return coeff*roulette_coeff

################################################################################################################

def vjp_iterations(vjp_fun, v, n_terms):

  @jit
  def scan_fun(carry, inputs):
    w = carry
    w_updated, = vjp_fun(w)
    return w_updated, w

  k = jnp.arange(n_terms - 1)
  w, terms = jax.lax.scan(scan_fun, v, k)
  terms = jnp.concatenate([terms, w[None]], axis=0)

  return terms

def unbiased_neumann_vjp_terms(vjp_fun, v, rng, n_terms=10, n_exact=4):
  # This function assumes that we start at k=0!

  # Compute the terms in the power series.
  terms = vjp_iterations(vjp_fun, v, n_terms)

  # Compute the coefficients for each term
  coeff = unbiased_neumann_coefficients(rng, n_terms, n_exact)
  coeff = util.broadcast_to_first_axis(coeff, terms.ndim)

  return coeff*terms

################################################################################################################

def jacobian_power_iterations(J, n_terms):

  def scan_fun(carry, inputs):
    J_k = carry
    J_kp1 = J@J_k
    return J_kp1, J_k

  k = jnp.arange(n_terms - 1)
  jac_K, terms = jax.lax.scan(scan_fun, J, k)
  I = jnp.expand_dims(jnp.eye(J.shape[-1]), axis=tuple(range(len(J.shape) - 2)))
  I = jnp.broadcast_to(I, J.shape)
  terms = jnp.concatenate([I[None], terms], axis=0)

  return terms

def neumann_jacobian_terms(J, rng, n_terms=10, n_exact=4):

  terms = jacobian_power_iterations(J, n_terms)

  # Compute the coefficients for each term
  coeff = unbiased_neumann_coefficients(rng, n_terms, n_exact)
  coeff = util.broadcast_to_first_axis(coeff, terms.ndim)

  return coeff*terms

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

  return z, log_det, terms

@partial(jax.custom_vjp, nondiff_argnums=(0,))
def res_flow_estimate(apply_fun, params, state, x, rng, batch_info):
  z, log_det, _ = log_det_estimate(apply_fun, params, state, x, rng, batch_info)
  return z, log_det

def estimate_fwd(apply_fun, params, state, x, rng, batch_info):
  z, log_det, terms = log_det_estimate(apply_fun, params, state, x, rng, batch_info)

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
  return (z, log_det), ctx

def estimate_bwd(apply_fun, ctx, g):
  dLdz, dLdlogdet = g
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

################################################################################################################

def log_det_sliced_estimate(apply_fun, params, state, x, rng, batch_info):
  trace_key, roulette_key = random.split(rng, 2)

  # Evaluate the flow and get the vjp function
  gx, vjp_fun, state = jax.vjp(lambda x: apply_fun(params, state, x, rng), x, has_aux=True)
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

  return z, log_det, v, terms

@partial(jax.custom_vjp, nondiff_argnums=(0,))
def res_flow_sliced_estimate(apply_fun, params, state, x, rng, batch_info):
  z, log_det, _, _ = log_det_sliced_estimate(apply_fun, params, state, x, rng, batch_info)
  return z, log_det

def sliced_estimate_fwd(apply_fun, params, state, x, rng, batch_info):
  z, log_det, v, terms = log_det_sliced_estimate(apply_fun, params, state, x, rng, batch_info)

  # Accumulate the terms we need for the gradient
  summed_terms_for_grad = terms.sum(axis=0)

  x_shape, batch_shape = batch_info
  sum_axes = util.last_axes(x_shape)

  # Compute dlogdet(I + J(x;theta))/dtheta
  def vjvp(params, unbatched_x, unbatched_summed_terms, unbatched_v):
    _, vjp_fun, _ = jax.vjp(lambda x: apply_fun(params, state, x[None], rng), unbatched_x, has_aux=True)
    w, = vjp_fun(unbatched_summed_terms[None])
    return jnp.sum(w*unbatched_v)

  vmapped_vjvp = jax.grad(vjvp, argnums=(0, 1))
  for i in range(len(batch_shape)):
    vmapped_vjvp = jax.vmap(vmapped_vjvp, in_axes=(None, 0, 0, 0))

  dlogdet_dtheta, dlogdet_dx = vmapped_vjvp(params, x, summed_terms_for_grad, v)

  ctx = x, params, state, rng, batch_info, dlogdet_dtheta, dlogdet_dx
  return (z, log_det), ctx

def sliced_estimate_bwd(apply_fun, ctx, g):
  dLdz, dLdlogdet = g
  x, params, state, rng, batch_info, dlogdet_dtheta, dlogdet_dx = ctx
  x_shape, batch_shape = batch_info
  batch_axes = tuple(range(len(batch_shape)))

  dLdtheta = jax.tree_util.tree_map(lambda x: util.broadcast_to_first_axis(dLdlogdet, x.ndim)*x, dlogdet_dtheta)
  dLdx     = jax.tree_util.tree_map(lambda x: util.broadcast_to_first_axis(dLdlogdet, x.ndim)*x, dlogdet_dx)

  # Reduce over the batch axes
  if len(batch_axes) > 0:
    dLdtheta = jax.tree_map(lambda x: x.sum(axis=batch_axes), dLdtheta)

  with hk_base.frame_stack(CustomFrame.create_from_params_and_state(params, state)):

    # Add in the partial derivatives wrt x
    _, vjp_fun = jax.vjp(lambda params, x: x + apply_fun(params, state, x, rng)[0], params, x, has_aux=False)
    dtheta, dx = vjp_fun(dLdz)

    # Combine the partial derivatives
    dLdtheta = jax.tree_multimap(lambda x, y: x + y, dLdtheta, dtheta)
    dLdx     = jax.tree_multimap(lambda x, y: x + y, dLdx, dx)

    return dLdtheta, None, dLdx, None, None

res_flow_sliced_estimate.defvjp(sliced_estimate_fwd, sliced_estimate_bwd)

################################################################################################################

def res_flow_exact(res_block, x, rng):
  # This must be called using auto-batch so that jax.jacobian works!

  flat_x, unflatten = jax.flatten_util.ravel_pytree(x)

  def apply_res_block(flat_x):
    x = unflatten(flat_x)
    out = x + res_block(x[None], rng, update_params=False)[0]
    return jax.flatten_util.ravel_pytree(out)[0]

  J = jax.jacobian(apply_res_block)(flat_x)


  # value, jacobian = jax.value_and_jacfwd(apply_res_block)(flat_x)
  # import pdb; pdb.set_trace()

  log_det = jnp.linalg.slogdet(J)[1]

  z = x + res_block(x[None], rng, update_params=True)[0]
  return z, log_det

################################################################################################################

def _fixed_point(f, x_init):
  # http://www.autodiff.org/Docs/euroad/Second%20EuroAd%20Workshop%20-%20Sebastian%20Schlenkrich%20-%20Differentianting%20Fixed%20Point%20Iterations%20with%20ADOL-C.pdf
  max_iters = 10000
  atol = 1e-5

  def cond_fun(val):
    x_prev, x, i = val
    max_iters_reached = jnp.where(i >= max_iters, True, False)
    tolerance_achieved = jnp.allclose(x_prev - x, 0.0, atol=atol)
    return ~(max_iters_reached | tolerance_achieved)

  def body_fun(val):
    _, x, i = val
    fx = f(x)
    return x, fx, i + 1

  _, x, N = jax.lax.while_loop(cond_fun, body_fun, (x_init, f(x_init), 0.0))
  return x, N

def contractive_fixed_point(apply_fun, params, state, x_current, z):
  # Step of fixed point iteration
  gx, state = apply_fun(params, state, x_current)
  return z - gx

@partial(jax.custom_vjp, nondiff_argnums=(0,))
def fixed_point(apply_fun, params, state, z, roulette_rng):
  # Invert a contractive function using fixed point iterations.

  def fixed_point_iter(x):
    return contractive_fixed_point(apply_fun, params, state, x, z)

  x, N = _fixed_point(fixed_point_iter, z)
  return x

def fixed_point_fwd(apply_fun, params, state, z, roulette_rng):
  # We need to save frame_data and the fixed point solution for backprop
  x = fixed_point(apply_fun, params, state, z, roulette_rng)
  return x, (params, state, x, z, roulette_rng)

def fixed_point_bwd(apply_fun, ctx, dLdx):
  assert 0, "Bro, did you seriously just try to backprop through a fixed point iteration?"
  params, state, x, z, roulette_rng = ctx

  with hk_base.frame_stack(CustomFrame.create_from_params_and_state(params, state)):

    _, vjp_x = jax.vjp(lambda x: contractive_fixed_point(apply_fun, params, state, x, z), x)

    def rev_iter(zeta):
      zetaT_dFdx, = vjp_x(zeta)
      return dLdx + zetaT_dFdx

    zeta, N = _fixed_point(rev_iter, dLdx)

    # Go from zeta to the gradient of the frame data
    _, vjp_u = jax.vjp(lambda params: contractive_fixed_point(apply_fun, params, state, x, z), params)
    dparams, = vjp_u(zeta)

    # Also handle the gradient wrt z here.  To do this, we need to solve (dx/dz)^{-1}dx.
    # Do this with vjps against terms in the neumann series for dx/dz
    _, vjp_x = jax.vjp(lambda x: apply_fun(params, state, x)[0], x, has_aux=False)
    terms = unbiased_neumann_vjp_terms(vjp_x, dLdx, roulette_rng, n_terms=10, n_exact=10)
    dx_star = terms.sum(axis=0)

    return dparams, None, dx_star, None

fixed_point.defvjp(fixed_point_fwd, fixed_point_bwd)

################################################################################################################

class ResidualFlow(Layer):

  def __init__(self,
               scale: float=1.0,
               create_network: Callable=None,
               fixed_point_iters: Optional[int]=1000,
               exact_log_det: Optional[bool]=False,
               use_trace_estimator: bool=True,
               network_kwargs: Optional=None,
               name: str="residual_flow"
  ):
    """ Residual flows https://arxiv.org/pdf/1906.02735.pdf

    Args:
      create_network   : Function to create the conditioner network.  Should accept a tuple
                         specifying the output shape.  See coupling_base.py
      fixed_point_iters: Max number of iterations for inverse
      exact_log_det    : Whether or not to compute the exact jacobian determinant with autodiff
      network_kwargs   : Dictionary with settings for the default network (see get_default_network in util.py)
      name             : Optional name for this module.
    """
    super().__init__(name=name)
    self.create_network      = create_network
    self.fixed_point_iters   = fixed_point_iters
    self.exact_log_det       = exact_log_det
    self.network_kwargs      = network_kwargs
    self.use_trace_estimator = use_trace_estimator
    self.scale               = scale

  def get_network(self, out_shape):

    # The user can specify a custom network
    if self.create_network is not None:
      return self.create_network(out_shape)

    return util.get_default_network(out_shape, network_kwargs=self.network_kwargs, lipschitz=True)

  @property
  def auto_batched_res_block(self):
    return self.auto_batch(self.res_block, expected_depth=1, in_axes=(0, None))

  def exact_forward(self, x, rng):
    res_fun = partial(res_flow_exact, self.auto_batched_res_block)
    z, log_det = self.auto_batch(res_fun, in_axes=(0, None))(x, rng)
    return z, log_det

  def init_if_needed(self, x, rng):
    # Before extracting the frame data, we need to make sure that the
    # network is initialized!
    running_init_fn = not hk_base.params_frozen()
    if running_init_fn:
      self.auto_batched_res_block(x, rng)

  def forward(self, x, rng):
    self.init_if_needed(x, rng)

    batch_info = self.unbatched_input_shapes["x"], self.batch_shape

    fun = partial(self.auto_batched_res_block, update_params=False)
    with make_functional_modules([fun]) as ([apply_fun], \
                                             params, \
                                             state, \
                                             finalize):
      if self.use_trace_estimator:
        z, log_det = res_flow_sliced_estimate(apply_fun, params, state, x, rng, batch_info)
      else:
        z, log_det = res_flow_estimate(apply_fun, params, state, x, rng, batch_info)

      finalize(params, state)

    return z, log_det

  def invert(self, z, rng):
    self.init_if_needed(z, rng)

    # State will be held constant during the fixed point iterations
    fun = partial(self.auto_batched_res_block, update_params=False)
    (apply_fun,), params, state = make_functional_modules_with_fixed_state([fun])

    # Make sure we don't use a different random key at every step of the fixed point iterations.
    deterministic_apply_fun = lambda params, state, x: apply_fun(params, state, x, rng)

    # Run the fixed point iterations to invert at z.  We can do reverse-mode through this!
    x = fixed_point(deterministic_apply_fun, params, state, z, rng)
    return x

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           res_block_only: bool=False,
           use_exact_log_det: bool=False,
           scale: float=None,
           **kwargs
    ) -> Mapping[str, jnp.ndarray]:
    x_shape = self.get_unbatched_shapes(sample)["x"]
    self.unscaled_res_block = self.get_network(x_shape)
    scale = self.scale if scale is None else scale
    def _res_block(x, rng, **kwargs):
      return scale*self.unscaled_res_block(x, rng, **kwargs)
    self.res_block = _res_block
    # self.res_block = lambda x, rng : scale*self.unscaled_res_block(x, rng, **kwargs)

    if res_block_only:
      x = inputs["x"]
      gx = self.auto_batched_res_block(x, rng)
      return {"x": gx, "log_det": jnp.zeros(self.batch_shape)}

    if sample == False:
      x = inputs["x"]

      if self.exact_log_det or use_exact_log_det:
        z, log_det = self.exact_forward(x, rng)
      else:

        # Update the singular vectors
        # TODO: Figure out how to do this inside the custom_vjp
        self.res_block(x, rng, update_params=True)
        z, log_det = self.forward(x, rng)

      outputs = {"x": z, "log_det": log_det}
    else:
      z = inputs["x"]
      x = self.invert(z, rng)

      if self.exact_log_det or use_exact_log_det:
        _, log_det = self.exact_forward(x, rng)
      else:
        self.res_block(x, rng)
        _, log_det = self.forward(x, rng)


      outputs = {"x": x, "log_det": log_det}

    return outputs

################################################################################################################
