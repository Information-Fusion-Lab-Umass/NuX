import sys
import nux
import jax
from jax import vmap, jit, random
import jax.numpy as jnp
import nux.util as util
from debug import *

# jax.config.update('jax_log_compiles', True)

from functools import partial
from jax.scipy.special import logsumexp
import haiku as hk
from nux.tests.bijective_test import flow_test

if __name__ == "__main__":

  def create_fun():
    return nux.CouplingLogitsticMixtureLogit()
    # return nux.CouplingLogitsticMixtureLogit(reverse=True, use_condition=True)
    return nux.sequential(nux.SurjectiveMaxPool(),
                          nux.UnitGaussianPrior())

  rng = random.PRNGKey(0)
  # x = random.normal(rng, (10, 4))
  x = random.normal(rng, (10, 16, 16, 3))
  x = jax.nn.sigmoid(x)
  inputs = {"x": x[0], "condition": x[0]}

  flow_test(create_fun, inputs, rng)

  ################################################
  ################################################

  inputs = {"x": x, "condition": x}

  flow = nux.transform_flow(create_fun)
  params, state = flow.init(rng, inputs, batch_axes=(0,))

  @jit
  def nll(params, state, rng, inputs):
    outputs, _ = flow.apply(params, state, rng, inputs, accumulate=["log_det", "flow_norm"])
    return -jnp.mean(outputs["log_det"])

  gradfun = jit(jax.grad(nll))
  grad = gradfun(params, state, rng, inputs)

  outputs, _ = flow.apply(params, state, rng, inputs, accumulate=["log_det", "flow_norm"])
  reconstr, _ = flow.apply(params, state, rng, outputs, sample=True, ignore_prior=True)

  assert 0