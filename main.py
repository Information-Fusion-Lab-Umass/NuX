import sys
# sys.path.append('NuX')
# sys.path.append('NuX/nux')
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

if __name__ == "__main__":

  def create_fun():
    return nux.ResidualFlow()

  rng = random.PRNGKey(0)
  x = random.normal(rng, (1, 2))
  inputs = {'x': x}

  flow = nux.transform_flow(create_fun)
  params, state = flow.init(rng, inputs, batch_axes=(0,))

  outputs, _ = flow.apply(params, state, rng, inputs, accumulate=["log_det", "flow_norm"])
  reconstr, _ = flow.apply(params, state, rng, outputs, sample=True, ignore_prior=True)

  assert 0