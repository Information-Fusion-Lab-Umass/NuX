import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree
from functools import partial
import nux.util as util
import nux
import numpy as np
from nux.flows.base import Flow

def exact_test(flow, x, rng_key):
  assert isinstance(flow, Flow)

  if isinstance(flow, nux.SquareLogit) or \
     isinstance(flow, nux.Logit):
    # These flows need a bounded input
    x = jax.nn.softmax(x)

  # Initialize the flow
  flow(x, rng_key=rng_key)
  params = flow.get_params()

  # Scramble the parameters to undo the data dependent init
  flat_params, unflatten = ravel_pytree(params)
  flat_params = random.normal(rng_key, flat_params.shape)
  params = unflatten(flat_params)

  # Compute the log likelihood contribution of flow
  z, log_det = flow(x, params=params, rng_key=rng_key)

  # Reconstruct x
  x_reconstr, log_det2 = flow(z, params=params, rng_key=rng_key, inverse=True)
  assert jnp.allclose(x, x_reconstr)
  assert jnp.allclose(log_det, log_det2)

  # Compute the exact jacobian
  def unbatched_apply_fun(x):
    z, _ = flow(x[None], params=params, rng_key=rng_key)
    return z[0]

  J = jax.vmap(jax.jacobian(unbatched_apply_fun))(x)
  total_dim = np.prod(x.shape[1:])
  J_flat = J.reshape((-1, total_dim, total_dim))
  log_det_exact = jnp.linalg.slogdet(J_flat)[1]

  assert jnp.allclose(log_det_exact, log_det)
  print(f"{str(flow)} passed the reconstruction and log det test")

################################################################################################################

if __name__ == "__main__":
  # from debug import *
  rng_key = random.PRNGKey(0)
  x = random.normal(rng_key, shape=(16, 4, 4, 4))

  flows = [nux.CenterAndScale(0.5),
           nux.DiscreteBias(),
           nux.Bias(),
           nux.StaticScale(0.5),
           nux.Scale(),
           nux.ShiftScale(),
           nux.ShiftScalarScale(),
           nux.StaticShiftScale(0.5, 0.5),
           nux.DenseMVP(),
           nux.CaleyOrthogonalMVP(),
           nux.PLUMVP(),
           nux.CircularConv(),
           nux.OneByOneConv(),
           nux.LogisticCDFMixtureLogit(),
           nux.Softplus(),
           nux.LeakyReLU(),
           nux.SneakyReLU(),
           nux.SquarePlus(),
           nux.SquareSigmoid(),
           nux.SquareLogit(),
           nux.Sigmoid(),
           nux.Logit(),
           nux.SLog(),
           nux.Reverse(),
           nux.Squeeze(),
           nux.RationalQuadraticSpline()]

  for flow in flows:
    exact_test(flow, x, rng_key)
