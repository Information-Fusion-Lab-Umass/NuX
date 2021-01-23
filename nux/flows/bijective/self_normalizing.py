import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
from nux.internal.layer import Layer
import nux.util as util
import nux.util.weight_initializers as init

################################################################################################################

def _forwardW(x, W):
  lambd = 1.0
  z = W@x
  log_det = jnp.linalg.slogdet(W)[1]
  return z, log_det

def _forwardR(x, R):
  lambd = 1.0
  z = jnp.linalg.inv(R)@x
  log_det = -jnp.linalg.slogdet(R)[1]
  return z, log_det

@jax.custom_vjp
def forward(x, W, R, k):
  z = W@x
  log_det = jnp.linalg.slogdet(W)[1]
  return z, log_det

def forward_fwd(x, W, R, k):
  z = W@x
  log_det = 0.0#jnp.linalg.slogdet(W)[1]
  reconstr_error = R@z - x
  return (z, log_det), (z, x, reconstr_error, W, R, k)

def forward_bwd(ctx, g):
  dz, dlog_det = g
  z, x, reconstr_error, W, R, k = ctx

  dW = jnp.outer(dz, x) + R.T + 2*k*R.T@jnp.outer(reconstr_error, x)
  dR = -jnp.outer(W.T@z, dz) - W.T + 2*k*jnp.outer(reconstr_error, z)
  dx = W.T@dz

  return dx, dW, dR, None

forward.defvjp(forward_fwd, forward_bwd)

def inverse(z, W, R, lambd=1.0):
  x = R@z
  log_det = jnp.linalg.slogdet(W)[1]
  return x, log_det

################################################################################################################

class SNDense(Layer):

  def __init__(self,
               name: str="self_normalizing_affine_dense",
               **kwargs
  ):
    """ Self normalizing dense layer https://arxiv.org/pdf/2011.07248.pdf
    Args:
      name:  Optional name for this module.
    """
    super().__init__(name=name, **kwargs)
    assert (weight_norm and spectral_norm) == False
    self.spectral_norm = spectral_norm
    self.weight_norm = weight_norm
    self.max_singular_value = max_singular_value
    self.max_power_iters = max_power_iters

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    outputs = {}

    x_dim, dtype = x.shape[-1], inputs["x"].dtype

    W_init = hk.initializers.TruncatedNormal(1/jnp.sqrt(x_dim))
    W = hk.get_parameter("W", shape=(x_dim, x_dim), dtype=dtype, init=W_init)
    R = hk.get_parameter("R", shape=(x_dim, x_dim), dtype=dtype, init=W_init)

    if sample == False:
      z, log_det = self.auto_batch(forward, in_axes=(0, None, None))(x, W, R)
    else:
      z, log_det = self.auto_batch(inverse, in_axes=(0, None, None))(x, W, R)

    outputs["x"] = z
    outputs["log_det"] = log_det

    return outputs

################################################################################################################

if __name__ == "__main__":
  from debug import *
  key = random.PRNGKey(0)

  dim = 4
  W = random.normal(key, (dim, dim))
  R = jnp.linalg.inv(W)
  x = random.normal(key, (dim,))

  def testR(x, R):
    z, log_det = _forwardR(x, R)
    log_pz = -0.5*jnp.vdot(z, z)
    return log_pz + log_det

  def testW(x, W):
    z, log_det = _forwardW(x, W)
    log_pz = -0.5*jnp.vdot(z, z)
    return log_pz + log_det

  def test(x, W, R):
    z, log_det = forward(x, W, R, k=1.0)
    log_pz = -0.5*jnp.vdot(z, z)
    return log_pz + log_det

  dR_true = jax.grad(testR, argnums=(1,))(x, R)
  dW_true = jax.grad(testW, argnums=(1,))(x, W)

  dx, dW, dR = jax.grad(test, argnums=(0, 1, 2))(x, W, R)

  import pdb; pdb.set_trace()