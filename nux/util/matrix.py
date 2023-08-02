import jax
import jax.numpy as jnp
from functools import partial

__all__ = ["svd", "matmul", "transpose"]

def svd(A):
  if A.shape[-1] == A.shape[-2]:
    return my_svd(A)
  return jnp.linalg.svd(A)

@jax.custom_jvp
def my_svd(A):
  U, s, VT = jnp.linalg.svd(A)
  V = jnp.einsum("...ji->...ij", VT)
  return U, s, V

@my_svd.defjvp
def my_svd_jvp(primals, tangents):
  A, = primals
  dA, = tangents
  U, s, V = my_svd(A)
  dU, ds, dV = svd_jvp_work(U, s, V, dA)
  return (U, s, V), (dU, ds, dV)

@partial(jnp.vectorize, signature="(n,n),(n),(n,n),(n,n)->(n,n),(n),(n,n)")
def svd_jvp_work(U, s, V, dA):
  dS = jnp.einsum("ij,iu,jv->uv", dA, U, V)
  ds = jnp.diag(dS)

  sdS = s*dS
  dSs = s[:,None]*dS

  s_diff = s[:,None]**2 - s**2 + 1e-5
  N = s.shape[-1]
  one_over_s_diff = jnp.where(jnp.arange(N)[:,None] == jnp.arange(N), 0.0, 1/s_diff)
  u_components = one_over_s_diff*(sdS + sdS.T)
  v_components = one_over_s_diff*(dSs + dSs.T)

  dU = jnp.einsum("uv,iv->iu", u_components, U)
  dV = jnp.einsum("uv,iv->iu", v_components, V)
  # import pdb; pdb.set_trace()
  return (dU, ds, dV)

@partial(jnp.vectorize, signature="(i,j),(j,k)->(i,k)")
def matmul(A, B):
  return A@B

@partial(jnp.vectorize, signature="(n,m)->(m,n)")
def transpose(A):
  return A.T

################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

def getU(J):
    return svd(J)[0]

def getS(J):
    return svd(J)[1]

def getV(J):
    return svd(J)[2]

def getW(J):
    U, s, V = svd(J)
    return jnp.einsum("iu,u->iu", U, s)

def apply_fun(fun, J, dJdx):
    out, dout = jax.vmap(lambda J, dJ: jax.jvp(fun, (J,), (dJ,)), in_axes=(None, 0))(J, dJdx)
    return out[0], dout

def loss(params):
    J, dJdx = params
    dJdx = jax.lax.stop_gradient(dJdx)
    W, dW = jax.vmap(lambda J, dJ: jax.jvp(getW, (J,), (dJ,)), in_axes=(None, 0))(J, dJdx)
    W = W[0]
    Wu_Wv = jnp.einsum("uk,kmv->muv", W, dW)
    lie_bracket = Wu_Wv - transpose(Wu_Wv)
    lie_bracket = jax.vmap(lambda lb: lb.at[jnp.tril_indices(4)].set(0.0))(lie_bracket)
    return jnp.sum(lie_bracket**2)

################################################################################################################

if __name__ == "__main__":
  from debug import *
  import jax.random as random
  jax.config.update("jax_enable_x64", True)
  jax.config.update('jax_disable_jit', True)


  rng_key = random.PRNGKey(0)

  dim = 4
  J = random.normal(rng_key, (dim, dim))
  # J = svd(J)[0]@svd(J)[2]
  dJdx = random.normal(rng_key, (dim, dim, dim))
  dJdx = jnp.einsum("kij->jik", dJdx) + dJdx # Need this to be symmetric

  ###############################################################

  # U, dUdx = apply_fun(getU, J, dJdx)
  # s, dsdx = apply_fun(getS, J, dJdx)
  # V, dVdx = apply_fun(getV, J, dJdx)
  # W, dWdx = apply_fun(getW, J, dJdx)

  # Omega = jnp.einsum("ku,kjv->juv", W, dVdx)
  # Gamma = jnp.einsum("kv,kij,u,ju->iuv", V, dJdx, s, U)
  # Gamma = Gamma - transpose(Gamma)

  ###############################################################

  params = (random.normal(rng_key, (4, 4)), random.normal(rng_key, (4, 4, 4)))

  # out, dout = jax.jvp(loss, (params,), (jax.tree_util.tree_map(jnp.ones_like, params), ))

  l, vjp = jax.vjp(loss, params)
  out_vjp = vjp(jnp.ones_like(l))

  assert 0

  import optax
  opt = optax.adam(1e-2)
  opt_state = opt.init(params)

  valgrad = jax.jit(jax.value_and_grad(loss))

  ###############################################################

  import tqdm
  pbar = tqdm.tqdm(jnp.arange(1))
  losses = []
  for i in pbar:
      old_params = params
      val, grads = valgrad(params); losses.append(val)
      updates, opt_state = opt.update(grads, opt_state)
      params = optax.apply_updates(params, updates)
  losses = jnp.array(losses)


  import pdb; pdb.set_trace()
