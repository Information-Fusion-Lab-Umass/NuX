import jax
import jax.numpy as jnp
import jax.random as random

def kmeans(data, k=2):

  def cond(val, max_iters=100):
    last_assignments, assignments, i = val
    max_iters_reached = jnp.where(i >= max_iters, True, False)
    tolerance_achieved = jnp.allclose(last_assignments, assignments)
    first_iter = jnp.where(i > 0.0, False, True)
    return ~(max_iters_reached | tolerance_achieved) | first_iter

  def body(val):
    _, assignments, i = val

    mask = assignments[:,None] == jnp.arange(k)[None,:]
    data_in_clusters = data[:,:,None]*mask[:,None,:] # (batch,dim,cluster)
    means = data_in_clusters.mean(axis=0)
    distance_to_means = jnp.linalg.norm(data[:,:,None] - means[None,:,:], axis=1)
    new_assignments = distance_to_means.argmin(axis=1)
    return assignments, new_assignments, i + 1

  # Randomly initialize assignments
  rng_key = random.PRNGKey(1)
  assignments = random.randint(rng_key, minval=0, maxval=k, shape=data.shape[:1])

  val = (assignments, assignments, 0.0)
  _, assignments, i = jax.lax.while_loop(cond, body, val)
  return assignments

def spectral_cluster(W, k=2):
  D = jnp.diag(W@jnp.ones(W.shape[0]))
  L = D - W
  l, U = jnp.linalg.eigh(L)
  Uk = U[:,:k]
  cluster_assignments = kmeans(Uk)
  return cluster_assignments
