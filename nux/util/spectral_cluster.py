import jax
import jax.numpy as jnp
import jax.random as random

def kmeans(data, k=2):
  assert data.ndim == 2

  def cond(val, max_iters=100):
    last_assignments, assignments, means, i = val
    max_iters_reached = jnp.where(i >= max_iters, True, False)
    tolerance_achieved = jnp.allclose(last_assignments, assignments)
    first_iter = jnp.where(i > 0.0, False, True)
    return ~(max_iters_reached | tolerance_achieved) | first_iter

  def body(val):
    _, last_assignments, means, i = val

    # Find the assignments to the closest mean
    distance_to_means = jnp.linalg.norm(data[:,:,None] - means[None,:,:], axis=1)
    assignments = distance_to_means.argmin(axis=1)

    # Update the means
    mask = assignments[:,None] == jnp.arange(k)[None,:]
    data_in_clusters = data[:,:,None]*mask[:,None,:] # (batch,dim,cluster)
    new_means = data_in_clusters.mean(axis=0) # (dim,cluster)

    return last_assignments, assignments, new_means, i + 1

  # Randomly initialize assignments
  rng_key = random.PRNGKey(0)
  k1, k2 = random.split(rng_key, 2)
  means = random.normal(k1, (data.shape[1], k))
  assignments = random.randint(k2, minval=0, maxval=k, shape=data.shape[:1])

  val = (assignments, assignments, means, 0.0)
  # body(val)
  _, assignments, means, i = jax.lax.while_loop(cond, body, val)
  return assignments

def spectral_cluster(W, k=2):
  D = jnp.diag(W.sum(axis=-1))
  L = D - W
  l, U = jnp.linalg.eigh(L)
  Uk = U[:,1:k+1]
  cluster_assignments = kmeans(Uk, k=k)
  return cluster_assignments
