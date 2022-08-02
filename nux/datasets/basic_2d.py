import jax.numpy as jnp
import jax
from jax import random
from functools import partial
import nux.util as util
from pathlib import Path
import os

def get_regular_dataset(data,
                        labels=None,
                        batch_size=32,
                        n_batches=None,
                        split="train",
                        train_ratio=0.7,
                        classification=False,
                        label_keep_percent=1.0,
                        random_label_percent=0.0,
                        data_augmentation=False,
                        one_hot_labels=True,
                        **kwargs):

  n_train = int(data.shape[0]*train_ratio)
  if classification and labels is not None:
    n_classes = len(jnp.unique(labels))
    range_classes = jnp.arange(n_classes)

  key = random.PRNGKey(0)

  # Make some of the data points have random labels
  if classification and random_label_percent > 0.0:
    k1, k2 = random.split(key, 2)
    n_random_labels = int(labels.shape[0]*random_label_percent)
    random_indices = random.randint(k1, minval=0, maxval=labels.shape[0], shape=(n_random_labels,))
    random_labels = random.randint(k2, minval=0, maxval=n_classes, shape=(n_random_labels,))
    labels = jax.ops.index_update(labels, random_indices, random_labels)

  # Mask some of the labels
  if classification:
    labels_to_keep = random.bernoulli(key, label_keep_percent, shape=labels.shape)

  def get_train_ds(key=None):
    batch_shape = (batch_size,) if n_batches is None else (n_batches, batch_size)

    if key is None:
      key = random.PRNGKey(0)

    while True:
      key, data_key = random.split(key, 2)
      batch_idx = random.randint(data_key, minval=0, maxval=n_train, shape=batch_shape)
      data_batch = data[batch_idx,...]
      inputs = {"x": data_batch}

      if classification and labels is not None:
        y = labels[batch_idx,...]
        if one_hot_labels:
          y_one_hot = (y[...,None] == range_classes[...,:])*1.0
          inputs["y"] = y_one_hot
        else:
          inputs["y"] = y
        inputs["y_is_labeled"] = labels_to_keep[batch_idx,...]

      yield inputs

  def rebatch(data_batch, batch_size, n_batches):
    adjusted_n_batches = data_batch.shape[0]//batch_size
    data_batch = data_batch[:adjusted_n_batches*batch_size]
    data_batch = data_batch.reshape((adjusted_n_batches, batch_size) + data_batch.shape[1:])
    return data_batch

  def get_test_ds():
    batch_shape = (batch_size,) if n_batches is None else (n_batches, batch_size)

    big_batch_size = util.list_prod(batch_shape)
    start_idx = n_train
    get_end_idx = lambda start: start + big_batch_size

    while True:
      end_idx = get_end_idx(start_idx)

      data_batch = data[start_idx:end_idx]
      if n_batches is not None:
        data_batch = rebatch(data_batch, batch_size, n_batches)
      inputs = {"x": data_batch}

      if classification and labels is not None:
        y = labels[start_idx:end_idx]
        if n_batches is not None:
          y = rebatch(y, batch_size, n_batches)
        if one_hot_labels:
          y_one_hot = (y[...,None] == range_classes[...,:])*1.0
          inputs["y"] = y_one_hot
        else:
          inputs["y"] = y

      yield inputs

      start_idx += big_batch_size

      if start_idx >= data.shape[0]:
        # Stop iterating
        return

  if "train" in split:
    return get_train_ds()

  return get_test_ds()

################################################################################################################

def get_swiss_roll_dataset(batch_size=32,
                           n_batches=None,
                           split="train",
                           train_ratio=0.7,
                           classification=False,
                           label_keep_percent=1.0,
                           random_label_percent=0.0,
                           data_augmentation=False,
                           **kwargs):
  from sklearn.datasets import make_swiss_roll
  data = make_swiss_roll(n_samples=1000000, noise=0.3, random_state=0)[0][:,[0,2]]
  data = jnp.array(data)#/10.0
  key = random.PRNGKey(0)
  data = random.permutation(key, data)

  return get_regular_dataset(data,
                             labels=None,
                             batch_size=batch_size,
                             n_batches=n_batches,
                             split=split,
                             train_ratio=train_ratio,
                             classification=False,
                             label_keep_percent=label_keep_percent,
                             data_augmentation=False,
                             **kwargs)

def get_moons_dataset(batch_size=32,
                      n_batches=None,
                      split="train",
                      train_ratio=0.7,
                      classification=False,
                      label_keep_percent=1.0,
                      random_label_percent=0.0,
                      data_augmentation=False,
                      **kwargs):
  from sklearn.datasets import make_moons
  data, labels = make_moons(n_samples=1000000, noise=0.07, random_state=0)
  data, labels = jnp.array(data), jnp.array(labels)

  return get_regular_dataset(data,
                             labels=labels,
                             batch_size=batch_size,
                             n_batches=n_batches,
                             split=split,
                             train_ratio=train_ratio,
                             classification=classification,
                             label_keep_percent=label_keep_percent,
                             random_label_percent=random_label_percent,
                             data_augmentation=False,
                             **kwargs)

################################################################################################################

def generate_cosine(key, n_samples):
  x = jnp.linspace(-jnp.pi, jnp.pi, n_samples)
  y = jnp.cos(x)
  data = jnp.vstack([x, y]).T

  k1, k2 = random.split(key, 2)
  data = random.permutation(k1, data)
  data += random.normal(k2, data.shape)*0.2
  return data

def get_cosine_dataset(batch_size=32,
                       n_batches=None,
                       split="train",
                       train_ratio=0.7,
                       classification=False,
                       label_keep_percent=1.0,
                       random_label_percent=0.0,
                       data_augmentation=False,
                       **kwargs):
  key = random.PRNGKey(0)
  data = generate_cosine(key, n_samples=1000000)

  return get_regular_dataset(data,
                             labels=None,
                             batch_size=batch_size,
                             n_batches=n_batches,
                             split=split,
                             train_ratio=train_ratio,
                             classification=False,
                             label_keep_percent=label_keep_percent,
                             data_augmentation=False,
                             **kwargs)

################################################################################################################

def generate_double_roll(key, n_samples):
  theta = jnp.linspace(0.0, 2*jnp.pi, n_samples//2)
  r = jnp.linspace(0.0, 1.0, n_samples//2)
  x, y = r*jnp.cos(theta), r*jnp.sin(theta)
  data = jnp.vstack([x, y]).T
  data = jnp.vstack([data, -data])

  k1, k2 = random.split(key, 2)
  data = random.permutation(k1, data)
  data += random.normal(k2, data.shape)*0.05
  return data

def get_double_roll_dataset(batch_size=32,
                            n_batches=None,
                            split="train",
                            train_ratio=0.7,
                            classification=False,
                            label_keep_percent=1.0,
                            random_label_percent=0.0,
                            data_augmentation=False,
                            **kwargs):
  key = random.PRNGKey(0)
  data = generate_double_roll(key, n_samples=1000000)

  return get_regular_dataset(data,
                             labels=None,
                             batch_size=batch_size,
                             n_batches=n_batches,
                             split=split,
                             train_ratio=train_ratio,
                             classification=False,
                             label_keep_percent=label_keep_percent,
                             data_augmentation=False,
                             **kwargs)

################################################################################################################

def generate_3d_spiral(key, n_samples):
  z = jnp.linspace(-3, 3, n_samples)
  r = jnp.ones(n_samples)
  theta = random.normal(key, shape=(n_samples,))*2.0
  theta = jnp.sort(theta)
  x, y = r*jnp.cos(theta), r*jnp.sin(theta)
  data = jnp.vstack([x, y, z]).T
  data = random.permutation(key, data)
  data += random.normal(key, shape=data.shape)*0.2
  return data

def get_spiral_3d_dataset(batch_size=32,
                          n_batches=None,
                          split="train",
                          train_ratio=0.7,
                          classification=False,
                          label_keep_percent=1.0,
                          random_label_percent=0.0,
                          data_augmentation=False,
                          **kwargs):
  key = random.PRNGKey(0)
  data = generate_3d_spiral(key, n_samples=1000000)

  return get_regular_dataset(data,
                             labels=None,
                             batch_size=batch_size,
                             n_batches=n_batches,
                             split=split,
                             train_ratio=train_ratio,
                             classification=False,
                             label_keep_percent=label_keep_percent,
                             data_augmentation=False,
                             **kwargs)

################################################################################################################

def generate_line(key,
                  n_samples):
  k1, k2 = random.split(key, 2)
  x = (random.normal(k1, (n_samples,)) + 1)*1.3
  y = jnp.exp(-x*jnp.sin(x)**2)
  data = jnp.vstack([x, y]).T
  noise = random.normal(k2, data.shape)
  data += noise*0.06

  return data

def get_2d_manifold_dataset(batch_size=32,
                            n_batches=None,
                            split="train",
                            train_ratio=0.7,
                            classification=False,
                            label_keep_percent=1.0,
                            random_label_percent=0.0,
                            data_augmentation=False,
                            **kwargs):
  key = random.PRNGKey(0)
  data = generate_line(key, n_samples=1000000)

  return get_regular_dataset(data,
                             labels=None,
                             batch_size=batch_size,
                             n_batches=n_batches,
                             split=split,
                             train_ratio=train_ratio,
                             classification=False,
                             label_keep_percent=label_keep_percent,
                             random_label_percent=random_label_percent,
                             data_augmentation=False,
                             **kwargs)

################################################################################################################

def generate_6_points(key,
                  n_samples):
  k1, k2 = random.split(key, 2)
  t = random.normal(k1, (n_samples,))
  b = 1.0
  k = 6.0
  a = k*b
  x = (a - b)*jnp.cos(t) + b*jnp.cos(t*(k - 1))
  y = (a - b)*jnp.sin(t) - b*jnp.sin(t*(k - 1))
  data = jnp.vstack([x, y]).T
  noise = random.normal(k2, data.shape)
  data += noise*0.06

  return data

def get_6_point_manifold_dataset(batch_size=32,
                                 n_batches=None,
                                 split="train",
                                 train_ratio=0.7,
                                 classification=False,
                                 label_keep_percent=1.0,
                                 random_label_percent=0.0,
                                 data_augmentation=False,
                                 **kwargs):
  key = random.PRNGKey(0)
  data = generate_6_points(key, n_samples=1000000)

  return get_regular_dataset(data,
                             labels=None,
                             batch_size=batch_size,
                             n_batches=n_batches,
                             split=split,
                             train_ratio=train_ratio,
                             classification=False,
                             label_keep_percent=label_keep_percent,
                             random_label_percent=random_label_percent,
                             data_augmentation=False,
                             **kwargs)

################################################################################################################

def generate_nested_circles(key,
                            n_samples,
                            inner_radius=2,
                            outer_radius=4,
                            noise=0.15):

  k1, k2, k3, k4 = random.split(key, 4)

  # Generate the circles
  inner_t = random.uniform(k1, shape=(n_samples//2,))*2*jnp.pi
  inner_circle = inner_radius*jnp.vstack([jnp.cos(inner_t), jnp.sin(inner_t)])

  outer_t = random.uniform(k2, shape=(n_samples//2,))*2*jnp.pi
  outer_circle = outer_radius*jnp.vstack([jnp.cos(outer_t), jnp.sin(outer_t)])

  data = jnp.vstack([inner_circle.T, outer_circle.T])

  # Keep track of the labels
  y = jnp.hstack([jnp.zeros(n_samples//2), jnp.ones(n_samples//2)])

  # Shuffle the data
  idx = jnp.arange(n_samples)
  idx = random.permutation(k3, idx)
  data = data[idx]
  y = y[idx]

  data += random.normal(k4, data.shape)*noise
  return data, y

def get_circles_dataset(batch_size=32,
                        n_batches=None,
                        split="train",
                        train_ratio=0.7,
                        classification=False,
                        label_keep_percent=1.0,
                        random_label_percent=0.0,
                        data_augmentation=False,
                        **kwargs):
  key = random.PRNGKey(0)
  data, labels = generate_nested_circles(key, n_samples=1000000)

  return get_regular_dataset(data,
                             labels=labels,
                             batch_size=batch_size,
                             n_batches=n_batches,
                             split=split,
                             train_ratio=train_ratio,
                             classification=classification,
                             label_keep_percent=label_keep_percent,
                             random_label_percent=random_label_percent,
                             data_augmentation=False,
                             **kwargs)

################################################################################################################

def generate_grid(key,
                  n_samples,
                  min_val,
                  max_val,
                  n_clusters_per_axis):
  x, y = jnp.linspace(min_val, max_val, n_clusters_per_axis), jnp.linspace(min_val, max_val, n_clusters_per_axis)
  X, Y = jnp.meshgrid(x, y)
  xy = jnp.dstack([X, Y]).reshape((-1, 2))

  # Repeat the data so that we can add noise to different copies
  n_repeats = n_samples//(n_clusters_per_axis**2)
  data = jnp.repeat(xy, repeats=n_repeats, axis=0)

  # Add just enough noise so that we see each cluster without overlapping
  std = (max_val - min_val)/n_clusters_per_axis*0.25

  noise = random.normal(key, data.shape)*std
  data += noise

  data = random.permutation(key, data)
  return data

def get_grid_dataset(batch_size=32,
                     n_batches=None,
                     split="train",
                     train_ratio=0.7,
                     classification=False,
                     label_keep_percent=1.0,
                     random_label_percent=0.0,
                     data_augmentation=False,
                     **kwargs):
  key = random.PRNGKey(0)
  data = generate_grid(key, n_samples=1000000, min_val=-10, max_val=10, n_clusters_per_axis=4)
  return get_regular_dataset(data,
                             labels=None,
                             batch_size=batch_size,
                             n_batches=n_batches,
                             split=split,
                             train_ratio=train_ratio,
                             classification=False,
                             label_keep_percent=label_keep_percent,
                             **kwargs)

################################################################################################################

@jax.jit
def rotate(x, theta):
    s, c = jnp.sin(theta), jnp.cos(theta)
    rotation_matrix = jnp.array([[c, -s], [s, c]])
    return jnp.dot(x, rotation_matrix.T)

def gen_cluster(key,
                theta,
                stretch,
                scale,
                n_samples):
    # Generate a cluster that is rotated by theta and stretched by scale
    basis = jnp.array([[1, 0], [0, 1]])
    rotated_basis = rotate(basis, theta)

    cov = rotated_basis@(scale*jnp.array([[stretch, 0], [0, 1]]))@jnp.linalg.inv(rotated_basis)
    cov_chol = jnp.linalg.cholesky(cov)

    noise = random.normal(key, shape=(n_samples, 2))
    data = jnp.dot(noise, cov_chol.T)
    return data

def gen_all_clusters(key,
                     n_samples,
                     n_clusters=5,
                     stretch=10,
                     scale=0.1,
                     radius=3,
                     swirl_theta=jnp.pi/2):
    k1, k2 = random.split(key, 2)
    points_per_cluster = n_samples//n_clusters

    thetas = jnp.linspace(0, 2*jnp.pi, n_clusters + 1)[:-1]
    means = radius*jnp.vstack([jnp.cos(thetas), jnp.sin(thetas)]).T

    keys = random.split(k1, n_clusters)
    cluster_fun = jax.vmap(partial(gen_cluster, stretch=stretch, scale=scale, n_samples=points_per_cluster))
    clusters = cluster_fun(keys, -thetas)
    clusters += means[:,None,:]

    data = jnp.concatenate(clusters)
    y = jnp.concatenate([jnp.ones(points_per_cluster)*i for i in range(n_clusters)])

    # Shuffle the data
    idx = jnp.arange(n_samples)
    idx = random.permutation(k2, idx)
    data = data[idx]
    y = y[idx]

    norms = jnp.linalg.norm(data, axis=1)**2
    min_norm, max_norm = jnp.min(norms), jnp.max(norms)
    thetas = (norms - min_norm)/max_norm*swirl_theta

    data = jax.vmap(rotate)(data, -thetas)

    return data, y

def get_swirl_clusters_dataset(batch_size=32,
                               n_batches=None,
                               split="train",
                               train_ratio=0.7,
                               classification=False,
                               label_keep_percent=1.0,
                               random_label_percent=0.0,
                               data_augmentation=False,
                               **kwargs):
  key = random.PRNGKey(0)
  data, labels = gen_all_clusters(key, n_samples=1000000)

  return get_regular_dataset(data,
                             labels=labels,
                             batch_size=batch_size,
                             n_batches=n_batches,
                             split=split,
                             train_ratio=train_ratio,
                             classification=classification,
                             label_keep_percent=label_keep_percent,
                             random_label_percent=random_label_percent,
                             data_augmentation=False,
                             **kwargs)
