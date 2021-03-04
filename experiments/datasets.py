import jax.numpy as jnp
import jax
from jax import random
from functools import partial
import nux.util as util
from pathlib import Path
import os
import tensorflow_datasets as tfds
from experiments.datasetdownload import download_file_from_google_drive

def get_tf_dataset(*,
                   quantize_bits,
                   batch_size,
                   dataset_name,
                   n_batches=None,
                   split="train",
                   crop=False,
                   classification=False,
                   label_keep_percent=1.0,
                   random_label_percent=0.0):

  if random_label_percent > 0:
    assert 0, "Random labels not implemented for tf datasets"

  # https://github.com/tensorflow/datasets/issues/1441#issuecomment-581660890
  import resource
  low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

  import tensorflow as tf
  tf.config.set_visible_devices([], "GPU")
  import tensorflow_datasets as tfds

  if dataset_name == "cifar10":
    n_classes = 10
  elif dataset_name == "cifar100":
    n_classes = 100
  elif "imagenet" in dataset_name:
    n_classes = 1000
  elif "svhn" in dataset_name:
    n_classes = 10
  elif "mnist" in dataset_name:
    n_classes = 10
  else:
    n_classes = None

  def central_crop(x):
    x["image"] = x["image"][::2,::2][26:-19,12:-13]
    return x

  def to_float(x):
    x["image"] = tf.cast(x["image"], dtype=tf.float32)
    return x

  def random_flip(x):
    x["image"] = tf.image.random_flip_left_right(x["image"])
    return x

  def quantize(x):
    quantize_factor = 256/(2**quantize_bits)
    x["image"] = x["image"]//quantize_factor
    return x

  def to_expected_input(x):
    out = {"x": x["image"]}
    if classification == True and "label" in x:
      labels = x["label"]
      labels_one_hot = tf.one_hot(labels, n_classes)
      out["y"] = labels_one_hot
      out["y_non_one_hot"] = x["label"]
    return out

  rng = tf.random.Generator.from_seed(int(split == "train"))

  def make_semi_supervised(x):
    if "y_non_one_hot" in x:
      labels = x["y_non_one_hot"]
      x["y_is_labeled"] = rng.binomial(shape=labels.shape,
                                       counts=tf.ones(labels.shape),
                                       probs=tf.ones(labels.shape)*label_keep_percent)
      del x["y_non_one_hot"]
    return x

  ds = tfds.load(dataset_name, split=split)
  if crop:
    ds = ds.map(central_crop)

  ds = ds.map(to_float)
  ds = ds.map(random_flip)
  ds = ds.map(quantize)
  ds = ds.map(to_expected_input)

  if classification:
    ds = ds.map(make_semi_supervised)

  if split == "train":
    ds = ds.shuffle(15000).repeat() # TODO: Make dynamic.  Fails for larger datasets.

  ds = ds.batch(batch_size)
  if(n_batches is not None):
    ds = ds.batch(n_batches, drop_remainder=True)

  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  ds = ds.as_numpy_iterator()
  return ds

################################################################################################################

def get_mnist_dataset(quantize_bits=8,
                      batch_size=64,
                      n_batches=1000,
                      split="train",
                      classification=False,
                      label_keep_percent=1.0,
                      random_label_percent=0.0,
                      **kwargs):

  return get_tf_dataset(quantize_bits=quantize_bits,
                        batch_size=batch_size,
                        dataset_name="mnist",
                        n_batches=n_batches,
                        split=split,
                        crop=False,
                        classification=classification,
                        label_keep_percent=label_keep_percent,
                        random_label_percent=random_label_percent)

def get_fashion_mnist_dataset(quantize_bits=8,
                              batch_size=64,
                              n_batches=1000,
                              split="train",
                              classification=False,
                              label_keep_percent=1.0,
                              random_label_percent=0.0,
                              **kwargs):

  return get_tf_dataset(quantize_bits=quantize_bits,
                        batch_size=batch_size,
                        dataset_name="fashion_mnist",
                        n_batches=n_batches,
                        split=split,
                        crop=False,
                        classification=classification,
                        label_keep_percent=label_keep_percent,
                        random_label_percent=random_label_percent)

def get_cifar10_dataset(quantize_bits=8,
                        batch_size=64,
                        n_batches=1000,
                        split="train",
                        classification=False,
                        label_keep_percent=1.0,
                        random_label_percent=0.0,
                        **kwargs):

  return get_tf_dataset(quantize_bits=quantize_bits,
                        batch_size=batch_size,
                        dataset_name="cifar10",
                        n_batches=n_batches,
                        split=split,
                        crop=False,
                        classification=classification,
                        label_keep_percent=label_keep_percent,
                        random_label_percent=random_label_percent)

def get_cifar100_dataset(quantize_bits=8,
                         batch_size=64,
                         n_batches=1000,
                         split="train",
                         classification=False,
                         label_keep_percent=1.0,
                         random_label_percent=0.0,
                         **kwargs):

  return get_tf_dataset(quantize_bits=quantize_bits,
                        batch_size=batch_size,
                        dataset_name="cifar100",
                        n_batches=n_batches,
                        split=split,
                        crop=False,
                        classification=classification,
                        label_keep_percent=label_keep_percent,
                        random_label_percent=random_label_percent)

def get_svhn_dataset(quantize_bits=8,
                     batch_size=64,
                     n_batches=1000,
                     split="train",
                     classification=False,
                     label_keep_percent=1.0,
                     random_label_percent=0.0,
                     **kwargs):

  return get_tf_dataset(quantize_bits=quantize_bits,
                        batch_size=batch_size,
                        dataset_name="svhn_cropped",
                        n_batches=n_batches,
                        split=split,
                        crop=False,
                        classification=classification,
                        label_keep_percent=label_keep_percent,
                        random_label_percent=random_label_percent)

def get_celeba_dataset(quantize_bits=8,
                       batch_size=64,
                       n_batches=1000,
                       split="train",
                       classification=False,
                       label_keep_percent=1.0,
                       random_label_percent=0.0,
                       **kwargs):

  return get_tf_dataset(quantize_bits=quantize_bits,
                        batch_size=batch_size,
                        dataset_name="celeb_a",
                        n_batches=n_batches,
                        split=split,
                        crop=True,
                        classification=classification,
                        label_keep_percent=label_keep_percent,
                        random_label_percent=random_label_percent)

def get_celebahq_dataset(quantize_bits=8,
                         batch_size=64,
                         n_batches=1000,
                         split="train",
                         classification=False,
                         label_keep_percent=1.0,
                         random_label_percent=0.0,
                         **kwargs):
  dirtemp = os.path.expanduser("~/tensorflow_datasets/downloads/manual/data1024x1024.tar")
  Path(os.path.expanduser("~/tensorflow_datasets/downloads/manual/")).mkdir(parents=True, exist_ok=True)

  if(not os.path.exists(dirtemp)):
    print("Celeb_ahq not detected, proceeding with download")
    download_file_from_google_drive("1aNQw43R0EV4v9EJDFBFX7ZYEZuPpfo-v", dirtemp)
    print("download finished")

  return get_tf_dataset(quantize_bits=quantize_bits,
                        batch_size=batch_size,
                        dataset_name="celeb_a_hq",
                        n_batches=n_batches,
                        split=split,
                        crop=False,
                        classification=classification,
                        label_keep_percent=label_keep_percent,
                        random_label_percent=random_label_percent)

def get_imagenet_dataset(quantize_bits=8,
                         batch_size=64,
                         n_batches=1000,
                         split="train",
                         classification=False,
                         label_keep_percent=1.0,
                         random_label_percent=0.0,
                         **kwargs):

  if split == "test":
    split = "validation"

  return get_tf_dataset(quantize_bits=quantize_bits,
                        batch_size=batch_size,
                        dataset_name="imagenet_resized/64x64",
                        n_batches=n_batches,
                        split=split,
                        crop=False,
                        classification=classification,
                        label_keep_percent=label_keep_percent,
                        random_label_percent=random_label_percent)

################################################################################################################

def get_regular_dataset(data,
                        labels=None,
                        batch_size=32,
                        n_batches=None,
                        split="train",
                        train_ratio=0.7,
                        classification=False,
                        label_keep_percent=1.0,
                        random_label_percent=0.0):

  n_train = int(data.shape[0]*train_ratio)
  if classification and labels is not None:
    n_classes = len(set(labels.ravel()))
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
      data_batch = data[batch_idx]
      inputs = {"x": data_batch}

      if classification and labels is not None:
        y = labels[batch_idx]
        y_one_hot = (y[...,None] == range_classes[...,:])*1.0

        inputs["y"] = y_one_hot
        inputs["y_is_labeled"] = labels_to_keep[batch_idx]

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
        y_one_hot = (y[...,None] == range_classes[...,:])*1.0

        inputs["y"] = y_one_hot

      yield inputs

      start_idx += big_batch_size

      if start_idx >= data.shape[0]:
        # Stop iterating
        return

  if split == "train":
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
                           **kwargs):
  from sklearn.datasets import make_swiss_roll
  data = make_swiss_roll(n_samples=20000, noise=0.3, random_state=0)[0][:,[0,2]]
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
                             label_keep_percent=label_keep_percent)

def get_moons_dataset(batch_size=32,
                      n_batches=None,
                      split="train",
                      train_ratio=0.7,
                      classification=False,
                      label_keep_percent=1.0,
                      random_label_percent=0.0,
                      **kwargs):
  from sklearn.datasets import make_moons
  data, labels = make_moons(n_samples=20000, noise=0.07, random_state=0)
  data, labels = jnp.array(data), jnp.array(labels)

  return get_regular_dataset(data,
                             labels=labels,
                             batch_size=batch_size,
                             n_batches=n_batches,
                             split=split,
                             train_ratio=train_ratio,
                             classification=classification,
                             label_keep_percent=label_keep_percent,
                             random_label_percent=random_label_percent)

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
                        **kwargs):
  key = random.PRNGKey(0)
  data, labels = generate_nested_circles(key, n_samples=20000)

  return get_regular_dataset(data,
                             labels=labels,
                             batch_size=batch_size,
                             n_batches=n_batches,
                             split=split,
                             train_ratio=train_ratio,
                             classification=classification,
                             label_keep_percent=label_keep_percent,
                             random_label_percent=random_label_percent)

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
                     **kwargs):
  key = random.PRNGKey(0)
  data = generate_grid(key, n_samples=20000, min_val=-10, max_val=10, n_clusters_per_axis=4)
  return get_regular_dataset(data,
                             labels=None,
                             batch_size=batch_size,
                             n_batches=n_batches,
                             split=split,
                             train_ratio=train_ratio,
                             classification=False,
                             label_keep_percent=label_keep_percent)

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
                               **kwargs):
  key = random.PRNGKey(0)
  data, labels = gen_all_clusters(key, n_samples=20000)

  return get_regular_dataset(data,
                             labels=labels,
                             batch_size=batch_size,
                             n_batches=n_batches,
                             split=split,
                             train_ratio=train_ratio,
                             classification=classification,
                             label_keep_percent=label_keep_percent,
                             random_label_percent=random_label_percent)

################################################################################################################

def make_train_test(dataset_fun,
                    train_batch_size,
                    train_n_batches,
                    test_batch_size,
                    test_n_batches,
                    quantize_bits=8,
                    classification=False,
                    label_keep_percent=1.0,
                    random_label_percent=0.0):
  train_ds = dataset_fun(quantize_bits=quantize_bits,
                         batch_size=train_batch_size,
                         n_batches=train_n_batches,
                         split="train",
                         classification=classification,
                         label_keep_percent=label_keep_percent,
                         random_label_percent=random_label_percent)

  get_test_ds = lambda : dataset_fun(quantize_bits=quantize_bits,
                                     batch_size=test_batch_size,
                                     n_batches=test_n_batches,
                                     split="test",
                                     classification=classification,
                                     label_keep_percent=1.0,
                                     random_label_percent=0.0)

  return train_ds, get_test_ds

def get_dataset(dataset_name,
                train_batch_size,
                train_n_batches,
                test_batch_size,
                test_n_batches,
                quantize_bits=8,
                classification=False,
                label_keep_percent=1.0,
                random_label_percent=0.0):

  kwargs = dict(train_batch_size=train_batch_size,
                train_n_batches=train_n_batches,
                test_batch_size=test_batch_size,
                test_n_batches=test_n_batches,
                quantize_bits=quantize_bits,
                classification=classification,
                label_keep_percent=label_keep_percent,
                random_label_percent=random_label_percent)

  if dataset_name == "mnist":
    train_ds, get_test_ds = make_train_test(get_mnist_dataset, **kwargs)
  elif dataset_name == "fashion_mnist":
    train_ds, get_test_ds = make_train_test(get_fashion_mnist_dataset, **kwargs)
  elif dataset_name == "cifar10":
    train_ds, get_test_ds = make_train_test(get_cifar10_dataset, **kwargs)
  elif dataset_name == "cifar100":
    train_ds, get_test_ds = make_train_test(get_cifar100_dataset, **kwargs)
  elif dataset_name == "svhn":
    train_ds, get_test_ds = make_train_test(get_svhn_dataset, **kwargs)
  elif dataset_name == "celeb_a":
    train_ds, get_test_ds = make_train_test(get_celeba_dataset, **kwargs)
  elif dataset_name == "celeb_ahq":
    train_ds, get_test_ds = make_train_test(get_celebahq_dataset, **kwargs)
  elif dataset_name == "imagenet":
    train_ds, get_test_ds = make_train_test(get_imagenet_dataset, **kwargs)
  elif dataset_name == "swiss_roll":
    if classification:
      assert 0, "swiss_roll has no labels"
    train_ds, get_test_ds = make_train_test(get_swiss_roll_dataset, **kwargs)
  elif dataset_name == "grid":
    if classification:
      assert 0, "grid has no labels"
    train_ds, get_test_ds = make_train_test(get_grid_dataset, **kwargs)
  elif dataset_name == "moons":
    train_ds, get_test_ds = make_train_test(get_moons_dataset, **kwargs)
  elif dataset_name == "circles":
    train_ds, get_test_ds = make_train_test(get_circles_dataset, **kwargs)
  elif dataset_name == "swirl_clusters":
    train_ds, get_test_ds = make_train_test(get_swirl_clusters_dataset, **kwargs)
  else:
    assert 0, "Invalid dataset name"

  return train_ds, get_test_ds