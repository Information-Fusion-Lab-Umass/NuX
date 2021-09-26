import jax.numpy as jnp
import jax
from jax import random
from functools import partial
import nux.util as util
from pathlib import Path
import os
import tensorflow_datasets as tfds
from .util.datasetdownload import download_file_from_google_drive

def get_tf_image_dataset(*,
                         quantize_bits,
                         batch_size,
                         dataset_name,
                         n_batches=None,
                         split="train",
                         crop=False,
                         classification=False,
                         label_keep_percent=1.0,
                         random_label_percent=0.0,
                         data_augmentation=False):

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
    x["image"] = x["image"][::2,::2][26:-19,12:-13]#[::2,::2]
    if "augmented" in x:
      x["augmented"] = x["augmented"][::2,::2][26:-19,12:-13]#[::2,::2]

    if dataset_name == "celeb_a32x32":
      x["image"] = x["image"][::2,::2]
      if "augmented" in x:
        x["augmented"] = x["augmented"][::2,::2]

    return x

  def to_float(x):
    x["image"] = tf.cast(x["image"], dtype=tf.float32)
    if "augmented" in x:
      x["augmented"] = tf.cast(x["augmented"], dtype=tf.float32)
    return x

  def random_flip(x):
    x["image"] = tf.image.random_flip_left_right(x["image"])
    return x

  def random_augmentation(x):
    im = x["image"]
    H, W = im.shape[:2]
    im = tf.image.random_brightness(im, 0.2)
    im = tf.image.random_contrast(im, 0.8, 1.2)
    im = tf.image.random_hue(im, 0.01)
    im = tf.image.random_jpeg_quality(im, 50, 100)

    # im = tf.image.random_flip_left_right(im)

    crop_height, crop_width = int(0.9*H), int(0.9*W)
    im = tf.image.random_crop(value=im, size=(crop_height, crop_width, 3))
    im = tf.image.resize(im, size=(H, W))

    x["augmented"] = im
    return x

  def quantize(x):
    quantize_factor = 256/(2**quantize_bits)
    x["image"] = x["image"]//quantize_factor

    if "augmented" in x:
      x["augmented"] = x["augmented"]//quantize_factor

    return x

  def to_expected_input(x):
    out = {"x": x["image"]}
    if classification == True and "label" in x:
      labels = x["label"]
      labels_one_hot = tf.one_hot(labels, n_classes)
      out["y"] = labels_one_hot
      out["y_non_one_hot"] = x["label"]

    if data_augmentation and "augmented" in x:
      out["x_aug"] = x["augmented"]
    return out

  rng = tf.random.Generator.from_seed(int("train" in split))

  def make_semi_supervised(x):
    if "y_non_one_hot" in x:
      labels = x["y_non_one_hot"]
      x["y_is_labeled"] = rng.binomial(shape=labels.shape,
                                       counts=tf.ones(labels.shape),
                                       probs=tf.ones(labels.shape)*label_keep_percent)
      del x["y_non_one_hot"]
    return x

  # TODO: Get rid of this hack
  if dataset_name == "celeb_a32x32":
    ds = tfds.load("celeb_a", split=split)
  else:
    ds = tfds.load(dataset_name, split=split)

  if crop:
    ds = ds.map(central_crop)

  if "train" in split and "mnist" not in dataset_name:
    ds = ds.map(random_flip)

  if data_augmentation:
    ds = ds.map(random_augmentation)

  ds = ds.map(to_float)
  ds = ds.map(quantize)
  ds = ds.map(to_expected_input)

  if classification:
    ds = ds.map(make_semi_supervised)

  if "train" in split:
    if "celeb_a_hq" in dataset_name:
      ds = ds.repeat().shuffle(1)
    else:
      ds = ds.repeat().shuffle(15000) # Don't want hard epoch splits
    # ds = ds.shuffle(15000).repeat()

  ds = ds.batch(batch_size)
  if(n_batches is not None):
    ds = ds.batch(n_batches, drop_remainder=True)

  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  ds = ds.as_numpy_iterator()

  if random_augmentation and False:
    # Apply some final augmentation steps that only work on numpy arrays
    class final_ds():
      def __init__(self, ds):
        self.ds = ds
        self.rng_key = random.PRNGKey(0)

      def __iter__(self):
        key, self.rng_key = random.split(self.rng_key, 2)

        data = next(self.ds)
        x_aug = data["x_aug"]

        # Random scale and translation
        fun = lambda x: jax.image.scale_and_translate(x,
                                                      shape=x.shape,
                                                      spatial_dims=(0, 1),
                                                      scale=jnp.array([1.0, 1.0]),
                                                      translation=jnp.array([10, 10]),
                                                      method="linear")
        x_aug = jax.vmap(fun)(x_aug)


        data["x_aug"] = x_aug
        yield data

    fds = final_ds(ds)
    return iter(fds)

  return ds

################################################################################################################

def get_mnist_dataset(quantize_bits=8,
                      batch_size=64,
                      n_batches=1000,
                      split="train",
                      classification=False,
                      label_keep_percent=1.0,
                      random_label_percent=0.0,
                      data_augmentation=False,
                      **kwargs):

  return get_tf_image_dataset(quantize_bits=quantize_bits,
                        batch_size=batch_size,
                        dataset_name="mnist",
                        n_batches=n_batches,
                        split=split,
                        crop=False,
                        classification=classification,
                        label_keep_percent=label_keep_percent,
                        random_label_percent=random_label_percent,
                        data_augmentation=data_augmentation,
                        **kwargs)

def get_fashion_mnist_dataset(quantize_bits=8,
                              batch_size=64,
                              n_batches=1000,
                              split="train",
                              classification=False,
                              label_keep_percent=1.0,
                              random_label_percent=0.0,
                              data_augmentation=False,
                              **kwargs):

  return get_tf_image_dataset(quantize_bits=quantize_bits,
                        batch_size=batch_size,
                        dataset_name="fashion_mnist",
                        n_batches=n_batches,
                        split=split,
                        crop=False,
                        classification=classification,
                        label_keep_percent=label_keep_percent,
                        random_label_percent=random_label_percent,
                        data_augmentation=data_augmentation,
                        **kwargs)

def get_cifar10_dataset(quantize_bits=8,
                        batch_size=64,
                        n_batches=1000,
                        split="train",
                        classification=False,
                        label_keep_percent=1.0,
                        random_label_percent=0.0,
                        data_augmentation=False,
                        **kwargs):

  return get_tf_image_dataset(quantize_bits=quantize_bits,
                        batch_size=batch_size,
                        dataset_name="cifar10",
                        n_batches=n_batches,
                        split=split,
                        crop=False,
                        classification=classification,
                        label_keep_percent=label_keep_percent,
                        random_label_percent=random_label_percent,
                        data_augmentation=data_augmentation,
                        **kwargs)

def get_cifar100_dataset(quantize_bits=8,
                         batch_size=64,
                         n_batches=1000,
                         split="train",
                         classification=False,
                         label_keep_percent=1.0,
                         random_label_percent=0.0,
                         data_augmentation=False,
                         **kwargs):

  return get_tf_image_dataset(quantize_bits=quantize_bits,
                        batch_size=batch_size,
                        dataset_name="cifar100",
                        n_batches=n_batches,
                        split=split,
                        crop=False,
                        classification=classification,
                        label_keep_percent=label_keep_percent,
                        random_label_percent=random_label_percent,
                        data_augmentation=data_augmentation,
                        **kwargs)

def get_svhn_dataset(quantize_bits=8,
                     batch_size=64,
                     n_batches=1000,
                     split="train",
                     classification=False,
                     label_keep_percent=1.0,
                     random_label_percent=0.0,
                     data_augmentation=False,
                     **kwargs):

  return get_tf_image_dataset(quantize_bits=quantize_bits,
                        batch_size=batch_size,
                        dataset_name="svhn_cropped",
                        n_batches=n_batches,
                        split=split,
                        crop=False,
                        classification=classification,
                        label_keep_percent=label_keep_percent,
                        random_label_percent=random_label_percent,
                        data_augmentation=data_augmentation,
                        **kwargs)

def get_celeba_dataset(quantize_bits=8,
                       batch_size=64,
                       n_batches=1000,
                       split="train",
                       classification=False,
                       label_keep_percent=1.0,
                       random_label_percent=0.0,
                       data_augmentation=False,
                       **kwargs):

  return get_tf_image_dataset(quantize_bits=quantize_bits,
                        batch_size=batch_size,
                        dataset_name="celeb_a",
                        n_batches=n_batches,
                        split=split,
                        crop=True,
                        classification=classification,
                        label_keep_percent=label_keep_percent,
                        random_label_percent=random_label_percent,
                        data_augmentation=data_augmentation,
                        **kwargs)

def get_celeb_a32x32_dataset(quantize_bits=8,
                       batch_size=64,
                       n_batches=1000,
                       split="train",
                       classification=False,
                       label_keep_percent=1.0,
                       random_label_percent=0.0,
                       data_augmentation=False,
                       **kwargs):

  return get_tf_image_dataset(quantize_bits=quantize_bits,
                        batch_size=batch_size,
                        dataset_name="celeb_a32x32",
                        n_batches=n_batches,
                        split=split,
                        crop=True,
                        classification=classification,
                        label_keep_percent=label_keep_percent,
                        random_label_percent=random_label_percent,
                        data_augmentation=data_augmentation,
                        **kwargs)


def get_celebahq_dataset(quantize_bits=8,
                         batch_size=64,
                         n_batches=1000,
                         split="train",
                         classification=False,
                         label_keep_percent=1.0,
                         random_label_percent=0.0,
                         data_augmentation=False,
                         **kwargs):
  dirtemp = os.path.expanduser("~/tensorflow_datasets/downloads/manual/data1024x1024.tar")
  Path(os.path.expanduser("~/tensorflow_datasets/downloads/manual/")).mkdir(parents=True, exist_ok=True)

  if(not os.path.exists(dirtemp)):
    print("Celeb_ahq not detected, proceeding with download")
    download_file_from_google_drive("1aNQw43R0EV4v9EJDFBFX7ZYEZuPpfo-v", dirtemp)
    print("download finished")

  return get_tf_image_dataset(quantize_bits=quantize_bits,
                        batch_size=batch_size,
                        dataset_name="celeb_a_hq",
                        n_batches=n_batches,
                        split=split,
                        crop=False,
                        classification=classification,
                        label_keep_percent=label_keep_percent,
                        random_label_percent=random_label_percent,
                        data_augmentation=data_augmentation,
                        **kwargs)

def get_imagenet_dataset(quantize_bits=8,
                         batch_size=64,
                         n_batches=1000,
                         split="train",
                         classification=False,
                         label_keep_percent=1.0,
                         random_label_percent=0.0,
                         data_augmentation=False,
                         **kwargs):

  if split == "test":
    split = "validation"

  return get_tf_image_dataset(quantize_bits=quantize_bits,
                        batch_size=batch_size,
                        dataset_name="imagenet_resized/64x64",
                        n_batches=n_batches,
                        split=split,
                        crop=False,
                        classification=classification,
                        label_keep_percent=label_keep_percent,
                        random_label_percent=random_label_percent,
                        data_augmentation=data_augmentation,
                        **kwargs)

def get_imagenet32_dataset(quantize_bits=8,
                           batch_size=64,
                           n_batches=1000,
                           split="train",
                           classification=False,
                           label_keep_percent=1.0,
                           random_label_percent=0.0,
                           data_augmentation=False,
                           **kwargs):

  if split == "test":
    split = "validation"

  return get_tf_image_dataset(quantize_bits=quantize_bits,
                        batch_size=batch_size,
                        dataset_name="imagenet_resized/32x32",
                        n_batches=n_batches,
                        split=split,
                        crop=False,
                        classification=classification,
                        label_keep_percent=label_keep_percent,
                        random_label_percent=random_label_percent,
                        data_augmentation=data_augmentation,
                        **kwargs)

def get_downsampled_imagenet_dataset(quantize_bits=8,
                                     batch_size=64,
                                     n_batches=1000,
                                     split="train",
                                     classification=False,
                                     label_keep_percent=1.0,
                                     random_label_percent=0.0,
                                     data_augmentation=False,
                                     **kwargs):

  if split == "test":
    split = "validation"

  return get_tf_image_dataset(quantize_bits=quantize_bits,
                        batch_size=batch_size,
                        dataset_name="my_downsampled_imagenet/64x64",
                        n_batches=n_batches,
                        split=split,
                        crop=False,
                        classification=classification,
                        label_keep_percent=label_keep_percent,
                        random_label_percent=random_label_percent,
                        data_augmentation=data_augmentation,
                        **kwargs)

def get_downsampled_imagenet32_dataset(quantize_bits=8,
                                       batch_size=64,
                                       n_batches=1000,
                                       split="train",
                                       classification=False,
                                       label_keep_percent=1.0,
                                       random_label_percent=0.0,
                                       data_augmentation=False,
                                       **kwargs):

  if split == "test":
    split = "validation"

  return get_tf_image_dataset(quantize_bits=quantize_bits,
                        batch_size=batch_size,
                        dataset_name="my_downsampled_imagenet/32x32",
                        n_batches=n_batches,
                        split=split,
                        crop=False,
                        classification=classification,
                        label_keep_percent=label_keep_percent,
                        random_label_percent=random_label_percent,
                        data_augmentation=data_augmentation,
                        **kwargs)
