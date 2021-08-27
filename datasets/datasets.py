import jax.numpy as jnp
import jax
from jax import random
from functools import partial
import nux.util as util
from pathlib import Path
import os
from .basic_2d import *
from .tfds import *
from .uci.uci import *

################################################################################################################

def make_train_test(dataset_fun,
                    train_batch_size,
                    train_n_batches,
                    test_batch_size,
                    test_n_batches,
                    quantize_bits=8,
                    classification=False,
                    label_keep_percent=1.0,
                    random_label_percent=0.0,
                    data_augmentation=False,
                    train_split=None):
  if train_split is None:
    train_split = "train"
  train_ds = dataset_fun(quantize_bits=quantize_bits,
                         batch_size=train_batch_size,
                         n_batches=train_n_batches,
                         split=train_split,
                         classification=classification,
                         label_keep_percent=label_keep_percent,
                         random_label_percent=random_label_percent,
                         data_augmentation=data_augmentation)

  get_test_ds = lambda : dataset_fun(quantize_bits=quantize_bits,
                                     batch_size=test_batch_size,
                                     n_batches=test_n_batches,
                                     split="test",
                                     classification=classification,
                                     label_keep_percent=1.0,
                                     random_label_percent=0.0,
                                     data_augmentation=False)

  return train_ds, get_test_ds

def get_dataset(dataset_name,
                train_batch_size,
                train_n_batches,
                test_batch_size,
                test_n_batches,
                quantize_bits=8,
                classification=False,
                label_keep_percent=1.0,
                random_label_percent=0.0,
                data_augmentation=False,
                train_split=None):

  kwargs = dict(train_batch_size=train_batch_size,
                train_n_batches=train_n_batches,
                test_batch_size=test_batch_size,
                test_n_batches=test_n_batches,
                quantize_bits=quantize_bits,
                classification=classification,
                label_keep_percent=label_keep_percent,
                random_label_percent=random_label_percent,
                data_augmentation=data_augmentation,
                train_split=train_split)

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
  elif dataset_name == "imagenet32":
    train_ds, get_test_ds = make_train_test(get_imagenet32_dataset, **kwargs)
  elif dataset_name == "downsampled_imagenet":
    train_ds, get_test_ds = make_train_test(get_downsampled_imagenet_dataset, **kwargs)
  elif dataset_name == "downsampled_imagenet32":
    train_ds, get_test_ds = make_train_test(get_downsampled_imagenet32_dataset, **kwargs)
  elif dataset_name == "swiss_roll":
    train_ds, get_test_ds = make_train_test(get_swiss_roll_dataset, **kwargs)
  elif dataset_name == "grid":
    train_ds, get_test_ds = make_train_test(get_grid_dataset, **kwargs)
  elif dataset_name == "moons":
    train_ds, get_test_ds = make_train_test(get_moons_dataset, **kwargs)
  elif dataset_name == "circles":
    train_ds, get_test_ds = make_train_test(get_circles_dataset, **kwargs)
  elif dataset_name == "swirl_clusters":
    train_ds, get_test_ds = make_train_test(get_swirl_clusters_dataset, **kwargs)
  elif dataset_name == "2d_manifold":
    train_ds, get_test_ds = make_train_test(get_2d_manifold_dataset, **kwargs)
  elif dataset_name == "6_points":
    train_ds, get_test_ds = make_train_test(get_6_point_manifold_dataset, **kwargs)
  elif dataset_name == "uci/BSDS300":
    train_ds, get_test_ds = make_train_test(get_BSDS300_dataset, **kwargs)
  elif dataset_name == "uci/Gas":
    train_ds, get_test_ds = make_train_test(get_Gas_dataset, **kwargs)
  elif dataset_name == "uci/MiniBooNE":
    train_ds, get_test_ds = make_train_test(get_MiniBooNE_dataset, **kwargs)
  elif dataset_name == "uci/Power":
    train_ds, get_test_ds = make_train_test(get_Power_dataset, **kwargs)
  elif dataset_name == "uci/HEPMASS":
    train_ds, get_test_ds = make_train_test(get_HEPMASS_dataset, **kwargs)

  else:
    assert 0, "Invalid dataset name"

  return train_ds, get_test_ds

################################################################################################################

if __name__ == "__main__":
  import matplotlib.pyplot as plt
  from debug import *
  import pandas as pd
  import numpy as np
  import tqdm
  from hashlib import sha1
  import tensorflow_datasets as tfds
  # import tensorflow_graphics as tfg
  # from tensorflow_graphics.datasets.shapenet import Shapenet

  # # https://github.com/tensorflow/datasets/issues/1441#issuecomment-581660890
  # import resource
  # low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
  # resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

  # import tensorflow as tf
  # tf.config.set_visible_devices([], "GPU")
  # import tensorflow_datasets as tfds

  # ds = Shapenet.load(split="train",
  #                    download_and_prepare_kwargs={"download_config": tfds.download.DownloadConfig(manual_dir="~/ShapeNetCore.v2")})

  # # Get only the model ids so that we can find a specific point cloud
  # def only_model_id(x):
  #   return x["model_id"]
  # model_ids = ds.map(only_model_id)
  # model_ids = model_ids.batch(60000)
  # model_ids = model_ids.as_numpy_iterator()
  # model_ids = next(model_ids).astype(str)

  # laptop_key = "164b84d13b3d4650e096f6db40afe7bf"
  # laptop_index = list(model_ids).index(laptop_key)
  # assert laptop_index >= 0

  # # Now find the corresponding model
  # laptop = next(ds.skip(laptop_index).take(1).as_numpy_iterator())
  # v = laptop["trimesh"]["vertices"]



  # from mpl_toolkits import mplot3d
  # fig = plt.figure()
  # ax = plt.axes(projection="3d")
  # ax.scatter3D(*v.T)
  # plt.show()


  # import pdb; pdb.set_trace()

  # for example in data_set.take(1):
  #   trimesh, label, model_id = example['trimesh'], example['label'], example['model_id']

  train_ds = get_cifar10_dataset(quantize_bits=8,
                                  batch_size=100,
                                  n_batches=None,
                                  split="train",
                                  classification=False,
                                  label_keep_percent=1.0,
                                  random_label_percent=0.0,
                                  data_augmentation=True)

  inputs = next(train_ds)
  import pdb; pdb.set_trace()

  df = pd.DataFrame()
  hash_map = {}
  for i in tqdm.tqdm(jnp.arange(1000)):
    inputs = next(train_ds)
    x = inputs["x"].astype(int)
    x = np.array(x)
    x = x.reshape((-1, np.prod(x.shape[-3:])))
    hashes = np.apply_along_axis(lambda x: hash(tuple(x)), 1, x)
    value_counts = pd.Series(hashes).value_counts()
    value_counts = pd.DataFrame(value_counts)
    df = pd.concat([df, value_counts], axis=1)

  df = df.fillna(0)

  # This should not be 100 every time
  unique = df.apply(lambda x: x.value_counts())

  import pdb; pdb.set_trace()






  # test_ds = get_cifar10_dataset(quantize_bits=8,
  #                               batch_size=10,
  #                               n_batches=1000,
  #                               split="test",
  #                               classification=False,
  #                               label_keep_percent=1.0,
  #                               random_label_percent=0.0,
  #                               data_augmentation=True)
  # inputs = next(test_ds)

  # def unpickle(file):
  #   import pickle
  #   with open(file, 'rb') as fo:
  #       dict = pickle.load(fo, encoding='bytes')
  #   return dict

  # import glob
  # paths = glob.glob("/home/eddie/Downloads/cifar-10-batches-py/test_batch")

  # data = unpickle(paths[0])[b"data"]
  # data = jnp.array(data)

  # import pdb; pdb.set_trace()






  train_ds, get_test_ds = get_dataset("celeb_a",
                                      train_batch_size=8,
                                      train_n_batches=None,
                                      test_batch_size=8,
                                      test_n_batches=None,
                                      quantize_bits=8,
                                      classification=False,
                                      label_keep_percent=1.0,
                                      random_label_percent=0.0,
                                      data_augmentation=True,
                                      train_split=None)
  inputs = next(train_ds)

  x, x_aug = inputs["x"], inputs["x_aug"]
  N = x.shape[0]
  fig, (ax1, ax2) = plt.subplots(2, N)
  for i in range(N):
    ax1[i].imshow(x[i]/256)
    ax2[i].imshow(x_aug[i]/256)

  plt.show()



  import pdb; pdb.set_trace()