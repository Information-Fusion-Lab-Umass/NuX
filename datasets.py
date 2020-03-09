# TAKEN FROM https://github.com/google/jax/blob/master/examples/datasets.py
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Datasets used in examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import array
import gzip
import os
from os import path
import struct
from six.moves.urllib.request import urlretrieve

from six.moves import cPickle as pickle
from imageio import imread
import platform

import numpy as np


_MNIST_DATA = "/tmp/mnist/"
_FASHION_MNIST_DATA = "/tmp/fashion_mnist/"


def _download(data_folder, url, filename):
  """Download a url to a file in the JAX data temp directory."""
  if not path.exists(data_folder):
    os.makedirs(data_folder)
  out_file = path.join(data_folder, filename)
  if not path.isfile(out_file):
    urlretrieve(url, out_file)
    print("downloaded {} to {}".format(url, data_folder))


def _partial_flatten(x):
  """Flatten all but the first dimension of an ndarray."""
  return np.reshape(x, (x.shape[0], -1))


def _one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)


def mnist_raw(data_folder, base_url):
  """Download and parse the raw MNIST dataset."""

  def parse_labels(filename):
    with gzip.open(filename, "rb") as fh:
      _ = struct.unpack(">II", fh.read(8))
      return np.array(array.array("B", fh.read()), dtype=np.uint8)

  def parse_images(filename):
    with gzip.open(filename, "rb") as fh:
      _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
      return np.array(array.array("B", fh.read()),
                      dtype=np.uint8).reshape(num_data, rows, cols)

  for filename in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                   "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
    _download(data_folder, base_url + filename, filename)

  train_images = parse_images(path.join(data_folder, "train-images-idx3-ubyte.gz"))
  train_labels = parse_labels(path.join(data_folder, "train-labels-idx1-ubyte.gz"))
  test_images = parse_images(path.join(data_folder, "t10k-images-idx3-ubyte.gz"))
  test_labels = parse_labels(path.join(data_folder, "t10k-labels-idx1-ubyte.gz"))

  return train_images, train_labels, test_images, test_labels


def mnist(permute_train=False):
  """Download, parse and process MNIST data to unit scale and one-hot labels."""
  # CVDF mirror of http://yann.lecun.com/exdb/mnist/
  base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
  train_images, train_labels, test_images, test_labels = mnist_raw(_MNIST_DATA, base_url)

  train_images = _partial_flatten(train_images) / np.float32(255.)
  test_images = _partial_flatten(test_images) / np.float32(255.)
  train_labels = _one_hot(train_labels, 10)
  test_labels = _one_hot(test_labels, 10)

  if permute_train:
    perm = np.random.RandomState(0).permutation(train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]

  return train_images, train_labels, test_images, test_labels


def fashion_mnist(permute_train=False):
  """Download, parse and process MNIST data to unit scale and one-hot labels."""
  base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
  train_images, train_labels, test_images, test_labels = mnist_raw(_FASHION_MNIST_DATA, base_url)

  train_images = _partial_flatten(train_images) / np.float32(255.)
  test_images = _partial_flatten(test_images) / np.float32(255.)
  train_labels = _one_hot(train_labels, 10)
  test_labels = _one_hot(test_labels, 10)

  if permute_train:
    perm = np.random.RandomState(0).permutation(train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]

  return train_images, train_labels, test_images, test_labels



def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = load_pickle(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
      mean_image = np.mean(X_train, axis=0)
      X_train -= mean_image
      X_val -= mean_image
      X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }

def get_celeb_dataset( n_images=50000 ):
    import matplotlib.pyplot as plt
    from tqdm import tqdm_notebook

    all_files = []
    for root, dirs, files in os.walk( 'img_align_celeba/' ):
        for file in files:
            if( file.endswith( '.jpg' ) ):
                path = os.path.join( root, file )
                all_files.append( path )

    images = []
    for path in tqdm_notebook( all_files[:n_images] ):
        im = plt.imread( path, format='jpg' )
        im = np.pad( im, ( ( 0, 2 ), ( 0, 2 ), ( 0, 0 ) ) )
        images.append( im )

    np_images = np.array( images, dtype=np.int32 ) / 255.0

    return np_images