# MNIST downloaders are taken from https://github.com/google/jax/blob/master/examples/datasets.py

"""Datasets used in examples."""
import tarfile
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

def download_url(data_folder, filename, url):
    # language=rst
    """
    Download a url to a specified location

    :param data_folder: Target folder location.  Will be created if doesn't exist
    :param filename: What to name the file
    :param url: url to download
    """
    if(path.exists(data_folder) == False):
        os.makedirs(data_folder)

    out_file = path.join(data_folder, filename)
    if(path.isfile(out_file) == False):
        urlretrieve(url, out_file)
        print('downloaded {} to {}'.format(url, data_folder))

    return out_file

def parse_mnist_struct(filename, struct_format='>II'):
    # language=rst
    """
    Unpack the data in the mnist files

    :param filename: MNIST .gz filename
    :param struct_format: How to read the files
    """
    struct_size = struct.calcsize(struct_format)
    with gzip.open(filename, 'rb') as file:
        header = struct.unpack(struct_format, file.read(struct_size))
        return header, np.array(array.array("B", file.read()), dtype=np.uint8)

def download_mnist(data_folder, base_url):
    # language=rst
    """
    Get the raw mnist data

    :param data_folder: Where to download the data to
    :param base_url: Where to download the files from
    """
    mnist_filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    for filename in mnist_filenames:
        download_url(data_folder, filename, base_url + filename)

    (_, n_train_data, n_rows, n_cols), train_images = parse_mnist_struct(path.join(data_folder, "train-images-idx3-ubyte.gz"), struct_format='>IIII')
    (_, n_test_data, n_rows, n_cols), test_images = parse_mnist_struct(path.join(data_folder, "t10k-images-idx3-ubyte.gz"), struct_format='>IIII')
    train_images = train_images.reshape((n_train_data, n_rows, n_cols))
    test_images = test_images.reshape((n_test_data, n_rows, n_cols))

    _, train_labels = parse_mnist_struct(path.join(data_folder, "train-labels-idx1-ubyte.gz"), struct_format='>II')
    _, test_labels = parse_mnist_struct(path.join(data_folder, "t10k-labels-idx1-ubyte.gz"), struct_format='>II')

    return train_images, train_labels, test_images, test_labels

def get_mnist_data(data_folder='/tmp/mnist/', kind='digits'):
    # language=rst
    """
    Retrive an mnist dataset.  Either get the digits or fashion datasets.

    :param data_folder: Where to download the data to
    :param kind: Choice of dataset to retrieve
    """
    if(kind == 'digits'):
        base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    elif(kind == 'fashion'):
        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

    # Download and get the raw dataset
    train_images, train_labels, test_images, test_labels = download_mnist(data_folder, base_url)

    # Flatten the images?
    train_images = train_images.reshape((train_images.shape[0], -1))
    test_images = test_images.reshape((test_images.shape[0], -1))

    # Add a dummy channel dimension
    train_images = train_images[...,None]
    test_images = test_images[...,None]

    # Turn the labels to one hot vectors
    train_labels = train_labels == np.arange(10)[:,None]
    test_labels = test_labels == np.arange(10)[:,None]

    train_labels = train_labels.astype(np.int32).T
    test_labels = test_labels.astype(np.int32).T

    return train_images, train_labels, test_images, test_labels

############################################################################################################################################################

def download_cifar10(data_folder, base_url):
    # language=rst
    """
    Get the raw cifar data

    :param data_folder: Where to download the data to
    :param base_url: Where to download the files from
    """
    # Download the cifar data
    filename = 'cifar-10-python.tar.gz'
    download_filename = download_url(data_folder, filename, base_url)

    # Extract the batches
    with tarfile.open(download_filename) as tar_file:
        tar_file.extractall(data_folder)

    # Remove the tar file
    os.remove(download_filename)

def load_cifar_batch(filename):
    # language=rst
    """
    Load a single batch of the cifar dataset

    :param filename: Where the batch is located
    """
    version = platform.python_version_tuple()
    py_version = version[0]
    assert py_version == '2' or py_version == '3', 'Invalid python version'
    with open(filename, 'rb') as f:
        # Load the data into a dictionary
        datadict = pickle.load(f) if py_version == '2' else pickle.load(f, encoding='latin1')
        images, labels = datadict['data'], datadict['labels']

        # Reshape the images so that the channel dim is at the end
        images = images.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1).astype(np.float32)

        # Turn the labels into onehot vectors
        labels = np.array(labels)
        return images, labels

def load_cifar10(batches_data_folder):
    # language=rst
    """
    Load a single batch of the cifar dataset

    :param filename: Where the batch is located
    """
    # Load the cifar training data batches
    xs, ys = [], []
    for batch_idx in range(1,6):
        filename = os.path.join(batches_data_folder, 'data_batch_%d'%batch_idx)
        images, labels = load_cifar_batch(filename)
        xs.append(images)
        ys.append(labels)
    train_images = np.concatenate(xs)
    train_labels = np.concatenate(ys) == np.arange(10)[:,None]

    # Load the test data
    test_images, test_labels = load_cifar_batch(os.path.join(batches_data_folder, 'test_batch'))
    test_labels = test_labels == np.arange(10)[:,None]

    train_labels = train_labels.astype(np.int32).T
    test_labels = test_labels.astype(np.int32).T
    return train_images, train_labels, test_images, test_labels

def get_cifar10_data(quantize_level_bits=2, data_folder='/tmp/cifar10/'):
    # language=rst
    """
    Load the cifar 10 dataset.

    :param data_folder: Where to download the data to
    """
    cifar10_dir = os.path.join(data_folder, 'cifar-10-batches-py')

    if(os.path.exists(cifar10_dir) == False):
        # Download the cifar dataset
        cifar_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        download_cifar10(data_folder, cifar_url)

    # Load the raw cifar-10 data
    train_images, train_labels, test_images, test_labels = load_cifar10(cifar10_dir)

    # Quantize
    factor = 256/(2**quantize_level_bits)
    train_images = train_images//factor
    test_images = test_images//factor

    return train_images, train_labels, test_images, test_labels

############################################################################################################################################################

def download_celeb(data_folder, base_url):
    # language=rst
    """
    Get the raw cifar data

    :param data_folder: Where to download the data to
    :param base_url: Where to download the files from
    """
    # Download the cifar data
    filename = 'img_align_celeba.zip'
    download_filename = download_url(data_folder, filename, base_url)

    assert 0

    # Extract the batches
    with tarfile.open(download_filename) as tar_file:
        tar_file.extractall(data_folder)

    # Remove the zip file
    os.remove(download_filename)

def get_celeb_dataset(downsize=True, quantize_level_bits=2, n_images=10000, data_folder='.'):
    # language=rst
    """
    Load the celeb A dataset.

    :param data_folder: Where to download the data to
    """
    celeb_dir = os.path.join(data_folder, 'img_align_celeba')

    if(os.path.exists(celeb_dir) == False):
        assert 0, 'Need to manually download the celeb-A dataset.  Download the zip file from here: %s'%('https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM')

    import matplotlib.pyplot as plt
    from tqdm import tqdm_notebook

    def file_iter():
        for root, dirs, files in os.walk(celeb_dir):
            for file in files:
                if(file.endswith('.jpg')):
                    path = os.path.join(root, file)
                    yield path

    all_files = []
    for path in file_iter():
        all_files.append(path)
        if(len(all_files) == n_images):
            break

    quantize_factor = 256/(2**quantize_level_bits)

    images = []
    for path in tqdm_notebook(all_files):
        im = plt.imread(path, format='jpg')
        im = im[::6,::6][7:]
        images.append(im//quantize_factor)

    np_images = np.array(images, dtype=np.int32)

    return np_images