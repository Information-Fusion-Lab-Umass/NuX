# MNIST downloaders are taken from https://github.com/google/jax/blob/master/examples/datasets.py

"""Datasets used in examples."""
import zipfile
import glob
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

import numpy as onp
import pandas as pd
import scipy.stats

import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from functools import partial

from jax import random, vmap, jit, value_and_grad
import jax.numpy as np

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
        print('Downloading {} to {}'.format(url, data_folder))
        urlretrieve(url, out_file)
        print('Done.')

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
        return header, onp.array(array.array("B", file.read()), dtype=onp.uint8)

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

def get_mnist_data(quantize_level_bits=2, data_folder='/tmp/mnist/', kind='digits'):
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

    # Add a dummy channel dimension
    train_images = train_images[...,None]
    test_images = test_images[...,None]

    # Turn the labels to one hot vectors
    train_labels = train_labels == onp.arange(10)[:,None]
    test_labels = test_labels == onp.arange(10)[:,None]

    train_labels = train_labels.astype(onp.int32).T
    test_labels = test_labels.astype(onp.int32).T

    # Quantize
    factor = 256/(2**quantize_level_bits)
    train_images = train_images//factor
    test_images = test_images//factor

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
        images = images.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1).astype(onp.float32)

        # Turn the labels into onehot vectors
        labels = onp.array(labels)
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
    train_images = onp.concatenate(xs)
    train_labels = onp.concatenate(ys) == onp.arange(10)[:,None]

    # Load the test data
    test_images, test_labels = load_cifar_batch(os.path.join(batches_data_folder, 'test_batch'))
    test_labels = test_labels == onp.arange(10)[:,None]

    train_labels = train_labels.astype(onp.int32).T
    test_labels = test_labels.astype(onp.int32).T
    return train_images, train_labels, test_images, test_labels

def get_cifar10_data(quantize_level_bits=2, data_folder='data/cifar10/'):
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
'''
def cifar10_data_loader(quantize_level_bits=2, data_folder='data/cifar10/'):
    # language=rst
    """
    Load the cifar 10 dataset.

    :param data_folder: Where to download the data to
    """
    train_images, train_labels, test_images, test_labels = get_cifar10_data(quantize_level_bits=quantize_level_bits, data_folder=data_folder)
    x_shape = train_images.shape[1:]

    def data_loader(key, n_gpus, batch_size, train=True, label=False, test_batch=None):
        batch_idx = random.randint(key, (n_gpus, batch_size), minval=0, maxval=train_images.shape[0])
        if(train):
            if(label):
                return train_images[batch_idx,...], train_labels[batch_idx]
            else:
                return train_images[batch_idx,...]
        else:
            if(label):
                return test_images[test_batch,...], test_labels[batch_idx]
            else:
                return test_images[test_batch,...]
'''
def cifar10_data_loader(quantize_level_bits=2, data_folder='data/cifar10/'):
    # language=rst
    """
    Load the cifar 10 dataset.
    :param data_folder: Where to download the data to
    """
    train_images, train_labels, test_images, test_labels = get_cifar10_data(quantize_level_bits=quantize_level_bits, data_folder=data_folder)

    x_shape = train_images.shape[1:]
    def data_loader(batch_shape, key=None, start=None, test=False, labels=False):
        assert (key is None)^(start is None)
        test_labels = np.nonzero(test_labels)[1]
        if(key is not None):
            batch_idx = random.randint(key, batch_shape, minval=0, maxval=train_images.shape[0])
        else:
            n_points = batch_shape[0]
            batch_idx = np.arange(n_points)
        # Broadcast the batch indices to be correct
        assert 0, 'Unimplemented'
        batch_idx = batch_idx
        data = (train_images[batch_idx], train_labels[batch_idx]) if train else (test_images[batch_idx], test_labels[batch_idx])
        data = data if labels else data[0]
        return data
    return data_loader, x_shape



############################################################################################################################################################

def get_celeb_dataset(quantize_level_bits=8, strides=(5, 5), crop=(12, 4), n_images=20000, data_folder='data/img_align_celeba/'):
    # language=rst
    """
    Load the celeb A dataset.

    :param data_folder: Where to download the data to
    """
    celeb_dir = data_folder

    if(os.path.exists(celeb_dir) == False):
        assert 0, 'Need to manually download the celeb-A dataset.  Download the zip file from here: %s'%('https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM')

    all_files = glob.glob('%s*.jpg'%celeb_dir)
    all_files = all_files[:n_images]

    quantize_factor = 256/(2**quantize_level_bits)

    images = []
    for path in tqdm_notebook(all_files):
        im = plt.imread(path, format='jpg')
        im = im[::strides[0],::strides[1]][crop[0]:,crop[1]:]
        images.append(im//quantize_factor)

    np_images = onp.array(images, dtype=onp.int32)

    return np_images

############################################################################################################################################################

def celeb_dataset_loader(quantize_level_bits=8, strides=(2, 2), crop=((26, -19), (12, -13)), n_images=200000, data_folder='data/img_align_celeba/'):
    # language=rst
    """
    Load the celeb A dataset.

    :param data_folder: Where to download the data to
    """
    celeb_dir = data_folder

    all_files = glob.glob('%s*.jpg'%celeb_dir)
    if(n_images == -1 or n_images is None):
        n_images = len(all_files)

    # Load the file paths
    all_files = all_files[:n_images]
    total_files = len(all_files)
    quantize_factor = 256/(2**quantize_level_bits)
    all_files = onp.array(all_files)

    # Load a single file so that we can get the shape
    im = plt.imread(all_files[0], format='jpg')
    im = im[::strides[0],::strides[1]][crop[0][0]:crop[0][1],crop[1][0]:crop[1][1]]
    x_shape = np.array(im).shape

    # The loader will be used to pull batches of data
    def data_loader(key, n_gpus, batch_size):
        batch_idx = random.randint(key, (n_gpus*batch_size,), minval=0, maxval=total_files)
        batch_files = all_files[onp.array(batch_idx)]

        images = onp.zeros((n_gpus, batch_size) + x_shape, dtype=onp.int32)
        for k, path in enumerate(batch_files):
            im = plt.imread(path, format='jpg')
            im = im[::strides[0],::strides[1]][crop[0][0]:crop[0][1],crop[1][0]:crop[1][1]]
            im = im//quantize_factor

            j = k%batch_size
            i = k//batch_size
            images[i, j] = im

        return images

    return data_loader, x_shape

############################################################################################################################################################

def get_BSDS300_data(quantize_level_bits=8, strides=(1, 1), crop=(0, 0), data_folder='/tmp/BSDS300/'):
    # language=rst
    """
    Load the BSDS300 dataset.

    :param data_folder: Where to download the data to
    """

    filename = 'BSDS300-images'
    full_filename = os.path.join(data_folder, filename)
    if(os.path.exists(full_filename) == False):
        bsds300_url = 'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz'
        download_filename = download_url(data_folder, filename, bsds300_url)
        assert full_filename == download_filename

        # Extract the batches
        with tarfile.open(download_filename) as tar_file:
            tar_file.extractall(data_folder)

    # Find the files from the folder
    train_images = glob.glob('/tmp/BSDS300/BSDS300/images/train/*.jpg')
    test_images = glob.glob('/tmp/BSDS300/BSDS300/images/test/*.jpg')

    quantize_factor = 256/(2**quantize_level_bits)

    # Load the files
    images = []
    shape = None
    for path in tqdm_notebook(train_images):
        im = plt.imread(path, format='jpg')
        if(shape is None):
            shape = im.shape
        if(im.shape != shape):
            im = im.transpose((1, 0, 2))
        im = im[::strides[0],::strides[1]][crop[0]:,crop[1]:]
        images.append(im//quantize_factor)

    train_images = onp.array(images, dtype=onp.int32)

    images = []
    for path in tqdm_notebook(test_images):
        im = plt.imread(path, format='jpg')
        if(im.shape != shape):
            im = im.transpose((1, 0, 2))
        im = im[::strides[0],::strides[1]][crop[0]:,crop[1]:]
        images.append(im//quantize_factor)

    test_images = onp.array(images, dtype=onp.int32)

    return train_images, test_images

############################################################################################################################################################

def make_train_test_split(x, percentage):
    n_train = int(x.shape[0]*percentage)
    onp.random.shuffle(x)
    return x[n_train:], x[:n_train]

def decorrelate_data(data, threshold=0.98):
    # language=rst
    """
    Drop highly correlated columns.
    Adapted from NSF repo https://github.com/bayesiains/nsf/blob/master/data/gas.py

    :param threshold: Threshold where columns are considered correlated
    """
    # Find the correlation between each column
    col_correlation = onp.sum(data.corr() > threshold, axis=1)

    while onp.any(col_correlation > 1):
        # Remove columns that are highly correlated with more than 1 other column
        col_to_remove = onp.where(col_correlation > 1)[0][0]
        col_name = data.columns[col_to_remove]
        data.drop(col_name, axis=1, inplace=True)

        # Find the correlation again
        col_correlation = onp.sum(data.corr() > threshold, axis=1)
    return data

def remove_outliers_in_df(df, max_z_score=2):
    z_scores = scipy.stats.zscore(df)
    return df[onp.all(onp.abs(z_scores) < max_z_score, axis=1)]

def whiten_data(x):
    U, _, VT = onp.linalg.svd(x, full_matrices=False)
    return onp.dot(U, VT)

############################################################################################################################################################

def get_gas_data(train_test_split=True, decorrelate=True, normalize=False, return_dequantize_scale=True, remove_outliers=True, whiten=True, co_only=True, data_folder='/tmp/gas/', **kwargs):
    # language=rst
    """
    Load the gas dataset.  Adapted from NSF repo https://github.com/bayesiains/nsf/tree/master/data

    :param data_folder: Where to download the data to
    """
    filename = 'data.zip'
    full_filename = os.path.join(data_folder, filename)

    # Download the dataset is we haven't already
    if(os.path.exists(full_filename) == False):
        gas_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00322/data.zip'
        download_filename = download_url(data_folder, filename, gas_url)
        assert full_filename == download_filename

        with zipfile.ZipFile(full_filename, 'r') as zip_ref:
            zip_ref.extractall(data_folder)

    # Load the datasets
    co_data = pd.read_csv(os.path.join(data_folder, 'ethylene_CO.txt'), delim_whitespace=True, header=None, skiprows=1)
    methane_data = pd.read_csv(os.path.join(data_folder, 'ethylene_methane.txt'), delim_whitespace=True, header=None, skiprows=1)

    # 0 is time, 1 and 2 are labels
    co_data.drop(columns=[0, 1, 2], inplace=True)
    methane_data.drop(columns=[0, 1, 2], inplace=True)

    # Turn everything to numeric values
    co_data = co_data.apply(partial(pd.to_numeric, errors='coerce')).dropna(axis=0)
    methane_data = methane_data.apply(partial(pd.to_numeric, errors='coerce')).dropna(axis=0)

    # Remove columns that are highly correlated
    if(decorrelate == True):
        threshold = kwargs.get('threshold', 0.98)
        co_data = decorrelate_data(co_data, threshold=threshold)
        methane_data = decorrelate_data(methane_data, threshold=threshold)

    # Normalize the data.  If we're going to use dequantization, don't do this.  Instead
    # seed an actnorm layer with the mean and std.
    if(normalize):
        co_data = (co_data - co_data.mean())/co_data.std()
        methane_data = (methane_data - methane_data.mean())/methane_data.std()

    # Column 4 is really weird
    co_data.drop(columns=[4], inplace=True)

    # The data only contains 2 decimals, so do uniform dequantization
    co_dequantization_scale = onp.ones(co_data.shape[1])*0.01*0.0
    methane_dequantization_scale = onp.ones(methane_data.shape[1])*0.01*0.0

    # Remove outliers
    if(remove_outliers):
        remove_outliers_in_df(co_data)

    # Switch from pandas to numpy
    co_data = co_data.to_numpy(dtype=onp.float32)
    methane_data = methane_data.to_numpy(dtype=onp.float32)

    # Whiten the data
    if(whiten):
        co_data = whiten_data(co_data)

    # Train test split
    if(train_test_split):
        train_percentage = kwargs.get('train_percentage', 0.7)
        co_data = make_train_test_split(co_data, train_percentage)
        methane_data = make_train_test_split(methane_data, train_percentage)

    # Only return the co data
    if(co_only == True):
        data = co_data
        dequant = co_dequantization_scale
    else:
        data = co_data, methane_data
        dequant = co_dequantization_scale, methane_dequantization_scale

    if(return_dequantize_scale):
        return data, dequant
    return data

############################################################################################################################################################

def get_miniboone_data(train_test_split=True, decorrelate=False, normalize=False, return_dequantize_scale=True, remove_outliers=True, whiten=True, data_folder='/tmp/miniboone', **kwargs):
    # language=rst
    """
    Load the miniboone dataset.  No dequantization is needed here, they use a lot of decimals.

    :param data_folder: Where to download the data to
    """
    filename = 'MiniBooNE_PID.txt'
    full_filename = os.path.join(data_folder, filename)

    # Download the dataset if we haven't already
    if(os.path.exists(full_filename) == False):
        miniboone_data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt'
        download_filename = download_url(data_folder, filename, miniboone_data_url)
        assert full_filename == download_filename

    # Load the dataset
    data = pd.read_csv(full_filename, delim_whitespace=True, header=None, skiprows=1)

    # Remove some invalid entries
    data.drop(data[data[0] < -100].index, inplace=True)

    # Turn everything to numeric values
    data = data.apply(partial(pd.to_numeric, errors='coerce')).dropna(axis=0)

    # Remove columns that are highly correlated
    if(decorrelate == True):
        threshold = kwargs.get('threshold', 0.99)
        data = decorrelate_data(data, threshold=threshold)

    # Remove outliers
    if(remove_outliers):
        remove_outliers_in_df(data)

    # Normalize the data.  If we're going to use dequantization, don't do this.  Instead
    # seed an actnorm layer with the mean and std.
    if(normalize == True):
        data = (data - data.mean())/data.std()

    # Switch from pandas to numpy
    data = data.to_numpy(dtype=onp.float32)

    # Whiten the data
    if(whiten):
        data = whiten_data(data)

    # Train test split
    if(train_test_split):
        train_percentage = kwargs.get('train_percentage', 0.7)
        data = make_train_test_split(data, train_percentage)

    # For consistency, return a dummy dequantization array
    if(return_dequantize_scale):
        n_cols = data[0].shape[1] if train_test_split else data.shape[1]
        return data, onp.zeros(n_cols)

    return data

############################################################################################################################################################

def get_power_data(train_test_split=True, decorrelate=False, normalize=False, return_dequantize_scale=True, remove_outliers=True, whiten=True, data_folder='/tmp/power/', **kwargs):
    # language=rst
    """
    Load the power dataset.

    :param data_folder: Where to download the data to
    """
    filename = 'household_power_consumption.zip'
    full_filename = os.path.join(data_folder, filename)

    # Download the dataset if we haven't already
    if(os.path.exists(full_filename) == False):
        power_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip'
        download_filename = download_url(data_folder, filename, power_url)
        assert full_filename == download_filename

        with zipfile.ZipFile(full_filename, 'r') as zip_ref:
            zip_ref.extractall(data_folder)

    # Load the dataset
    data = pd.read_csv(full_filename, sep=';')

    # Combine the time stamp into a single time
    time_ns = pd.to_datetime(data['Time']).astype(onp.int64)
    date_ns = pd.to_datetime(data['Date']).astype(onp.int64)
    data = data.drop(columns=['Time', 'Date'])
    data['Time'] = (date_ns + time_ns - time_ns[0])/1e16

    # Turn everything to numeric values
    data = data.apply(partial(pd.to_numeric, errors='coerce')).dropna(axis=0)

    # Remove columns that are highly correlated
    if(decorrelate == True):
        threshold = kwargs.get('threshold', 0.99)
        data = decorrelate_data(data, threshold=threshold)

    # Remove outliers
    if(remove_outliers):
        remove_outliers_in_df(data)

    # We have different dequantization scales
    dequantize_scale = onp.array([0.001, 0.001, 0.01, 0.1, 1.0, 1.0, 1.0, 0.0])*0.0

    # Normalize the data.  If we're going to use dequantization, don't do this.  Instead
    # seed an actnorm layer with the mean and std.
    if(normalize == True):
        data = (data - data.mean())/data.std()

    # Switch from pandas to numpy
    data = data.to_numpy(dtype=onp.float32)

    # Whiten the data
    if(whiten):
        data = whiten_data(data)

    # Train test split
    if(train_test_split):
        train_percentage = kwargs.get('train_percentage', 0.7)
        data = make_train_test_split(data, train_percentage)

    if(return_dequantize_scale):
        return data, dequantize_scale
    return data

############################################################################################################################################################

def get_hepmass_data(train_test_split=True, decorrelate=False, normalize=False, return_dequantize_scale=True, remove_outliers=True, whiten=True, retrieve_files=['1000_train', '1000_test'], data_folder='/tmp/hepmass/'):
    # language=rst
    """
    Load the HEPMASS dataset.

    :param data_folder: Where to download the data to
    """
    # There are a bunch of files in the hepmass dataset.  Only want some of them
    all_filenames = ['1000_test.csv.gz',
                     '1000_train.csv.gz',
                     'all_test.csv.gz',
                     'all_train.csv.gz',
                     'not1000_test.csv.gz',
                     'not1000_train.csv.gz']

    filenames = []
    for fname in all_filenames:
        for ret in retrieve_files:
            if(fname.startswith(ret)):
                filenames.append(fname)

    # Get each dataset
    data_dict = {}
    for filename in filenames:
        full_filename = os.path.join(data_folder, filename)

        if(os.path.exists(full_filename) == False):
            # Download the cifar dataset
            hepmass_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00347/%s'%filename
            download_filename = download_url(data_folder, filename, hepmass_url)
            assert full_filename == download_filename

        data_dict[filename.strip('.csv.gz')] = pd.read_csv(full_filename, compression='gzip')

    # Assumes we have chosen only 1 kind of dataset
    if(train_test_split):
        assert len(filenames) == 2
        train_file = [fname for fname in retrieve_files if 'train' in fname][0]
        test_file = [fname for fname in retrieve_files if 'test' in fname][0]
        train_data, test_data = data_dict[train_file], data_dict[test_file]

        # The train data is columns are messed up!
        train_data = train_data.reset_index()
        train_data.columns = test_data.columns

        # Remove the data associated with background noise
        train_data = train_data[train_data['# label'] == 1.0]
        test_data = test_data[test_data['# label'] == 1.0]

        train_data.drop(columns='# label', inplace=True)
        test_data.drop(columns='# label', inplace=True)

        # Remove outliers
        if(remove_outliers):
            remove_outliers_in_df(train_data)

        # We don't have to dequantize
        dequantize_scale = onp.zeros(train_data.shape[1])

        # Normalize the data
        if(normalize):
            train_data = (train_data - train_data.mean())/train_data.std()
            test_data = (test_data - test_data.mean())/test_data.std()

        # Switch from pandas to numpy
        train_data = train_data.to_numpy(dtype=onp.float32)
        test_data = test_data.to_numpy(dtype=onp.float32)

        # Whiten the data
        if(whiten):
            whitened_data = whiten_data(onp.concatenate([train_data, test_data], axis=0))
            train_data, test_data = onp.split(whitened_data, onp.array([train_data.shape[0]]), axis=0)

        if(return_dequantize_scale):
            return (train_data, test_data), dequantize_scale
        return train_data, test_data

    return data_dict

############################################################################################################################################################

def uci_loader(datasets=['hepmass', 'gas', 'miniboone', 'power'], data_root='data/'):
    kwargs = dict(train_test_split=True, decorrelate=False, normalize=False, return_dequantize_scale=True, remove_outliers=False, whiten=True)

    for d in datasets:
        if(d == 'hepmass'):
            data_folder = os.path.join(data_root, 'hepmass')
            (hepmass_train_data, hepmass_test_data), hepmass_noise_scale = get_hepmass_data(data_folder=data_folder, **kwargs)
            yield hepmass_train_data, hepmass_test_data, hepmass_noise_scale, 'hepmass'
            del hepmass_train_data
            del hepmass_test_data
        elif(d == 'gas'):
            data_folder = os.path.join(data_root, 'gas')
            (gas_train_data, gas_test_data), gas_noise_scale = get_gas_data(data_folder=data_folder, **kwargs)
            yield gas_train_data, gas_test_data, gas_noise_scale, 'gas'
            del gas_train_data
            del gas_test_data
        elif(d == 'miniboone'):
            data_folder = os.path.join(data_root, 'miniboone')
            (miniboone_train_data, miniboone_test_data), miniboone_noise_scale = get_miniboone_data(data_folder=data_folder, **kwargs)
            yield miniboone_train_data, miniboone_test_data, miniboone_noise_scale, 'miniboone'
            del miniboone_train_data
            del miniboone_test_data
        elif(d == 'power'):
            data_folder = os.path.join(data_root, 'power')
            (power_train_data, power_test_data), power_noise_scale = get_power_data(data_folder=data_folder, **kwargs)
            yield power_train_data, power_test_data, power_noise_scale, 'power'
            del power_train_data
            del power_test_data

############################################################################################################################################################

def STL10_dataset_loader(quantize_level_bits=8, strides=(1, 1), crop=(0, 0), data_folder='data/STL10/'):
    # language=rst
    """
    Load the STL10 dataset.

    :param data_folder: Where to download the data to
    """
    filename = 'STL10-images'
    full_filename = os.path.join(data_folder, filename)
    if(os.path.exists(full_filename) == False):
        bsds300_url =  'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
        download_filename = download_url(data_folder, filename, bsds300_url)
        assert full_filename == download_filename

        # Extract the batches
        with tarfile.open(download_filename) as tar_file:
            tar_file.extractall(data_folder)

    quantize_factor = 256/(2**quantize_level_bits)

    data_file = os.path.join(data_folder, 'stl10_binary/unlabeled_X.bin')
    data = onp.fromfile(data_file, dtype=np.uint8)
    data = data.reshape((-1, 3, 96, 96)).transpose((0, 3, 2, 1))
    data = data[:,::strides[0],::strides[1]][crop[0]:,crop[1]:]

    x_shape = data.shape[1:]

    def data_loader(key, n_gpus, batch_size):
        batch_idx = random.randint(key, (n_gpus, batch_size), minval=0, maxval=data.shape[0])
        return data[batch_idx,...]//quantize_factor

    return data_loader, x_shape
