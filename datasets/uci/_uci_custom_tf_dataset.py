""" Adapted from https://github.com/didriknielsen/survae_flows/blob/master/survae/data/datasets/tabular/uci_maf.py """
""" This file should go in your tensorflow-datasets folder! See https://www.tensorflow.org/datasets/add_dataset """

import tensorflow_datasets.public_api as tfds
import glob
import tensorflow.compat.v2 as tf
import os
import numpy as np
import h5py
import pandas as pd
from collections import Counter

# TODO(uci): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(uci): BibTeX citation
_CITATION = """
"""

_DATA_OPTIONS = ["BSDS300",
                 "Gas",
                 "HEPMASS",
                 "MiniBooNE",
                 "Power"]

################################################################################################################

class BSDS300Config(tfds.core.BuilderConfig):
  def __init__(self, *, data=None, **kwargs):
    super(BSDS300Config, self).__init__(**kwargs)
    self.data = data

  def set_data_path(self, root):
    self.data_path = os.path.join(root, "BSDS300", "BSDS300.hdf5")

  def process_example(self, x):
    return x

  def generate_examples(self, split):
    handle = h5py.File(self.data_path, 'r')
    data = np.array(handle[split]).astype(np.float32)

    for i in range(data.shape[0]):
      record = {"x": data[i]}
      key = f"BSDS300_{split}_{i}"
      yield key, record

################################################################################################################

class GasConfig(tfds.core.BuilderConfig):
  def __init__(self, *, data=None, **kwargs):
    super(GasConfig, self).__init__(**kwargs)
    self.data = data

  def set_data_path(self, root):
    self.data_path = os.path.join(root, "gas", "ethylene_CO.pickle")

  def get_correlation_numbers(self, data):
    C = data.corr()
    A = C > 0.98
    B = A.sum(axis=1)
    return B

  def load_data(self):
    data = pd.read_pickle(self.data_path)
    data.drop("Meth", axis=1, inplace=True)
    data.drop("Eth", axis=1, inplace=True)
    data.drop("Time", axis=1, inplace=True)
    return data

  def load_data_and_clean(self):
    data = self.load_data()
    B = self.get_correlation_numbers(data)

    while np.any(B > 1):
      col_to_remove = np.where(B > 1)[0][0]
      col_name = data.columns[col_to_remove]
      data.drop(col_name, axis=1, inplace=True)
      B = self.get_correlation_numbers(data)
    data = (data - data.mean()) / data.std()

    return data.values

  def load_data_and_clean_and_split(self):
    data = self.load_data_and_clean().astype(np.float32)
    N_test = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
    data_train = data[0:-N_test]
    N_validate = int(0.1 * data_train.shape[0])
    data_validate = data_train[-N_validate:]
    data_train = data_train[0:-N_validate]
    return data_train, data_validate, data_test

  def generate_examples(self, split):
    data_train, data_validate, data_test = self.load_data_and_clean_and_split()
    if split == "train":
      data = data_train
    elif split == "test":
      data = data_validate
    elif split == "validation":
      data = data_test

    for i in range(data.shape[0]):
      record = {"x": data[i]}
      key = f"gas_{split}_{i}"
      yield key, record

################################################################################################################

class HEPMASSConfig(tfds.core.BuilderConfig):
  def __init__(self, *, data=None, **kwargs):
    super(HEPMASSConfig, self).__init__(**kwargs)
    self.data = data

  def set_data_path(self, root):
    self.train_data_path = os.path.join(root, "hepmass", "1000_train.csv")
    self.test_data_path = os.path.join(root, "hepmass", "1000_test.csv")

  def load_data(self):
    data_train = pd.read_csv(filepath_or_buffer=self.train_data_path, index_col=False).astype(np.float32)
    data_test = pd.read_csv(filepath_or_buffer=self.test_data_path, index_col=False).astype(np.float32)
    return data_train, data_test

  def load_data_no_discrete(self):
    """Loads the positive class examples from the first 10% of the dataset."""
    data_train, data_test = self.load_data()

    # Gets rid of any background noise examples i.e. class label 0.
    data_train = data_train[data_train[data_train.columns[0]] == 1]
    data_train = data_train.drop(data_train.columns[0], axis=1)
    data_test = data_test[data_test[data_test.columns[0]] == 1]
    data_test = data_test.drop(data_test.columns[0], axis=1)
    # Because the data_ set is messed up!
    data_test = data_test.drop(data_test.columns[-1], axis=1)

    return data_train, data_test

  def load_data_no_discrete_normalised(self):

    data_train, data_test = self.load_data_no_discrete()
    mu = data_train.mean()
    s = data_train.std()
    data_train = (data_train - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_test

  def load_data_no_discrete_normalised_as_array(self):

    data_train, data_test = self.load_data_no_discrete_normalised()
    data_train, data_test = data_train.values, data_test.values

    i = 0
    # Remove any features that have too many re-occurring real values.
    features_to_remove = []
    for feature in data_train.T:
      c = Counter(feature)
      max_count = np.array([v for k, v in sorted(c.items())])[0]
      if max_count > 5:
          features_to_remove.append(i)
      i += 1

    data_train = data_train[:, np.array(
        [i for i in range(data_train.shape[1]) if i not in features_to_remove])]
    data_test = data_test[:, np.array(
        [i for i in range(data_test.shape[1]) if i not in features_to_remove])]

    N = data_train.shape[0]
    N_validate = int(N * 0.1)
    data_validate = data_train[-N_validate:]
    data_train = data_train[0:-N_validate]

    return data_train, data_validate, data_test

  def generate_examples(self, split):
    data_train, data_validate, data_test = self.load_data_no_discrete_normalised_as_array()

    if split == "train":
      data = data_train
    elif split == "test":
      data = data_validate
    elif split == "validation":
      data = data_test

    for i in range(data.shape[0]):
      record = {"x": data[i]}
      key = f"hepmass_{split}_{i}"
      yield key, record

################################################################################################################

class MiniBooNEConfig(tfds.core.BuilderConfig):
  def __init__(self, *, data=None, **kwargs):
    super(MiniBooNEConfig, self).__init__(**kwargs)
    self.data = data

  def set_data_path(self, root):
    self.data_path = os.path.join(root, "miniboone", "data.npy")

  def load_data(self):
    data = np.load(self.data_path).astype(np.float32)
    N_test = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(0.1 * data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]

    return data_train, data_validate, data_test

  def load_data_normalised(self):
    data_train, data_validate, data_test = self.load_data()
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu) / s
    data_validate = (data_validate - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_validate, data_test

  def generate_examples(self, split):
    data_train, data_validate, data_test = self.load_data_normalised()

    if split == "train":
      data = data_train
    elif split == "test":
      data = data_validate
    elif split == "validation":
      data = data_test

    for i in range(data.shape[0]):
      record = {"x": data[i]}
      key = f"miniboone_{split}_{i}"
      yield key, record

################################################################################################################

class PowerConfig(tfds.core.BuilderConfig):
  def __init__(self, *, data=None, **kwargs):
    super(PowerConfig, self).__init__(**kwargs)
    self.data = data

  def set_data_path(self, root):
    self.data_path = os.path.join(root, "power", "data.npy")

  def load_data_split_with_noise(self):
    rng = np.random.RandomState(42)

    data = np.load(self.data_path).astype(np.float32)
    rng.shuffle(data)
    N = data.shape[0]

    data = np.delete(data, 3, axis=1)
    data = np.delete(data, 1, axis=1)
    ############################
    # Add noise
    ############################
    # global_intensity_noise = 0.1*rng.rand(N, 1)
    voltage_noise = 0.01 * rng.rand(N, 1)
    # grp_noise = 0.001*rng.rand(N, 1)
    gap_noise = 0.001 * rng.rand(N, 1)
    sm_noise = rng.rand(N, 3)
    time_noise = np.zeros((N, 1))
    # noise = np.hstack((gap_noise, grp_noise, voltage_noise, global_intensity_noise, sm_noise, time_noise))
    # noise = np.hstack((gap_noise, grp_noise, voltage_noise, sm_noise, time_noise))
    noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
    data += noise

    N_test = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(0.1 * data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]

    return data_train, data_validate, data_test

  def load_data_normalised(self):
    data_train, data_validate, data_test = self.load_data_split_with_noise()
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu) / s
    data_validate = (data_validate - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_validate, data_test

  def generate_examples(self, split):
    data_train, data_validate, data_test = self.load_data_normalised()

    if split == "train":
      data = data_train
    elif split == "test":
      data = data_validate
    elif split == "validation":
      data = data_test

    for i in range(data.shape[0]):
      record = {"x": data[i]}
      key = f"power_{split}_{i}"
      yield key, record

################################################################################################################

data_options_map = dict(BSDS300=BSDS300Config,
                        Gas=GasConfig,
                        HEPMASS=HEPMASSConfig,
                        MiniBooNE=MiniBooNEConfig,
                        Power=PowerConfig)

def create_config(config_name):
  return data_options_map[config_name](name=config_name,
                                       description=(
                                           "UCI dataset - " + config_name),
                                       version=tfds.core.Version("2.0.0"),
                                       data=config_name,
                                       release_notes={
                                           "2.0.0": "New split API (https://tensorflow.org/datasets/splits)",
                                       })

class UCI(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for uci dataset."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  """

  BUILDER_CONFIGS = [create_config(config_name) for config_name in _DATA_OPTIONS]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "x": tfds.features.Tensor(shape=(None,), dtype=tf.float32),
        }),
        supervised_keys=None,
        homepage="https://archive.ics.uci.edu/ml/datasets.php",
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract("https://zenodo.org/record/1161203/files/data.tar.gz")

    self.builder_config.set_data_path(data_path)

    return {"train": self._generate_examples(split="train"),
            "validation": self._generate_examples(split="validation"),
            "test": self._generate_examples(split="test")}

  def _generate_examples(self, split):
    import resource
    low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

    return self.builder_config.generate_examples(split)
