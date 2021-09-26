
def get_uci_dataset(dataset_name, split, batch_size, n_batches, **kwargs):

  # https://github.com/tensorflow/datasets/issues/1441#issuecomment-581660890
  import resource
  low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

  import tensorflow as tf
  tf.config.set_visible_devices([], "GPU")
  import tensorflow_datasets as tfds

  ds = tfds.load(dataset_name, split=split)

  if "train" in split:
    ds = ds.repeat().shuffle(15000)

  ds = ds.batch(batch_size)
  if(n_batches is not None):
    ds = ds.batch(n_batches, drop_remainder=True)

  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  ds = ds.as_numpy_iterator()
  return ds

################################################################################################################

def get_BSDS300_dataset(batch_size=64,
                        n_batches=1000,
                        split="train",
                        **kwargs):

  return get_uci_dataset(batch_size=batch_size,
                              dataset_name="uci/BSDS300",
                              n_batches=n_batches,
                              split=split,
                              **kwargs)

def get_Gas_dataset(batch_size=64,
                    n_batches=1000,
                    split="train",
                    **kwargs):
  # 128 dims

  return get_uci_dataset(batch_size=batch_size,
                              dataset_name="uci/Gas",
                              n_batches=n_batches,
                              split=split,
                              **kwargs)

def get_MiniBooNE_dataset(batch_size=64,
                          n_batches=1000,
                          split="train",
                          **kwargs):
  # 50 dims

  return get_uci_dataset(batch_size=batch_size,
                              dataset_name="uci/MiniBooNE",
                              n_batches=n_batches,
                              split=split,
                              **kwargs)

def get_Power_dataset(batch_size=64,
                      n_batches=1000,
                      split="train",
                      **kwargs):
  # 9 dims

  return get_uci_dataset(batch_size=batch_size,
                              dataset_name="uci/Power",
                              n_batches=n_batches,
                              split=split,
                              **kwargs)

def get_HEPMASS_dataset(batch_size=64,
                        n_batches=1000,
                        split="train",
                        **kwargs):
  # 28 dims

  return get_uci_dataset(batch_size=batch_size,
                              dataset_name="uci/HEPMASS",
                              n_batches=n_batches,
                              split=split,
                              **kwargs)
