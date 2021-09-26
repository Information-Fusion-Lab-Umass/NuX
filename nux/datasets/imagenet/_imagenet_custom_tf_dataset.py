"""downsampled_imagenet dataset.
   https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image/downsampled_imagenet.py"""

""" The TF download links break often, so here is a workaround. """
""" This file should go in your tensorflow-datasets folder! See https://www.tensorflow.org/datasets/add_dataset """

import tensorflow_datasets.public_api as tfds
import glob
import tensorflow.compat.v2 as tf
import os
import numpy as np

_CITATION = """\
@article{DBLP:journals/corr/OordKK16,
  author    = {A{\"{a}}ron van den Oord and
               Nal Kalchbrenner and
               Koray Kavukcuoglu},
  title     = {Pixel Recurrent Neural Networks},
  journal   = {CoRR},
  volume    = {abs/1601.06759},
  year      = {2016},
  url       = {http://arxiv.org/abs/1601.06759},
  archivePrefix = {arXiv},
  eprint    = {1601.06759},
  timestamp = {Mon, 13 Aug 2018 16:46:29 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/OordKK16},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """\
Dataset with images of 2 resolutions (see config name for information on the resolution).
It is used for density estimation and generative modeling experiments.
For resized ImageNet for supervised learning ([link](https://patrykchrabaszcz.github.io/Imagenet32/)) see `imagenet_resized`.
"""

_DATA_OPTIONS = ["32x32", "64x64"]


class DownsampledImagenetConfig(tfds.core.BuilderConfig):
  """BuilderConfig for Downsampled Imagenet."""

  def __init__(self, *, data=None, **kwargs):
    """Constructs a DownsampledImagenetConfig.
    Args:
      data: `str`, one of `_DATA_OPTIONS`.
      **kwargs: keyword arguments forwarded to super.
    """
    if data not in _DATA_OPTIONS:
      raise ValueError("data must be one of %s" % _DATA_OPTIONS)

    super(DownsampledImagenetConfig, self).__init__(**kwargs)
    self.data = data
    name = kwargs["name"]
    scale = name.split("x")[0]
    self.file_names_pattern = f"Imagenet{scale}_*_npz/*.npz"

class MyDownsampledImagenet(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_downsampled_imagenet dataset."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Download downsampled_imagenet32x32 and downsampled_imagenet64x64
  from the imagenet website
  """

  BUILDER_CONFIGS = [
      DownsampledImagenetConfig(  # pylint: disable=g-complex-comprehension
          name=config_name,
          description=(
              "A dataset consisting of Train and Validation images of " +
              config_name + " resolution."),
          version=tfds.core.Version("2.0.0"),
          data=config_name,
          release_notes={
              "2.0.0": "New split API (https://tensorflow.org/datasets/splits)",
          },
      ) for config_name in _DATA_OPTIONS
  ]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(shape=(None, None, 3), dtype=tf.uint8),
        }),
        supervised_keys=None,
        homepage="http://image-net.org/",
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    data_files_pattern = os.path.join(dl_manager.manual_dir,
                                      "imagenet",
                                      self.builder_config.file_names_pattern)
    data_files = glob.glob(data_files_pattern)

    for f in data_files:
      if not tf.io.gfile.exists(f):
        msg = "You must download the dataset files manually and place them in: "
        msg += f"{dl_manager.manual_dir}/imagenet"
        msg += " as .npz folders"
        raise AssertionError(msg)

    train_files = sorted([f for f in data_files if "train" in f])
    validation_files = sorted([f for f in data_files if "val" in f])

    # train_paths = dl_manager.extract(train_files)
    # validation_paths = dl_manager.extract(validation_files)

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "paths": train_files,
            }),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                "paths": validation_files,
            }),
    ]

    # return {
    #     'train': self._generate_examples(path=train_paths),
    #     'validation': self._generate_examples(path=validation_paths),
    # }

  def _generate_examples(self, paths):
    """Generator of examples for each split."""

    import resource
    low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

    for path in paths:
      with np.load(path) as npz_items:
        for key, val in npz_items.items():
          if key != "data":
            continue

          try:
            data = val.reshape((-1, 3, scale, scale))
          except:
            scale = int(np.sqrt(val.shape[1]/3))
            data = val.reshape((-1, 3, scale, scale))
          data = data.transpose((0, 2, 3, 1))

          for i in range(data.shape[0]):
            record = {"image": data[i]}
            yield f"{path}_{i}", record
