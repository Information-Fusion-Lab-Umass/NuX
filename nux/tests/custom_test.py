import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree
from functools import partial
import nux.util as util
import nux
import numpy as np
from nux.flows.base import Flow

def custom_test(flow, x, rng_key):
  assert isinstance(flow, Flow)

  # Initialize the flow
  flow(x, rng_key=rng_key)
  params = flow.get_params()

  # Scramble the parameters to undo the data dependent init
  flat_params, unflatten = ravel_pytree(params)
  flat_params = random.normal(rng_key, flat_params.shape)
  params = unflatten(flat_params)

  # Run the custom test
  flow.test(x, params, rng_key)
  print(f"{str(flow)} passed")

def image_tests():
  rng_key = random.PRNGKey(0)
  x = random.normal(rng_key, shape=(16, 4, 4, 4))

  # import matplotlib.pyplot as plt
  # from nux.datasets.tfds import get_mnist_dataset
  # train_ds = get_mnist_dataset(quantize_bits=8,
  #                              batch_size=4,
  #                              n_batches=None,
  #                              split="train",
  #                              classification=False,
  #                              label_keep_percent=1.0,
  #                              random_label_percent=0.0,
  #                              data_augmentation=False)
  # x = next(train_ds)["x"].astype(x.dtype)/255

  res_flow = nux.ResidualFlow(nux.LipschitzConvResBlock(filter_shape=(3, 3),
                                                        hidden_channel=16,
                                                        n_layers=2,
                                                        dropout_prob=0.0,
                                                        norm="linf"))

  convex_potential_flow = nux.ImageCPFlow(hidden_dim=2, aug_dim=1, n_hidden_layers=2)

  # flows = [convex_potential_flow]
  flows = [res_flow, convex_potential_flow]

  for flow in flows:
    custom_test(flow, x, rng_key)


def flat_test():
  rng_key = random.PRNGKey(0)
  x = random.normal(rng_key, shape=(16, 4))

  res_flow = nux.ResidualFlow(nux.LipschitzDenseResBlock(hidden_dim=16,
                                                         n_layers=2,
                                                         dropout_prob=0.0,
                                                         norm="linf"))

  convex_potential_flow = nux.CPFlow(hidden_dim=8, aug_dim=8, n_hidden_layers=2)

  # flows = [convex_potential_flow]
  flows = [res_flow, convex_potential_flow]

  for flow in flows:
    custom_test(flow, x, rng_key)


if __name__ == "__main__":
  from debug import *
  flat_test()
  image_tests()