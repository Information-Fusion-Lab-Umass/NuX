import jax
from jax import random, jit
import jax.numpy as jnp
import nux
from functools import partial
from debug import *
import optax

################################################################################################################
# Create the dataset

import tensorflow_datasets as tfds
import tensorflow as tf

def gen_dataset(quantize_bits=3):
  def quantize(x, quantize_bits, tf_cast=True):
    quantize_factor = 256/(2**quantize_bits)
    if tf_cast:
      x["image"] = x["image"]//tf.cast(quantize_factor, dtype=tf.uint8)
    else:
      x["image"] = x["image"]//quantize_factor

    x["image"] = tf.cast(x["image"], dtype=tf.float32)
    return x

  batch_size = 16

  train_ds = tfds.load("cifar10", split="train").repeat()
  train_ds = train_ds.map(partial(quantize, quantize_bits=quantize_bits))
  train_ds = train_ds.shuffle(10*batch_size, seed=0)
  train_ds = train_ds.batch(batch_size)
  train_ds = tfds.as_numpy(train_ds)

  test_ds = tfds.load("cifar10", split="test", batch_size=-1)
  test_ds = tfds.as_numpy(test_ds)
  test_ds["image"] = test_ds["image"].reshape((-1, 100) + test_ds["image"].shape[1:])
  test_ds["label"] = test_ds["label"].reshape((-1, 100) + test_ds["label"].shape[1:])
  test_ds = quantize(test_ds, quantize_bits=quantize_bits, tf_cast=False)

  return train_ds, test_ds

################################################################################################################

def create_model():
  return nux.sequential(nux.UniformDequantization(scale=2**quantize_bits),
                        nux.Logit(),
                        nux.MultiscaleGLOW(n_multiscale=2, n_blocks_per_scale=4),
                        nux.Flatten(),
                        nux.GMMPrior(n_classes=10))

################################################################################################################

@partial(jit, static_argnums=(0,))
def get_nll(flow, params, state, rng, inputs):
  outputs, state = flow.apply(params, state, rng, inputs)

  nll = -jnp.mean(outputs["log_det"] + outputs["log_pz"])
  predictions = outputs["prediction"]

  mean_accuracy = jnp.mean(predictions == inputs["y"])

  return nll, (outputs, mean_accuracy)

################################################################################################################

if __name__ == "__main__":
  rng = random.PRNGKey(0)
  quantize_bits = 3

  # Create the dataset and load the first batch to initialize the flow
  train_ds, test_ds = gen_dataset(quantize_bits)
  data_batch = next(train_ds)
  inputs = {"x": data_batch["image"], "y": data_batch["label"]}

  # Initialize the model
  flow = nux.transform_flow(create_model)
  params, state = flow.init(rng, inputs, batch_axes=(0,))

  # Train it
  valgrad = jax.value_and_grad(partial(get_nll, flow), has_aux=True)
  valgrad = jit(valgrad)

  opt_init, opt_update = optax.adam(1e-4)
  apply_updates = jit(optax.apply_updates)
  opt_update = jit(opt_update)
  opt_state = opt_init(params)

  rngs = random.split(rng, 100)
  for i, rng in enumerate(rngs):
    data_batch = next(train_ds)
    inputs = {"x": data_batch["image"], "y": data_batch["label"]}

    (nll, (outputs, mean_accuracy)), grad = valgrad(params, state, rng, inputs)
    updates, opt_state = opt_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)

    print(f"nll: {nll}, acc: {mean_accuracy}")

