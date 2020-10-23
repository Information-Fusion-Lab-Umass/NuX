import jax
from jax import random, jit
import jax.numpy as jnp
import nux
from functools import partial
from debug import *
import optax
import os
from nux.training import GenerativeModel
import tqdm
import nux.util as util

################################################################################################################

def get_dataset(quantize_bits, batch_size, dataset_name="celeb_a", n_batches=None, split="train"):
  import tensorflow as tf
  import tensorflow_datasets as tfds

  def central_crop(x):
    x["image"] = x["image"][::2,::2][26:-19,12:-13]
    return x

  def to_float(x):
    x["image"] = tf.cast(x["image"], dtype=tf.float32)
    return x

  def random_flip(x):
    x["image"] = tf.image.random_flip_left_right(x["image"])
    return x

  def quantize(x):
    quantize_factor = 256/(2**quantize_bits)
    x["image"] = x["image"]//quantize_factor
    return x

  def to_expected_input(x):
    return {"x": x["image"]}

  if split == "train":
    train_ds = tfds.load(dataset_name, split="train").repeat()
    if(dataset_name == "celeb_a"):
      train_ds = train_ds.map(central_crop)
    # train_ds = train_ds.map(random_flip)
    train_ds = train_ds.map(to_float)
    train_ds = train_ds.map(quantize)
    train_ds = train_ds.map(to_expected_input)
    train_ds = train_ds.shuffle(10*batch_size)
    train_ds = train_ds.batch(batch_size)
    if(n_batches is not None):
      train_ds = train_ds.batch(n_batches)
    train_ds = tfds.as_numpy(train_ds)
    return train_ds
  else:
    test_ds = tfds.load(dataset_name, split="test")
    if(dataset_name == "celeb_a"):
      test_ds = test_ds.map(central_crop)
    test_ds = test_ds.map(to_float)
    test_ds = test_ds.map(quantize)
    test_ds = test_ds.map(to_expected_input)
    test_ds = test_ds.batch(batch_size)
    if(n_batches is not None):
      test_ds = test_ds.batch(n_batches)
    test_ds = tfds.as_numpy(test_ds)
    return test_ds

################################################################################################################

def create_model():

  network_kwargs = dict(n_blocks=6,
                        hidden_channel=64,
                        parameter_norm="weight_norm",
                        normalization="instance_norm",
                        nonlinearity="swish")

  return nux.sequential(nux.UniformDequantization(scale=256),
                        nux.Logit(),
                        nux.MultscaleFlowPP(n_scales=3,
                                            n_blocks=6,
                                            n_components=8,
                                            network_kwargs=network_kwargs),
                        nux.Flatten(),
                        nux.UnitGaussianPrior())

################################################################################################################

def loss_fun(apply_fun, params, state, key, inputs, **kwargs):
  outputs, updated_state = apply_fun(params, state, key, inputs, **kwargs)
  log_px = outputs["log_pz"] + outputs["log_det"]
  return -jnp.mean(log_px), (outputs, updated_state)

################################################################################################################

if __name__ == "__main__":
  key = random.PRNGKey(0)
  save_path = os.path.join("saved_models", "cifar.pickle")

  batch_size = 32
  n_batches = 500
  ds = get_dataset(quantize_bits=8,
                   batch_size=32,
                   n_batches=2000,
                   dataset_name="cifar10",
                   split="train")
  inputs = next(ds)

  # Initialize the model
  flow = nux.transform_flow(create_model)
  params, state = flow.init(key, {"x": inputs["x"][0]}, batch_axes=(0,))

  model = GenerativeModel(flow, params, state, loss_fun)
  bits_per_dim_scale = util.list_prod(inputs["x"][0][0].shape)*jnp.log(2)

  if(os.path.exists(save_path)):
    model.load_model(save_path)

  keys = random.split(key, 1500000)
  pbar = tqdm.tqdm(list(enumerate(keys)), leave=False)
  for i, key in pbar:

    # Perform a bunch of gradient steps
    inputs = next(ds)
    model.multi_grad_step(key, inputs)
    pbar.set_description(f"loss: {model.trainer.losses[-1]/bits_per_dim_scale:.2f}")

    # Evaluate the test set
    if(model.trainer.training_steps%50 == 0):
      eval_key = random.PRNGKey(0)
      test_ds = get_dataset(quantize_bits=8,
                            batch_size=32,
                            n_batches=1000,
                            dataset_name="cifar10",
                            split="test")

      test_losses = []
      total_examples = 0
      try:
        while(True):
          inputs = next(test_ds)
          n_examples_in_batch = jnp.prod(jnp.array(inputs["x"].shape[:2]))
          loss = model.multi_test_step(eval_key, inputs)
          test_losses.append(loss*n_examples_in_batch)
          total_examples += n_examples_in_batch
      except:
        pass

      test_loss = jnp.sum(jnp.array(test_losses))/total_examples
      model.tester.losses.append(float(test_loss))

    # Save the model
    model.save_model(save_path)
