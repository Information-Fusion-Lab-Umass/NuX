import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
import nux
import einops
from nux.training.trainer import Trainer
import os

class ImageTrainer(Trainer):

  def __init__(self, args, n_samples=16, quantize_bits=8, **kwargs):
    super().__init__(args, **kwargs)
    self.latent_shape = None
    self.n_samples = n_samples
    self.quantize_bits = quantize_bits
    self.sample_key = random.PRNGKey(0)

  def save_regular_samples(self, rng_key):
    k1, k2 = random.split(rng_key, 2)

    samples = self.jitted_sample(k2, self.params, self.n_samples)
    samples /= 2**self.args.quantize_bits

    size = 5
    fig, axes = plt.subplots(1, self.n_samples, figsize=(size*self.n_samples, size)); axes = axes.ravel()
    ax_iter = iter(axes)
    for i in range(self.n_samples):
      ax = next(ax_iter)
      ax.imshow(samples[i])
      ax.set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0)

    save_path = os.path.join(self.samples_path, f"sample_{self.n_train_steps}.png")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

  def save_samples(self, train_iter):
    key, self.sample_key = random.split(self.sample_key, 2)

    if self.latent_shape is None:
      data = next(train_iter)["x"][0]
      z, _ = self.apply_fun(data, self.params, self.eval_key)
      self.latent_shape = z.shape[1:]

    self.save_regular_samples(key)

################################################################################################################

def make_preprocess(args):
  return nux.Sequential([nux.UniformDequantization(),
                         nux.StaticScale(2**args.quantize_bits),
                         nux.Logit(scale=0.05, force_values=False)])

def make_flow(args):

  def make_block(checkerboard):

    flow = nux.NonlinearCouplingImage(n_layers=4,
                                      working_channel=32,
                                      hidden_channel=64,
                                      nonlinearity=util.square_swish,
                                      dropout_prob=0.2,
                                      n_resnet_layers=8,
                                      K=8,
                                      kind="spline",
                                      with_affine_coupling=True,
                                      checkerboard=checkerboard)
    return flow

  return nux.Sequential([make_preprocess(args),
                         make_block(True),
                         nux.Squeeze(),
                         make_block(False),
                         make_block(True)], prior=nux.UnitGaussianPrior())

def create_flow(args):
  args.save_path = f"cifar10_outputs/"

  args.lr                   = 1e-3
  args.warmup               = 1000
  args.cosine_decay_steps   = int(1e6)
  args.cosine_decay_amount  = 5e-2

  args.clip = 5.0

  args.train_batch_size   = 32
  args.train_n_batches    = 1000
  args.test_batch_size    = 256
  args.dd_init_batch_size = 256

  args.max_iters = 2000

  flow = make_flow(args)

  return flow, args

def loss(flow, params, inputs, is_training=True, args=None, mean_aux=True, **kwargs):
  x, rng_key = inputs["x"], inputs["rng_key"]
  aux = {}

  if params is None:
    flow(x, params=params, rng_key=rng_key, is_training=is_training, **kwargs)
    params = flow.get_params()

  z, log_px = flow(x, params=params, rng_key=rng_key, is_training=is_training, **kwargs)
  objective = -log_px

  aux["log_px"] = log_px
  aux["objective"] = objective

  if mean_aux:
    aux = jax.tree_util.tree_map(jnp.mean, aux)

  # Pass out the parameters
  if is_training == True:
    params = flow.get_params()
    aux["params"] = params

  aux = jax.lax.stop_gradient(aux)
  return objective.mean(), aux

################################################################################################################

if __name__ == "__main__":
  from debug import *
  import matplotlib.pyplot as plt
  import tqdm

  def str2bool(v):
    # https://stackoverflow.com/a/43357954
    if isinstance(v, bool):
     return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
      return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
      return False
    else:
      raise argparse.ArgumentTypeError('Boolean value expected.')

  import argparse
  parser = argparse.ArgumentParser(description="Run an experiment")

  # Control args
  parser.add_argument("--retrain", type=str2bool, default=True, help="Retrain model.")
  parser.add_argument("--best_model", type=str2bool, default=False, help="Load the best model.")
  parser.add_argument("--save_path", type=str, default="research_results/test", help="Base path where everything will be saved to.")
  parser.add_argument("--evaluate", type=str2bool, default=False, help="Evaluate all of the results")
  parser.add_argument("--save_samples", type=str2bool, default=True, help="Save samples during training?")

  # Seeds
  parser.add_argument("--init_key_seed", type=int, default=0, help="Initialization random seed")
  parser.add_argument("--train_key_seed", type=int, default=0, help="Training random seed")
  parser.add_argument("--eval_key_seed", type=int, default=0, help="Evaluation random seed")

  # Optimizer args
  parser.add_argument("--clip", type=float, default=15.0, help="Amount to clip gradients.  Set to negative number to turn off.")
  parser.add_argument("--lr", type=float, default=4e-5, help="Default learning rate.")
  parser.add_argument("--warmup", type=int, default=1000, help="Number of steps to linearly increase the learning rate to args.lr.")
  parser.add_argument("--cosine_decay_steps", type=int, default=int(1e5), help="Decay interval if we want to cosine anneal the learning rate.")
  parser.add_argument("--cosine_decay_amount", type=float, default=1e-1, help="Amount to decay learning rate by.")

  # Dataset args
  parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to learn.")
  parser.add_argument("--quantize_bits", type=int, default=8, help="Number of color bits to keep from the input images.")
  parser.add_argument("--label_keep_percent", type=float, default=1.0, help="Percent of dataset that should be labeled.")
  parser.add_argument("--random_label_percent", type=float, default=0.0, help="Percent of labels to randomly change.")
  parser.add_argument("--train_split", type=str, default="train", help="Name of trainin split.  Can use to choose subset of data.")
  parser.add_argument("--data_augmentation", type=str2bool, default=False, help="Use data augmentation consistency reg.")

  # Batch args
  parser.add_argument("--dd_init_batch_size", type=int, default=64, help="Batch size for data dependent init.")
  parser.add_argument("--train_batch_size", type=int, default=64, help="Batch size to use.")
  parser.add_argument("--train_n_batches", type=int, default=1000, help="Number training steps to run inside a lax.scan loop.  This can significantly speed up training.")
  parser.add_argument("--test_batch_size", type=int, default=64, help="Batch size of test set.")

  # Training args
  parser.add_argument("--max_iters", type=int, default=2000, help="Number of training steps")
  parser.add_argument("--eval_interval", type=int, default=1000, help="Number of training steps needed before evaluating test set.  This does not take into account batch size.")

  args = parser.parse_args()

  # Create the model
  flow, args = create_flow(args)

  trainer = ImageTrainer(args, use_pmap=False, use_vmap=False, pmap_batch=False, train_for_loop=True)
  trainer.initialize(flow, loss_fun=partial(loss, flow, args=args))

  # Create the dataset iterators.  Also stateless
  train_ds, get_test_ds = trainer.load_data(classification=args.evaluate)

  if trainer.args.retrain == True:
    # Initialize the model parameters from scratch
    def initialize_flow(rng_key, x):
      inputs = dict(x=x, rng_key=rng_key)
      objective, aux = loss(flow, None, inputs, is_training=True, args=args)
      params = flow.get_params()
      return params

    data = next(train_ds)
    x = data["x"]
    x = einops.rearrange(x, "n b ... -> (n b) ...")
    x = x[:args.dd_init_batch_size]
    params = initialize_flow(trainer.init_key, x)

    # Give the trainer the initialized parameters and initialize opt_state
    trainer.initialize_optimizer(params)

  else:
    # Load the parameters from file
    trainer.load_from_file()

  if args.evaluate == False:
    # Train
    pbar = tqdm.tqdm(jnp.arange(args.max_iters))
    for i in pbar:

      # Train for a bit
      avg_train_loss = trainer.train(train_ds)
      pbar.set_description(f"Train loss: {avg_train_loss:5.3f}")

      # Evaluate the test set
      trainer.test(get_test_ds)

      # Save everything
      trainer.checkpoint()

      # Save some visual stuff
      trainer.save_training_plot()

      if args.save_samples:
        # Save samples
        trainer.save_samples(train_ds)

      if jnp.isnan(avg_train_loss):
        assert 0

  else:
    trainer.test(get_test_ds)
    trainer.save_samples(train_ds)
