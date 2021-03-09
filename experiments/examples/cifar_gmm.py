import jax
from jax import random, jit
import jax.numpy as jnp
import nux
from functools import partial
from debug import *
import os
import tqdm
import nux.util as util
import matplotlib.pyplot as plt
import nux.networks as net
import experiments as exp
import pathlib

################################################################################################################

# To run this script, be at the same directory level as "nux" and use the command:
# python -m experiments.examples.cifar_gmm <your args>

# For example,
# to retrain the model from scratch with default args, use
# python -m experiments.examples.cifar_gmm --retrain

# to retrain from a checkpoint, use
# python -m experiments.examples.cifar_gmm --train

################################################################################################################

def create_flow_model(args):

  layers = []
  layers.append(nux.UniformDequantization())
  layers.append(nux.Scale(2**args.quantize_bits))
  layers.append(nux.Logit())
  layers.append(nux.ResidualFlowArchitecture(hidden_channel_size=args.res_flow_hidden_channel_size,
                                             actnorm=True,
                                             one_by_one_conv=False,
                                             repititions=[args.n_resflow_repeats_per_scale]*args.n_resflow_scales))
  layers.append(nux.Flatten())
  layers.append(nux.GMMPrior(n_classes=10))

  return nux.sequential(*layers)

################################################################################################################

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Launch an experiment")

  parser.add_argument("--train", default=False, action="store_true", help="Include flag to train model.")
  parser.add_argument("--evaluate", default=False, action="store_true", help="Include flag to evaluate model.")
  parser.add_argument("--retrain", default=False, action="store_true", help="Include flag with --train to ignore saved checkpoint and retrain model.")

  parser.add_argument("--quantize_bits", type=int, default=8, help="Number of color bits to keep from the input images.")
  parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to learn.")
  parser.add_argument("--percent_labeled", type=float, default=0.2, help="Percent of the data to label")
  parser.add_argument("--max_iters", type=int, default=200, help="Number of training steps")
  parser.add_argument("--batch_size", type=int, default=16, help="Batch size to use.")
  parser.add_argument("--n_batches", type=int, default=1000, help="Number training steps to run inside a lax.scan loop.  This can significantly speed up training.")
  parser.add_argument("--test_batch_size", type=int, default=16, help="Batch size of test set.")
  parser.add_argument("--test_n_batches", type=int, default=625, help="Number of test eval steps to do inside a lax.scan loop.")
  parser.add_argument("--eval_interval", type=int, default=30000, help="Number of training steps needed before evaluating test set.  This does not take into account batch size.")

  parser.add_argument("--clip", type=float, default=15.0, help="Amount to clip gradients.  Set to negative number to turn off.")
  parser.add_argument("--lr", type=float, default=1e-3, help="Default learning rate.")
  parser.add_argument("--warmup", type=int, default=2000, help="Number of steps to linearly increase the learning rate to args.lr.")
  parser.add_argument("--cosine_decay_steps", type=int, default=100000, help="Decay interval if we want to cosine anneal the learning rate.")

  parser.add_argument("--save_path", type=str, default="example_outputs/cifar_gmm_model", help="Path to save model to")

  parser.add_argument("--init_key_seed", type=int, default=0, help="Initialization random seed")
  parser.add_argument("--train_key_seed", type=int, default=0, help="Training random seed")
  parser.add_argument("--eval_key_seed", type=int, default=0, help="Evaluation random seed")
  parser.add_argument("--label_keep_percent", default=1.0, type=float)
  parser.add_argument("--random_label_percent", default=0.0, type=float)

  parser.add_argument("--res_flow_hidden_channel_size", type=int, default=256, help="Size of hidden channels")
  parser.add_argument("--n_resflow_repeats_per_scale", type=int, default=8, help="Number of residual flow layers per scale")
  parser.add_argument("--n_resflow_scales", type=int, default=3, help="Number of times we squeeze the input")

  args = parser.parse_args()

  create_model = partial(create_flow_model, args)

  if args.train or args.retrain:
    exp.train_model(create_model, args, image=True, classification=True)

  elif args.evaluate:
    exp.evaluate_image_model(create_model, args, classification=True)


