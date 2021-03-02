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
# python -m experiments.examples.celeba <your args>

################################################################################################################

def create_flow_model(args):

  def image_architecture(out_shape):
    return net.ResNet(out_channel=out_shape[-1],
                      n_blocks=args.n_resnet_blocks,
                      hidden_channel=args.res_block_hidden_channel,
                      nonlinearity="relu",
                      normalization="batch_norm",
                      parameter_norm="weight_norm",
                      block_type="reverse_bottleneck",
                      zero_init=True,
                      use_bias=True,
                      dropout_rate=0.2,
                      gate=False,
                      gate_final=True,
                      squeeze_excite=False)

  layers = []
  layers.append(nux.UniformDequantization())
  layers.append(nux.Scale(2**args.quantize_bits))
  layers.append(nux.Logit())
  layers.append(nux.FlowPlusPlus(n_components=args.n_mixture_components,
                                 n_checkerboard_splits_before=args.n_checkerboard_splits_before,
                                 n_channel_splits=args.n_channel_splits,
                                 n_checkerboard_splits_after=args.n_checkerboard_splits_after,
                                 apply_transform_to_both_halves=False,
                                 create_network=image_architecture))

  layers.append(nux.Flatten())
  layers.append(nux.AffineGaussianPriorDiagCov(output_dim=args.output_dim))

  return nux.sequential(*layers)

################################################################################################################

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Launch an experiment")

  parser.add_argument("--train", default=False, action="store_true", help="Include flag to train model.")
  parser.add_argument("--evaluate", default=False, action="store_true", help="Include flag to evaluate model.")
  parser.add_argument("--retrain", default=False, action="store_true", help="Include flag with --train to ignore saved checkpoint and retrain model.")

  parser.add_argument("--quantize_bits", type=int, default=3, help="Number of color bits to keep from the input images.")
  parser.add_argument("--dataset", type=str, default="celeb_a", help="Dataset to learn.")
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
  parser.add_argument("--cosine_decay_steps", type=int, default=-1, help="Decay interval if we want to cosine anneal the learning rate.")

  parser.add_argument("--save_path", type=str, default="example_outputs/model", help="Path to save model to")

  parser.add_argument("--init_key_seed", type=int, default=0, help="Initialization random seed")
  parser.add_argument("--train_key_seed", type=int, default=0, help="Training random seed")
  parser.add_argument("--eval_key_seed", type=int, default=0, help="Evaluation random seed")
  parser.add_argument("--label_keep_percent", default=1.0, type=float)
  parser.add_argument("--random_label_percent", default=0.0, type=float)




  parser.add_argument("--n_resnet_blocks", type=int, default=3, help="Number of resnet blocks per transform")
  parser.add_argument("--res_block_hidden_channel", type=int, default=64, help="Hidden channel of residual flow block")
  parser.add_argument("--n_mixture_components", type=int, default=32, help="Number of logistic mixture cdf components")
  parser.add_argument("--n_checkerboard_splits_before", type=int, default=2, help="Number of first checkerboard splits")
  parser.add_argument("--n_channel_splits", type=int, default=2, help="Number of channel splits")
  parser.add_argument("--n_checkerboard_splits_after", type=int, default=2, help="Number of second checkerboard splits")
  parser.add_argument("--output_dim", type=int, default=128, help="Dimensionality of latent space")

  args = parser.parse_args()

  create_model = partial(create_flow_model, args)

  if args.train:
    exp.train_model(create_model, args, image=True, classification=False)

  elif args.evaluate:
    exp.evaluate_image_model(create_model, args, classification=False)


