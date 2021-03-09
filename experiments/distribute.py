import jax
import sys
from jax import vmap, jit, random
import jax.numpy as jnp
import os
from functools import partial
import time
import pathlib

def get_base_command_line_args(as_dict=False):

  import argparse

  parser = argparse.ArgumentParser(description="Launch an experiment")
  parser.add_argument("--train", default=False, action="store_true")
  parser.add_argument("--distribute", default=False, action="store_true")
  parser.add_argument("--evaluate", default=False, action="store_true")
  parser.add_argument("--retrain", default=False, action="store_true")
  parser.add_argument("--dataset", default="cifar10")
  parser.add_argument("--batch_size", default=64, type=int)
  parser.add_argument("--n_batches", default=1000, type=int)
  parser.add_argument("--test_batch_size", default=10, type=int)
  parser.add_argument("--test_n_batches", default=1000, type=int)
  parser.add_argument("--eval_interval", default=0, type=int)
  parser.add_argument("--clip", default=5, type=float)
  parser.add_argument("--lr", default=1e-3, type=float)
  parser.add_argument("--warmup", default=1000, type=float)
  parser.add_argument("--cosine_decay_steps", default=-1, type=float)
  parser.add_argument("--save_path", default="experiment_outputs")
  parser.add_argument("--init_key_seed", default=0, type=int)
  parser.add_argument("--train_key_seed", default=0, type=int)
  parser.add_argument("--eval_key_seed", default=0, type=int)
  parser.add_argument("--quantize_bits", default=8, type=int)
  parser.add_argument("--label_keep_percent", default=1.0, type=float)
  parser.add_argument("--random_label_percent", default=0.0, type=float)
  if as_dict:
    args = parser.parse_args()
    return vars(args)

  return parser

################################################################################################################

def generate_python_command(module_path, action, settings):
  args = []
  for key, val in settings.items():
    args.append(f"--{key}={val}")

  arg_str = " ".join(args)
  python_command = f"python -m {module_path} --{action} {arg_str}"
  return python_command

################################################################################################################

def distribute_experiment(python_command, gpu="2080ti-short", name=None, test=False, cluster=True):

  if cluster == False:
    os.system(python_command)
    return

  shell_script = f"#!/bin/sh\n{python_command}"

  # So that we don't have any collisions
  key = random.PRNGKey(int(10000*time.time()))
  time.sleep(0.1)
  random_number = random.randint(key, minval=0, maxval=100, shape=())

  # Save the shell script
  script_dir = f"tmp_run_scripts"
  if test == False:
    pathlib.Path(script_dir).mkdir(parents=True, exist_ok=True)
  else:
    print(f"Creating {script_dir}")

  tmp_name = f"temp_{random_number}.sh" if name is None else f"{name}.sh"
  scipt_path = os.path.join(script_dir, tmp_name)

  if test == False:
    with open(scipt_path, "w") as file:
      file.write(shell_script)
  else:
    print(f"Writing\n{shell_script}\nto {scipt_path}")

  # Distribute the task
  sbatch_command = f"sbatch -p {gpu} --gres=gpu:1 tmp_run_scripts/{tmp_name}"
  if test == False:
    os.system(sbatch_command)
  else:
    print(f"Sending {sbatch_command}")


