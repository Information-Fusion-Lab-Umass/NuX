import sys
from jax import random
import os
from functools import partial
import time
import pathlib

################################################################################################################

def generate_python_command(module_path, settings):
  args = []
  for key, val in settings.items():
    args.append(f"--{key}={val}")

  arg_str = " ".join(args)
  python_command = f"python -m {module_path} {arg_str}"
  return python_command

################################################################################################################

def distribute_experiment(python_command,
                          gpu="2080ti-short",
                          n_gpus=1,
                          name=None,
                          test=False,
                          cluster=True):

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
  sbatch_command = f"sbatch -p {gpu} --gres=gpu:{n_gpus} tmp_run_scripts/{tmp_name}"
  if test == False:
    os.system(sbatch_command)
  print(f"Sending {sbatch_command}")


