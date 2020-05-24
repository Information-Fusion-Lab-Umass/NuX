import os
from tqdm import tqdm
from jax import random, vmap, jit, value_and_grad
from jax.experimental import optimizers, stax
import staxplusplus as spp
from normalizing_flows import *
import matplotlib.pyplot as plt
from datasets import *
from jax.flatten_util import ravel_pytree
ravel_pytree = jit(ravel_pytree)
from jax.tree_util import tree_map, tree_leaves, tree_structure, tree_unflatten
from collections import namedtuple
from datasets import get_celeb_dataset
import argparse
from jax.lib import xla_bridge
import pickle
import jax.nn.initializers as jaxinit
import jax.numpy as np
import glob
clip_grads = jit(optimizers.clip_grads)
from experiment_evaluation import save_final_samples, \
                                  save_reconstructions, \
                                  save_temperature_comparisons, \
                                  compute_aggregate_posteriors, \
                                  interpolate_pairs, \
                                  save_fid_scores, \
                                  compute_fid_score

n_gpus = xla_bridge.device_count()
print('n_gpus:', n_gpus)

# Parse the user inputs
parser = argparse.ArgumentParser(description='Processing Noisy Injective Flows Checkpoint')
parser.add_argument('--name',            action='store', type=str, help='Name of model', default='CelebADefault')
parser.add_argument('--dataset',         action='store', type=str, help='Dataset to load', default='CelebA')
parser.add_argument('--quantize',        action='store', type=int, help='Sets the value of quantize_level_bits', default=5)
parser.add_argument('--start_it',        action='store', type=int, help='Experiment iteration to start with', default=0)
parser.add_argument('--model',           action='store', type=str, help='Sets the model to use', default='CelebADefault')
args = parser.parse_args()

dataset = args.dataset
quantize_level_bits = args.quantize
start_it = args.start_it
num_checkpoints = 1 # Only going to do this on one folder at a time
experiment_name = args.name
model_type = args.model

# Retrieve the file names of the checkpoints we will be using
experiment_folder = os.path.join('Experiments', experiment_name)

# Get all of the checkpoint folders
all_checkpoints = []
for root, dirs, _ in os.walk(experiment_folder):
    for d in dirs:
        try:
            all_checkpoints.append(int(d))
        except:
            pass
all_checkpoints = sorted(all_checkpoints)

# Use the last one
if(start_it == -1):
    start_it = all_checkpoints[-1]

# Keep specific folders
checkpoint_folders = []
for folder in all_checkpoints:
    if(folder >= start_it):
        full_path = os.path.join(experiment_folder, str(folder))
        checkpoint_folders.append(full_path)
        if(len(checkpoint_folders) == num_checkpoints):
            break

assert len(checkpoint_folders) > 0, 'Did not retrieve any checkpoint folders'
print('Done Loading Folder Names')

# Load the datasets
if(dataset == 'CelebA'):
    data_loader, x_shape = celeb_dataset_loader(quantize_level_bits=quantize_level_bits, strides=(2, 2), crop=((26, -19), (12, -13)), data_folder='data/img_align_celeba/')
    assert x_shape == (64, 64, 3)
elif(dataset == 'CIFAR'):
    data_loader, x_shape = cifar10_data_loader(quantize_level_bits=quantize_level_bits, data_folder='data/cifar10/')
elif(dataset == 'STL10'):
    data_loader, x_shape = STL10_dataset_loader(quantize_level_bits=quantize_level_bits, data_folder='data/STL10/')
else:
    assert 0, 'Invalid dataset type.'

print('Done Retrieving Data')

# Load the models
from CelebA_512 import CelebA512
from CelebA_256 import CelebA256
from CelebA_128 import CelebA128

from CIFAR10_512 import CIFAR512
from CIFAR10_256 import CIFAR256

from STL10_default_model import STL10Default
from upsample_vs_multiscale import CelebAUpscale

if(model_type == 'CelebA512'):
    assert dataset == 'CelebA', 'Dataset mismatch'
    nf, nif = CelebA512(False, quantize_level_bits), CelebA512(True, quantize_level_bits)
elif(model_type == 'CelebA256'):
    assert dataset == 'CelebA', 'Dataset mismatch'
    nf, nif = CelebA256(False, quantize_level_bits), CelebA256(True, quantize_level_bits)
elif(model_type == 'CelebA128'):
    assert dataset == 'CelebA', 'Dataset mismatch'
    nf, nif = CelebA128(False, quantize_level_bits), CelebA128(True, quantize_level_bits)
elif(model_type == 'CIFAR512'):
    assert dataset == 'CIFAR', 'Dataset mismatch'
    nf, nif = CIFAR512(False, quantize_level_bits), CIFAR512(True, quantize_level_bits)
elif(model_type == 'CIFAR256'):
    assert dataset == 'CIFAR', 'Dataset mismatch'
    nf, nif = CIFAR256(False, quantize_level_bits), CIFAR256(True, quantize_level_bits)
elif(model_type == 'STL10Default'):
    assert dataset == 'STL10', 'Dataset mismatch'
    nf, nif = STL10Default(False, quantize_level_bits), STL10Default(True, quantize_level_bits)
elif(model_type == 'CelebAUpsample'):
    nf, nif = CelebAUpscale(False, quantize_level_bits), CelebAUpscale(True, quantize_level_bits)
else:
    assert 0, 'Invalid model type.'

print('Done Loading Model')

# Initialze the models
Model = namedtuple('model', 'names output_shape params state forward inverse')

models = []
for flow in [nf, nif]:
    init_fun, forward, inverse = flow
    key = random.PRNGKey(0)
    names, output_shape, params, state = init_fun(key, x_shape, ())
    z_dim = output_shape[-1]

    models.append(Model(names, output_shape, params, state, forward, inverse))
nf_model, nif_model = models

print('Done Creating the Models')

# Not actually going to use these, just need the pytrees
opt_init, _, get_params = optimizers.adam(1.0)
opt_state_nf = opt_init(nf_model.params)
opt_state_nif = opt_init(nif_model.params)

def load_pytree(treedef, dir_save):
    with open(dir_save,'rb') as f: leaves = pickle.load(f)
    return tree_unflatten(treedef, leaves)

# Run the experiements
checkpoint_path = checkpoint_folders[0]

# Load the models
opt_state_nf = load_pytree(tree_structure(opt_state_nf), os.path.join(checkpoint_path, 'opt_state_nf_leaves.p'))
opt_state_nif = load_pytree(tree_structure(opt_state_nif), os.path.join(checkpoint_path, 'opt_state_nif_leaves.p'))
state_nf = load_pytree(tree_structure(nf_model.state), os.path.join(checkpoint_path, 'state_nf_leaves.p'))
state_nif = load_pytree(tree_structure(nif_model.state), os.path.join(checkpoint_path, 'state_nf_leaves.p'))

nf_model = nf_model._replace(state=state_nf)
nif_model = nif_model._replace(state=state_nif)
nf_model = nf_model._replace(params=get_params(opt_state_nf))
nif_model = nif_model._replace(params=get_params(opt_state_nif))

key = random.PRNGKey(0)

temp_pbar = tqdm(np.linspace(0.10, 4.0, 10))
for temp in temp_pbar:

    temp_as_str = str(temp)
    temp_as_str = temp_as_str.replace('.', 'p')

    # Compute the FID score for the NF first
    compute_fid_score(nf_model,
                      key,
                      quantize_level_bits,
                      temp=temp,
                      sigma=1.0,
                      TTUR_path='TTUR/',
                      stats_path='FID/fid_stats_celeba.npz',
                      n_samples=10000,
                      n_samples_per_batch=128,
                      checkpoint_folder=checkpoint_path,
                      fid_score_folder_name='fid_scores',
                      sample_fid_folder_name='nf_samples_temp_%s_fid_folder'%temp_as_str,
                      check_for_stats=True,
                      name='nf_fid_temp_%s.txt'%temp_as_str)

    # Compute the FID score for the NIF at each sigma
    sigma_pbar = tqdm(np.array([0.00, 0.50, 1.00]))
    for sigma in sigma_pbar:

        sigma_as_str = str(sigma)
        sigma_as_str = sigma_as_str.replace('.', 'p')

        compute_fid_score(nif_model,
                          key,
                          quantize_level_bits,
                          temp=temp,
                          sigma=sigma,
                          TTUR_path='TTUR/',
                          stats_path='FID/fid_stats_celeba.npz',
                          n_samples=10000,
                          n_samples_per_batch=128,
                          checkpoint_folder=checkpoint_path,
                          fid_score_folder_name='fid_scores',
                          sample_fid_folder_name='nif_samples_temp_%s_sigma_%s_fid_folder'%(temp_as_str, sigma_as_str),
                          check_for_stats=True,
                          name='nif_fid_temp_%s_sigma_%s.txt'%(temp_as_str, sigma_as_str))
