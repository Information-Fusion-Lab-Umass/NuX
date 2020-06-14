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
from experiment_evaluation import *

n_gpus = xla_bridge.device_count()
print('n_gpus:', n_gpus)

# Parse the user inputs
parser = argparse.ArgumentParser(description='Processing Noisy Injective Flows Checkpoint')
parser.add_argument('--name',            action='store', type=str, help='Name of model', default='CelebADefault')
parser.add_argument('--batchsize',       action='store', type=int, help='Batch Size', default=32)
parser.add_argument('--dataset',         action='store', type=str, help='Dataset to load', default='CelebA')
parser.add_argument('--quantize',        action='store', type=int, help='Sets the value of quantize_level_bits', default=5)
parser.add_argument('--start_it',        action='store', type=int, help='Experiment iteration to start with', default=0)
parser.add_argument('--num_checkpoints', action='store', type=int, help='Number of checkpoints to parse', default=-1)
parser.add_argument('--model',           action='store', type=str, help='Sets the model to use', default='CelebADefault')
args = parser.parse_args()

batch_size = args.batchsize
dataset = args.dataset
quantize_level_bits = args.quantize
start_it = args.start_it
num_checkpoints = args.num_checkpoints
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
    data_loader, x_shape = cifar10_data_loader(quantize_level_bits=quantize_level_bits, data_folder='data/cifar10/', onehot=False)
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
from CelebAImportanceSample import CelebAImportanceSample

from CelebAImportanceSample128 import CelebAIS128
from CelebAImportanceSample256 import CelebAIS256

if(model_type == 'CelebA512'):
    assert dataset == 'CelebA', 'Dataset mismatch'
    nf, nif = CelebA512(False, quantize_level_bits), CelebA512(True, quantize_level_bits)
elif(model_type == 'CelebA256'):
    assert dataset == 'CelebA', 'Dataset mismatch'
    nf, nif = CelebA256(False, quantize_level_bits), CelebA256(True, quantize_level_bits)
elif(model_type == 'CelebA128'):
    assert dataset == 'CelebA', 'Dataset mismatch'
    nf, nif = CelebA128(False, quantize_level_bits), CelebA128(True, quantize_level_bits)
elif(model_type == 'CelebAIS256'):
    assert dataset == 'CelebA', 'Dataset mismatch'
    nf, nif = CelebAIS256(False, quantize_level_bits), CelebAIS256(True, quantize_level_bits)
elif(model_type == 'CelebAIS128'):
    assert dataset == 'CelebA', 'Dataset mismatch'
    nf, nif = CelebAIS128(False, quantize_level_bits), CelebAIS128(True, quantize_level_bits)
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
elif(model_type == 'CelebAImportanceSample'):
    nf, nif = CelebAImportanceSample(False, quantize_level_bits), CelebAImportanceSample(True, quantize_level_bits)
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

    # models.append(Model(names, output_shape, params, state, jit(partial(forward, n_importance_samples=1)), jit(partial(inverse, n_importance_samples=1))))
    models.append(Model(names, output_shape, params, state, forward, inverse))
    # models.append(Model(names, output_shape, params, state, jit(forward), jit(inverse)))
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
pbar = tqdm(checkpoint_folders)
for checkpoint_path in pbar:

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

    # Compute the aggregate posteriors
    # pbar.set_description('Aggregate Posterior')
    # compute_aggregate_posteriors(key, data_loader, nf_model, nif_model, quantize_level_bits, n_samples=10000, n_samples_per_batch=32, results_folder=checkpoint_path, name='aggregate_posterior.txt')


    # # See how many importance samples are needed to accurately compute the log likelihood
    # pbar.set_description('Log Likelihood')
    # log_likelihood_estimation(key, data_loader, nif_model, results_folder=checkpoint_path, name='log_likelihood.pdf')

    # See what the standard deviation of the embeddings is
    # pbar.set_description('Embedding Standard Deviation')
    # posterior_variance(key, data_loader, nif_model, results_folder=checkpoint_path, name='embeddings_std.pdf')

    # Save some samples
    pbar.set_description('Samples')
    # save_final_samples(key, nif_model, quantize_level_bits, sigma=0.0, temp=1.0, n_samples=64, n_samples_per_batch=64, results_folder=checkpoint_path, name='nif_samples.pdf')
    # save_final_samples(key, nf_model, quantize_level_bits, sigma=0.0, temp=1.0, n_samples=64, n_samples_per_batch=64, results_folder=checkpoint_path, name='nf_samples.pdf')

    # Save higher temperature samples
    key_iter = iter(random.split(key, 51))
    pbar.set_description('High Temp Samples')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_1.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_2.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_3.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_4.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_5.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_6.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_7.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_8.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_9.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_10.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_11.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_12.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_13.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_14.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_15.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_16.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_17.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_18.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_19.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_20.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_21.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_22.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_23.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_24.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_25.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_26.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_27.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_28.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_29.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_30.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_31.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_32.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_33.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_34.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_35.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_36.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_37.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_38.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_39.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_40.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_41.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_42.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_43.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_44.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_45.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_46.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_47.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_48.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=20, n_cols=10, n_samples_per_batch=20, results_folder=checkpoint_path, name='nif_samples_temp1p0_49.pdf')

    save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=1.0, n_samples=100, n_cols=10, n_samples_per_batch=100, results_folder=checkpoint_path, name='nif_samples_temp1p0_aux.pdf')
    save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=100, n_cols=10, n_samples_per_batch=100, results_folder=checkpoint_path, name='nif_samples_temp2p0_aux.pdf')
    save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=4.0, n_samples=100, n_cols=10, n_samples_per_batch=100, results_folder=checkpoint_path, name='nif_samples_temp4p0_aux.pdf')
    save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=8.0, n_samples=100, n_cols=10, n_samples_per_batch=100, results_folder=checkpoint_path, name='nif_samples_temp8p0_aux.pdf')

    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=1.0, n_samples=24, n_samples_per_batch=24, results_folder=checkpoint_path, name='nif_samples_temp1p0_11.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=1.0, n_samples=24, n_samples_per_batch=24, results_folder=checkpoint_path, name='nif_samples_temp1p0_21.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=1.5, n_samples=24, n_samples_per_batch=24, results_folder=checkpoint_path, name='nif_samples_temp1p5_11.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=1.5, n_samples=24, n_samples_per_batch=24, results_folder=checkpoint_path, name='nif_samples_temp1p5_21.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=24, n_samples_per_batch=24, results_folder=checkpoint_path, name='nif_samples_temp2p0_11.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=24, n_samples_per_batch=24, results_folder=checkpoint_path, name='nif_samples_temp2p0_21.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.5, n_samples=24, n_samples_per_batch=24, results_folder=checkpoint_path, name='nif_samples_temp2p5_11.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.5, n_samples=24, n_samples_per_batch=24, results_folder=checkpoint_path, name='nif_samples_temp2p5_21.pdf')


    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=1.5, n_samples=64, n_samples_per_batch=64, results_folder=checkpoint_path, name='nif_samples_temp1p5_1.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=1.5, n_samples=64, n_samples_per_batch=64, results_folder=checkpoint_path, name='nif_samples_temp1p5_2.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=64, n_samples_per_batch=64, results_folder=checkpoint_path, name='nif_samples_temp2p0_1.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.0, n_samples=64, n_samples_per_batch=64, results_folder=checkpoint_path, name='nif_samples_temp2p0_2.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.5, n_samples=64, n_samples_per_batch=64, results_folder=checkpoint_path, name='nif_samples_temp2p5_1.pdf')
    # save_final_samples(next(key_iter), nif_model, quantize_level_bits, sigma=0.0, temp=2.5, n_samples=64, n_samples_per_batch=64, results_folder=checkpoint_path, name='nif_samples_temp2p5_2.pdf')

    # # Save higher temperature samples
    # pbar.set_description('High Temp Dim Change Samples')
    # save_final_samples(key, nif_model, quantize_level_bits, sigma=0.0, temp=1.0, n_samples=64, n_samples_per_batch=64, results_folder=checkpoint_path, name='nif_samples_sigma0p0.pdf')
    # save_final_samples(key, nif_model, quantize_level_bits, sigma=0.25, temp=1.0, n_samples=64, n_samples_per_batch=64, results_folder=checkpoint_path, name='nif_samples_sigma0p25.pdf')
    # save_final_samples(key, nif_model, quantize_level_bits, sigma=0.50, temp=1.0, n_samples=64, n_samples_per_batch=64, results_folder=checkpoint_path, name='nif_samples_sigma0p50.pdf')
    # save_final_samples(key, nif_model, quantize_level_bits, sigma=0.75, temp=1.0, n_samples=64, n_samples_per_batch=64, results_folder=checkpoint_path, name='nif_samples_sigma0p75.pdf')
    # save_final_samples(key, nif_model, quantize_level_bits, sigma=1.0, temp=1.0, n_samples=64, n_samples_per_batch=64, results_folder=checkpoint_path, name='nif_samples_sigma1p0.pdf')

    # # for t in [0.0, 1.0, 1.5, 2.0, 2.5]:
    # #     for s in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
    # #         t_str = str(t).replace('.', 'p')
    # #         s_str = str(s).replace('.', 'p')
    # #         name = 'nif_sample_temp%s_sigma%s.pdf'%(t_str, s_str)
    # #         save_final_samples(key, nif_model, quantize_level_bits, sigma=s, temp=t, n_samples=64, n_samples_per_batch=64, results_folder=checkpoint_path, name=name)

    # # Save some reconstructions
    # pbar.set_description('Reconstructions')
    # reconstructions_key = random.PRNGKey(1)
    # save_reconstructions(reconstructions_key, data_loader, nif_model, quantize_level_bits, n_samples=10, n_samples_per_batch=10, results_folder=checkpoint_path, name='nif_reconstructions.pdf')

    # # Save high temperature comparisons.  4th one looks the best!
    # pbar.set_description('Temperature Comparison')
    # key_iter = iter(random.split(key, 10))
    # for i in range(10):
    #     save_temperature_comparisons(next(key_iter), nf_model, nif_model, quantize_level_bits, n_samples=10, n_samples_per_batch=10, results_folder=checkpoint_path, name='temperature_comparisons%d.pdf'%i)

    # Interpolate images
    # pbar.set_description('Interpolations')
    # interpolate_pairs(key, data_loader, nif_model, quantize_level_bits, n_pairs=30, n_points=15, results_folder=checkpoint_path, name='interpolation.pdf')

    # # Create the increasing temperature plot
    # pbar.set_description('Increaging Temp')
    # save_increasing_temp(key, nif_model, quantize_level_bits, results_folder=checkpoint_path, name='temp_change.pdf')


    # # Vary t
    # data_key = random.PRNGKey(2)
    # pbar.set_description('Vary t')
    # compare_sample_over_t(data_key, key, nf_model, nif_model, quantize_level_bits, n_samples=8, n_samples_per_batch=8, results_folder=checkpoint_path, name='vary_t.pdf')

    # # Vary s
    # pbar.set_description('Vary s')
    # data_key = random.PRNGKey(2)
    # compare_sample_over_s(data_key, key, nf_model, nif_model, quantize_level_bits, n_samples=8, n_samples_per_batch=8, results_folder=checkpoint_path, name='vary_s.pdf')

    # # Compare the samples from the model
    # sample_comparison_key = random.PRNGKey(2)
    # save_sample_comparisons(sample_comparison_key, nf_model, nif_model, quantize_level_bits, results_folder=checkpoint_path, name='nif_vs_nf_samples.pdf')
    pbar.set_description('Calculate KNN Accuracy')
    #save_embeddings(k1, data_loader, nf_model, nif_model, checkpoint_path, False)
    #print_reduced_embeddings(k1, data_loader, nf_model, nif_model, checkpoint_path, True)
    #svm_reduced_embeddings(k1, data_loader, nf_model, nif_model, checkpoint_path)








