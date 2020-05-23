import os
from tqdm import tqdm
from jax import random, vmap, jit, value_and_grad
from jax.experimental import optimizers, stax
import staxplusplus as spp
from normalizing_flows import *
import matplotlib.pyplot as plt
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

def save_final_samples(key, model, quantize_level_bits, temp=1.0, n_samples=64, n_samples_per_batch=4, results_folder='./realnvp_results', name='samples.png'):
    """
    Save a bunch of final sampels
    """
    names, output_shape, params, state, forward, inverse = model

    # Pull noise samples
    zs = random.normal(key, (n_samples,) + output_shape)*temp

    # Create the axes
    n_cols = 8
    n_rows = int(np.ceil(n_samples/n_cols))

    # Make the subplots
    fig, axes = plt.subplots(n_rows, n_cols); axes = axes.ravel()
    axes_iter = iter(axes)
    fig.set_size_inches(2*n_cols, 2*n_rows)

    # Generate the samples
    for j in range(n_samples//n_samples_per_batch):
        z = zs[j*n_samples_per_batch:(j + 1)*n_samples_per_batch]
        _, fz, _ = inverse(params, state, np.zeros(n_samples_per_batch), z, (), test=TEST)
        fz /= (2.0**quantize_level_bits) # Put the image (mostly) between 0 and 1
        fz *= (1.0 - 2*0.05)
        fz += 0.05

        for k in range(n_samples_per_batch):
            ax = next(axes_iter)
            ax.imshow(fz[k])
            ax.set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    # Save the image samples
    samples_folder = os.path.join(results_folder, 'plots')

    if(os.path.exists(samples_folder) == False):
        os.makedirs(samples_folder)

    samples_path = os.path.join(samples_folder, name)
    plt.savefig(samples_path)
    plt.close()

def save_reconstructions(key, data_loader, model, quantize_level_bits, n_samples=16, n_samples_per_batch=4, results_folder='./realnvp_results', name='reconstructions.png'):
    """
    Save reconstructions
    """

    names, output_shape, params, state, forward, inverse = model

    # Create the axes
    n_cols = n_samples
    n_rows = 2

    # Make the subplots
    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(2*n_cols, 2*n_rows)

    for j in range(n_samples//n_samples_per_batch):
        key, *keys = random.split(key, 2)
        _x = data_loader(keys[0], 1, n_samples_per_batch)[0]

        log_px, finvx, _ = forward(params, state, np.zeros(n_samples_per_batch), _x, (), test=TEST, sigma=0.0)
        _, fz, _ = inverse(params, state, np.zeros(n_samples_per_batch), finvx, (), test=TEST)
        fz /= (2.0**quantize_level_bits)
        fz *= (1.0 - 2*0.05)
        fz += 0.05

        for i in range(n_samples_per_batch):
            axes[0,j*n_samples_per_batch + i].imshow(fz[i])
            axes[1,j*n_samples_per_batch + i].imshow(_x[i]/(2.0**quantize_level_bits))
            axes[0,j*n_samples_per_batch + i].set_axis_off()
            axes[1,j*n_samples_per_batch + i].set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    # Save the image samples
    reconstruction_folder = os.path.join(results_folder, 'plots')

    if(os.path.exists(reconstruction_folder) == False):
        os.makedirs(reconstruction_folder)

    reconstruction_path = os.path.join(reconstruction_folder, name)
    plt.savefig(reconstruction_path)
    plt.close()

def save_temperature_comparisons(key, nf_model, nif_model, quantize_level_bits, n_samples=16, n_samples_per_batch=4, results_folder='./realnvp_results', name='temperature_comparisons.png'):
    """
    Save reconstructions
    """
    # Pull noise samples
    temperatures = np.linspace(0.2, 5.0, n_samples)

    # Create the axes
    n_cols = n_samples
    n_rows = 2

    # Make the subplots
    fig, axes = plt.subplots(n_rows, n_cols); axes = axes.ravel()
    axes_iter = iter(axes)
    fig.set_size_inches(2*n_cols, 2*n_rows)

    # Generate the samples from the noisy injective flow
    zs = random.normal(key, (n_samples,) + nif_model.output_shape)
    zs *= temperatures[:,None]
    temp_iter = iter(temperatures)

    for j in range(n_samples//n_samples_per_batch):
        z = zs[j*n_samples_per_batch:(j + 1)*n_samples_per_batch]
        _, fz, _ = nif_model.inverse(nif_model.params, nif_model.state, np.zeros(n_samples_per_batch), z, (), key=key, test=TEST)
        fz /= (2.0**quantize_level_bits) # Put the image (mostly) between 0 and 1
        fz *= (1.0 - 2*0.05)
        fz += 0.05

        for k in range(n_samples_per_batch):
            ax = next(axes_iter)
            ax.imshow(fz[k])
            ax.set_title('T = %5.3f'%(next(temp_iter)))
            ax.set_axis_off()

    # Generate the samples from the normalizing flow
    zs = random.normal(key, (n_samples,) + nf_model.output_shape)
    zs *= temperatures[:,None]
    temp_iter = iter(temperatures)

    for j in range(n_samples//n_samples_per_batch):
        z = zs[j*n_samples_per_batch:(j + 1)*n_samples_per_batch]
        _, fz, _ = nf_model.inverse(nf_model.params, nf_model.state, np.zeros(n_samples_per_batch), z, (), key=key, test=TEST)
        fz /= (2.0**quantize_level_bits) # Put the image (mostly) between 0 and 1
        fz *= (1.0 - 2*0.05)
        fz += 0.05

        for k in range(n_samples_per_batch):
            ax = next(axes_iter)
            ax.imshow(fz[k])
            ax.set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    # Save the image samples
    temp_comparison_folder = os.path.join(results_folder, 'plots')

    if(os.path.exists(temp_comparison_folder) == False):
        os.makedirs(temp_comparison_folder)

    temp_comparison_path = os.path.join(temp_comparison_folder, name)
    plt.savefig(temp_comparison_path, bbox_inches='tight')
    plt.close()

def compute_aggregate_posteriors(key, data_loader, nf_model, nif_model, quantize_level_bits, n_samples=10000, n_samples_per_batch=32, results_folder='./realnvp_results', name='aggregate_posterior.txt'):
    """
    Save reconstructions
    """

    # Compute the aggregate posterior for our model
    zs = []
    pbar = tqdm(np.arange(n_samples//n_samples_per_batch))
    for j in pbar:
        key, *keys = random.split(key, 2)
        _x = data_loader(keys[0], 1, n_samples_per_batch)[0]
        _, z, _ = nif_model.forward(nif_model.params, nif_model.state, np.zeros(n_samples_per_batch), _x, (), key=key)
        zs.append(z)
    zs = np.concatenate(zs)
    nif_mean, nif_std = np.mean(zs), np.std(zs)

    # Compute the aggregate posterior for NFs
    zs = []
    pbar = tqdm(np.arange(n_samples//n_samples_per_batch))
    for j in pbar:
        key, *keys = random.split(key, 2)
        _x = data_loader(keys[0], 1, n_samples_per_batch)[0]
        _, z, _ = nf_model.forward(nf_model.params, nf_model.state, np.zeros(n_samples_per_batch), _x, (), key=key)
        zs.append(z)
    zs = np.concatenate(zs)
    nf_mean, nf_std = np.mean(zs), np.std(zs)

    # Save the results to a text file
    aggregate_stats = np.array([[nif_mean, nif_std], [nf_mean, nf_std]])
    onp.savetxt(os.path.join(results_folder, name), aggregate_stats, delimiter=",")