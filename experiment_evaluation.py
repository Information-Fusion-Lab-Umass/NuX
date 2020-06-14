import os
from tqdm import tqdm
from jax import random, vmap, jit, value_and_grad
from jax.experimental import optimizers, stax
import staxplusplus as spp
from normalizing_flows import *
import matplotlib
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
import tensorflow
import subprocess
clip_grads = jit(optimizers.clip_grads)

def save_increasing_temp(key, model, quantize_level_bits, results_folder='results', name='temp_change.pdf'):
    """
    Save a bunch of final sampels
    """
    names, output_shape, params, state, forward, inverse = model

    # Create the axes
    n_cols = 8
    n_rows = 2

    n_samples = n_cols*n_rows
    n_samples_per_batch = n_cols*n_rows
    temp = np.linspace(0.0, 4.0, n_cols)
    temp = np.hstack(list(temp)*n_rows)

    key, *keys = random.split(key, 5)

    # Pull noise samples
    zs = random.normal(key, (n_samples,) + output_shape)*temp[:,None]

    # Make the subplots
    fig, axes = plt.subplots(n_rows, n_cols); axes = axes.ravel()
    axes_iter = iter(axes)
    fig.set_size_inches(2*n_cols, 2*n_rows)

    # Generate the samples
    index = 0
    for j in range(n_samples//n_samples_per_batch):
        z = zs[j*n_samples_per_batch:(j + 1)*n_samples_per_batch]
        _, fz, _ = inverse(params, state, np.zeros(n_samples_per_batch), z, (), test=TEST, key=key, sigma=0.0)
        fz /= (2.0**quantize_level_bits)
        fz *= (1.0 - 2*0.05)
        fz += 0.05

        for k in range(n_samples_per_batch):
            ax = next(axes_iter)
            if(index < n_cols):
                ax.set_title('T = %5.3f'%(temp[index]))
            ax.imshow(fz[k])
            ax.set_axis_off()
            index += 1

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    # Save the image samples
    samples_folder = os.path.join(results_folder, 'plots')

    if(os.path.exists(samples_folder) == False):
        os.makedirs(samples_folder)

    samples_path = os.path.join(samples_folder, name)
    plt.savefig(samples_path, bbox_inches='tight', format='pdf')
    plt.close()

################################################################################################################################################

def save_final_samples(key, model, quantize_level_bits, sigma=1.0, temp=1.0, n_samples=64, n_cols=8, n_samples_per_batch=4, results_folder='results', name='samples.pdf'):
    """
    Save a bunch of final sampels
    """
    names, output_shape, params, state, forward, inverse = model

    # Pull noise samples
    zs = random.normal(key, (n_samples,) + output_shape)*temp

    # Create the axes
    n_rows = int(np.ceil(n_samples/n_cols))

    # Make the subplots
    fig, axes = plt.subplots(n_rows, n_cols); axes = axes.ravel()
    axes_iter = iter(axes)
    fig.set_size_inches(2*n_cols, 2*n_rows)

    # Generate the samples
    for j in range(n_samples//n_samples_per_batch):
        z = zs[j*n_samples_per_batch:(j + 1)*n_samples_per_batch]
        _, fz, _ = inverse(params, state, np.zeros(n_samples_per_batch), z, (), test=TEST, key=key, sigma=sigma)
        fz /= (2.0**quantize_level_bits)
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
    plt.savefig(samples_path, bbox_inches='tight')
    plt.close()

################################################################################################################################################

def save_reconstructions(key, data_loader, model, quantize_level_bits, n_samples=16, n_samples_per_batch=4, results_folder='results', name='reconstructions.pdf'):
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

    inital_key = key

    x = data_loader(key, 1, n_samples)[0]
    # x = data_loader(0, 0, 0, indices=[94, 136, 153, 195, 332, 347, 365, 407])[0]

    for j in range(n_samples//n_samples_per_batch):
        key, *keys = random.split(key, 3)
        _x = x[j*n_samples_per_batch:(j + 1)*n_samples_per_batch]

        log_px, finvx, _ = forward(params, state, np.zeros(n_samples_per_batch), _x, (), key=keys[0])
        _, fz, _ = inverse(params, state, np.zeros(n_samples_per_batch), finvx, (), key=keys[1], sigma=0.0)
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
    plt.savefig(reconstruction_path, bbox_inches='tight', format='pdf')
    plt.close()

################################################################################################################################################

def save_temperature_comparisons(key, nf_model, nif_model, quantize_level_bits, n_samples=16, n_samples_per_batch=4, results_folder='results', name='temperature_comparisons.pdf'):
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
        _, fz, _ = nif_model.inverse(nif_model.params, nif_model.state, np.zeros(n_samples_per_batch), z, (), key=key, test=TEST, sigma=0.0)
        fz /= (2.0**quantize_level_bits)
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
        fz /= (2.0**quantize_level_bits)
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
    plt.savefig(temp_comparison_path, bbox_inches='tight', format='pdf')
    plt.close()

################################################################################################################################################

def compute_aggregate_posteriors(key, data_loader, nf_model, nif_model, quantize_level_bits, n_samples=10000, n_samples_per_batch=32, results_folder='results', name='aggregate_posterior.txt'):
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

    print('nif_mean', nif_mean)
    print('nif_std', nif_std)
    print('nf_mean', nf_mean)
    print('nf_std', nf_std)

    # Save the results to a text file
    aggregate_stats = np.array([[nif_mean, nif_std], [nf_mean, nf_std]])
    onp.savetxt(os.path.join(results_folder, name), aggregate_stats, delimiter=",")

################################################################################################################################################

import jax.ops

@jit
def cartesian_to_spherical(x):
    r = np.sqrt(np.sum(x**2))
    denominators = np.sqrt(np.cumsum(x[::-1]**2)[::-1])[:-1]
    phi = np.arccos(x[:-1]/denominators)

    last_value = np.where(x[-1] >= 0, phi[-1], 2*np.pi - phi[-1])
    phi = jax.ops.index_update(phi, -1, last_value)

    return np.hstack([r, phi])

@jit
def spherical_to_cartesian(phi_x):
    r = phi_x[0]
    phi = phi_x[1:]
    return r*np.hstack([1.0, np.cumprod(np.sin(phi))])*np.hstack([np.cos(phi), 1.0])

def interpolate_pairs(key, data_loader, nif_model, quantize_level_bits, n_pairs=5, n_points=10, results_folder='results', name='interpolation.pdf'):
    """
    Interpolate images
    """
    k1, k2 = random.split(key, 2)
    x_for_interpolation = data_loader(key, 1, 2*n_pairs)[0]
    random_pairs = random.randint(key, (2*n_pairs,), minval=0, maxval=x_for_interpolation.shape[0])
    pairs_iter = iter(random_pairs)
    index_pairs = [(next(pairs_iter), next(pairs_iter)) for _ in range(n_pairs)]

    n_cols = n_points
    n_rows = len(index_pairs)

    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(2*n_cols, 2*n_rows)

    for i, (idx1, idx2) in enumerate(index_pairs):
        x = x_for_interpolation[[idx1, idx2]]
        # _, finvx, _ = nif_model.forward(nif_model.params, nif_model.state, np.zeros(2), x, (), test=TEST, key=key)
        key, _ = random.split(key, 2)
        finvx = random.normal(key, (2,) + nif_model.output_shape)

        # Interpolate
        phi = jit(vmap(cartesian_to_spherical))(finvx)
        phi1, phi2 = phi
        interpolation_phi = np.linspace(phi1, phi2, n_points)
        interpolation_z = jit(vmap(spherical_to_cartesian))(interpolation_phi)

        _, fz, _ = nif_model.inverse(nif_model.params, nif_model.state, np.zeros(n_points), interpolation_z, (), test=TEST)
        fz /= (2.0**quantize_level_bits)
        fz *= (1.0 - 2*0.05)
        fz += 0.05
        for j in range(n_points):
            axes[i,j].imshow(fz[j])
            axes[i,j].set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    temp_comparison_path = os.path.join(results_folder, 'plots', name)
    plt.savefig(temp_comparison_path, bbox_inches='tight', format='pdf')
    plt.close()

################################################################################################################################################

def log_likelihood_estimation(key, data_loader, nif_model, results_folder='results', name='log_likelihood.pdf'):
    """
    See how effectively we can estimate log likelihood
    """
    n_samples = 128
    importance_samples = np.array([1, 2, 4, 8, 16, 32, 64])
    # importance_samples = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    n_samples_per_batch = 128

    @jit
    def estimate_log_likelihood(x, key):
        log_px, _, _ = nif_model.forward(nif_model.params, nif_model.state, np.zeros(x.shape[0]), x, (), key=key, n_importance_samples=1)
        return log_px

    data_key = random.PRNGKey(0)
    importance_sample_key = random.PRNGKey(1)
    iw_estimates = []

    # Pull new data
    x = data_loader(data_key, 1, n_samples)[0]

    # Compute all of the estimates
    for n_importance_samples in tqdm(importance_samples):
        keys = random.split(importance_sample_key, n_importance_samples)
        log_likelihoods = []

        # Evaluate each importance sample
        for key in tqdm(keys):

            log_likelihoods_per_key = []

            # Use batches of data
            for j in tqdm(np.arange(n_samples//n_samples_per_batch)):
                key, *keys = random.split(key, 2)

                _x = x[j*n_samples_per_batch:(j + 1)*n_samples_per_batch]
                log_px = estimate_log_likelihood(_x, keys[0]) # (n_z,)
                log_likelihoods_per_key.append(log_px)

            log_likelihoods_per_key = np.concatenate(log_likelihoods_per_key) # (n_z,)
            log_likelihoods.append(log_likelihoods_per_key) # (n_keys, n_z)

        log_likelihoods = np.array(log_likelihoods) # (n_keys, n_z)
        iw_estimate = logsumexp(log_likelihoods, axis=0) - np.log(log_likelihoods.shape[0]) # (n_z,)
        iw_estimate_mean = iw_estimate.mean() # (1,)
        iw_estimates.append(iw_estimate_mean)

    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(importance_samples, iw_estimates)
    ax1.set_xscale('log')

    iw_comparison_path = os.path.join(results_folder, 'plots', name)
    plt.savefig(iw_comparison_path, bbox_inches='tight', format='pdf')
    plt.close()

def posterior_variance(key, data_loader, nif_model, results_folder='results', name='embeddings_std.pdf'):
    """
    Check how consistent embeddings will be
    """
    n_samples = 256
    n_samples_per_batch = 256

    @jit
    def sample_from_posterior(x, key):
        _, z, _ = nif_model.forward(nif_model.params, nif_model.state, np.zeros(x.shape[0]), x, (), key=key, n_importance_samples=1)
        return z

    data_key = random.PRNGKey(0)
    importance_sample_key = random.PRNGKey(1)
    point_stds = []

    # Pull new data
    x = data_loader(data_key, 1, n_samples)[0]
    data_key, _ = random.split(data_key)

    # For each data point, we want to see the distribution of embeddings
    for j in tqdm(np.arange(n_samples//n_samples_per_batch)):
        key, *keys = random.split(key, 2)
        _x = x[j*n_samples_per_batch:(j + 1)*n_samples_per_batch]

        zs = []
        for key in random.split(key, 16):
            z = sample_from_posterior(_x, key) # (n_z, dim_z)
            zs.append(z)

        zs = np.array(zs) # (keys, n_z, dim_z)
        zs_std = np.std(zs, axis=0) # (n_z, dim_z)
        point_stds.append(zs_std)

    point_stds = np.concatenate(point_stds) # (n_z, dim_z)
    point_std_means = np.mean(point_stds, axis=0) # (dim_z,)
    point_std = np.mean(point_std_means) # (1,)

    print('point_std', point_std)
    assert 0

    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(importance_samples, point_stds)
    ax1.set_xscale('log')

    iw_comparison_path = os.path.join(results_folder, 'plots', name)
    plt.savefig(iw_comparison_path, bbox_inches='tight', format='pdf')
    plt.close()


################################################################################################################################################

def compare_sample_over_t(data_key, key, nf_model, nif_model, quantize_level_bits, n_samples=16, n_samples_per_batch=4, results_folder='results', name='vary_t.pdf'):
    """
    Save reconstructions
    """
    # Use a sample from the NF model for these plots
    z = random.normal(data_key, (1,) + nf_model.output_shape)*0.7
    _, x, _ = nf_model.inverse(nf_model.params, nf_model.state, np.zeros(1), z, ())

    # Pull noise samples
    temperatures = np.linspace(0.0, 3.0, n_samples)

    # Create the axes
    n_cols = n_samples
    n_rows = 2

    # Make the subplots
    fig, axes = plt.subplots(n_rows, n_cols); axes = axes.ravel()
    axes_iter = iter(axes)
    fig.set_size_inches(2*n_cols, 2*n_rows)

    # Generate the samples from the noisy injective flow
    _, zs, _ = nif_model.forward(nif_model.params, nif_model.state, np.zeros(1), x, (), key=None)
    zs *= temperatures[:,None]
    temp_iter = iter(temperatures)

    for j in range(n_samples//n_samples_per_batch):
        z = zs[j*n_samples_per_batch:(j + 1)*n_samples_per_batch]
        _, fz, _ = nif_model.inverse(nif_model.params, nif_model.state, np.zeros(n_samples_per_batch), z, (), key=key, test=TEST, sigma=0.0)
        fz /= (2.0**quantize_level_bits)
        fz *= (1.0 - 2*0.05)
        fz += 0.05

        for k in range(n_samples_per_batch):
            ax = next(axes_iter)
            ax.imshow(fz[k])
            ax.set_title('t = %5.3f'%(next(temp_iter)))
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.tick_params(axis='both', which='both',length=0)
            if(k == 0 and j == 0):
                ax.set_ylabel('NIF', fontsize=20)

    # Generate the samples from the normalizing flow
    _, zs, _ = nf_model.forward(nf_model.params, nf_model.state, np.zeros(1), x, (), key=None)
    zs *= temperatures[:,None]
    temp_iter = iter(temperatures)

    for j in range(n_samples//n_samples_per_batch):
        z = zs[j*n_samples_per_batch:(j + 1)*n_samples_per_batch]
        _, fz, _ = nf_model.inverse(nf_model.params, nf_model.state, np.zeros(n_samples_per_batch), z, (), key=key, test=TEST)
        fz /= (2.0**quantize_level_bits)
        fz *= (1.0 - 2*0.05)
        fz += 0.05

        for k in range(n_samples_per_batch):
            ax = next(axes_iter)
            ax.imshow(fz[k])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.tick_params(axis='both', which='both',length=0)
            if(k == 0 and j == 0):
                ax.set_ylabel('NF', fontsize=20)

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    # Save the image samples
    temp_comparison_folder = os.path.join(results_folder, 'plots')

    if(os.path.exists(temp_comparison_folder) == False):
        os.makedirs(temp_comparison_folder)

    temp_comparison_path = os.path.join(temp_comparison_folder, name)
    plt.savefig(temp_comparison_path, bbox_inches='tight', format='pdf')
    plt.close()

################################################################################################################################################

def compare_sample_over_s(data_key, key, nf_model, nif_model, quantize_level_bits, n_samples=16, n_samples_per_batch=4, results_folder='results', name='vary_s.pdf'):
    """
    Save reconstructions
    """
    # Use a sample from the NF model for these plots
    z = random.normal(data_key, (1,) + nf_model.output_shape)*0.7
    _, x, _ = nf_model.inverse(nf_model.params, nf_model.state, np.zeros(1), z, ())

    sigmas = np.linspace(0.0, 1.5, n_samples)

    # Create the axes
    n_cols = n_samples
    n_rows = 1

    # Make the subplots
    fig, axes = plt.subplots(n_rows, n_cols); axes = axes.ravel()
    axes_iter = iter(axes)
    fig.set_size_inches(2*n_cols, 2*n_rows)

    # Plot the nif model
    _, z, _ = nif_model.forward(nif_model.params, nif_model.state, np.zeros(1), x, (), key=None)
    z = z[0]

    for j, s in enumerate(sigmas):
        _, fz, _ = nif_model.inverse(nif_model.params, nif_model.state, np.zeros(1), z, (), key=key, sigma=s)
        fz /= (2.0**quantize_level_bits)
        fz *= (1.0 - 2*0.05)
        fz += 0.05

        ax = axes[j]
        ax.imshow(fz)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.tick_params(axis='both', which='both',length=0)
        ax.set_title('s = %5.3f'%(s))
        if(j == 0):
            ax.set_ylabel('NIF', fontsize=20)

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    # Save the image samples
    temp_comparison_folder = os.path.join(results_folder, 'plots')

    if(os.path.exists(temp_comparison_folder) == False):
        os.makedirs(temp_comparison_folder)

    temp_comparison_path = os.path.join(temp_comparison_folder, name)
    plt.savefig(temp_comparison_path, bbox_inches='tight', format='pdf')
    plt.close()

################################################################################################################################################

def save_sample_comparisons(key, nf_model, nif_model, quantize_level_bits, results_folder='results', name='sample_comparisons.pdf'):
    """
    Save reconstructions
    """
    # Create the axes
    n_rows = 2
    n_cols_per_model = 5
    n_cols = 2*n_cols_per_model
    n_samples = n_rows*n_cols_per_model
    n_samples_per_batch = min(n_samples, 16)

    # Make the subplots
    fig, axes = plt.subplots(n_rows, n_cols)
    axes_iter = iter(axes)
    fig.set_size_inches(2*n_cols, 2*n_rows)

    # Generate the samples from the noisy injective flow
    zs = random.normal(key, (n_samples,) + nif_model.output_shape)

    for j in range(n_samples//n_samples_per_batch):
        z = zs[j*n_samples_per_batch:(j + 1)*n_samples_per_batch]
        _, fz, _ = nif_model.inverse(nif_model.params, nif_model.state, np.zeros(n_samples_per_batch), z, (), key=key, test=TEST, sigma=0.0)
        fz /= (2.0**quantize_level_bits)
        fz *= (1.0 - 2*0.05)
        fz += 0.05

        for k in range(n_samples_per_batch):
            index = j*n_samples_per_batch + k
            u = index//n_cols_per_model
            v = index%n_cols_per_model
            ax = axes[u, v + n_cols_per_model]
            ax.imshow(fz[k])
            ax.set_axis_off()

    # Generate the samples from the normalizing flow
    zs = random.normal(key, (n_samples,) + nf_model.output_shape)

    for j in range(n_samples//n_samples_per_batch):
        z = zs[j*n_samples_per_batch:(j + 1)*n_samples_per_batch]
        _, fz, _ = nf_model.inverse(nf_model.params, nf_model.state, np.zeros(n_samples_per_batch), z, (), key=key, test=TEST)
        fz /= (2.0**quantize_level_bits)
        fz *= (1.0 - 2*0.05)
        fz += 0.05

        for k in range(n_samples_per_batch):
            index = j*n_samples_per_batch + k
            u = index//n_cols_per_model
            v = index%n_cols_per_model
            ax = axes[u, v]
            ax.imshow(fz[k])
            ax.set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    # Save the image samples
    temp_comparison_folder = os.path.join(results_folder, 'plots')

    if(os.path.exists(temp_comparison_folder) == False):
        os.makedirs(temp_comparison_folder)

    temp_comparison_path = os.path.join(temp_comparison_folder, name)
    plt.savefig(temp_comparison_path, bbox_inches='tight', format='pdf')
    plt.close()
