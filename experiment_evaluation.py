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
import numpy as onp
import glob
import tensorflow
import subprocess
import umap


print('good version')
clip_grads = jit(optimizers.clip_grads)

def save_final_samples(key, model, quantize_level_bits, temp=1.0, n_samples=64, n_samples_per_batch=4, results_folder='results', name='samples.png'):
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
        _, fz, _ = inverse(params, state, np.zeros(n_samples_per_batch), z, (), test=TEST, key=key)
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
    plt.savefig(samples_path)
    plt.close()

def save_reconstructions(key, data_loader, model, quantize_level_bits, n_samples=16, n_samples_per_batch=4, results_folder='results', name='reconstructions.png'):
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

    @jit
    def reconstruct(x, key):
        keys = random.split(key, 2)
        # log_px, finvx, _ = forward(params, state, np.zeros(n_samples_per_batch), x, (), key=inital_key)
        log_px, finvx, _ = forward(params, state, np.zeros(n_samples_per_batch), x, (), key=keys[0])
        # _, fz, _ = inverse(params, state, np.zeros(n_samples_per_batch), finvx, (), key=inital_key)
        _, fz, _ = inverse(params, state, np.zeros(n_samples_per_batch), finvx, (), key=keys[1])
        return fz

    for j in range(n_samples//n_samples_per_batch):
        key, *keys = random.split(key, 3)
        _x = data_loader(keys[0], 1, n_samples_per_batch)[0]

        keys = np.array(random.split(key, 64))
        fzs = jit(vmap(partial(reconstruct, _x)))(keys)
        fz = np.mean(fzs, axis=0)

        # log_px, finvx, _ = forward(params, state, np.zeros(n_samples_per_batch), _x, (), key=keys[0])
        # _, fz, _ = inverse(params, state, np.zeros(n_samples_per_batch), finvx, (), key=keys[1])
        fz /= (2.0**quantize_level_bits)
        fz *= (1.0 - 2*0.05)
        fz += 0.05

        for i in range(i):
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

def save_temperature_comparisons(key, nf_model, nif_model, quantize_level_bits, n_samples=16, n_samples_per_batch=4, results_folder='results', name='temperature_comparisons.png'):
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
    plt.savefig(temp_comparison_path, bbox_inches='tight')
    plt.close()

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

    # Save the results to a text file
    aggregate_stats = np.array([[nif_mean, nif_std], [nf_mean, nf_std]])
    onp.savetxt(os.path.join(results_folder, name), aggregate_stats, delimiter=",")

def interpolate_pairs(key, data_loader, nif_model, quantize_level_bits, n_pairs=5, n_points=10, results_folder='results', name='interpolation.png'):
    """
    Interpolate images
    """
    k1, k2 = random.split(key, 2)
    x_for_interpolation = data_loader(key, 1, n_pairs)[0]
    random_pairs = random.randint(key, (2*n_pairs,), minval=0, maxval=x_for_interpolation.shape[0])
    pairs_iter = iter(random_pairs)
    index_pairs = [(next(pairs_iter), next(pairs_iter)) for _ in range(n_pairs)]

    n_cols = n_points
    n_rows = len(index_pairs)

    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(2*n_cols, 2*n_rows)

    for i, (idx1, idx2) in enumerate(index_pairs):
        x = x_for_interpolation[[idx1, idx2]]
        _, finvx, _ = nif_model.forward(nif_model.params, nif_model.state, np.zeros(2), x, (), test=TEST, key=key)
        finvx1, finvx2 = finvx
        interpolation_z = np.linspace(finvx1, finvx2, n_points)
        _, fz, _ = nif_model.inverse(nif_model.params, nif_model.state, np.zeros(n_points), interpolation_z, (), test=TEST)
        fz /= (2.0**quantize_level_bits)
        fz *= (1.0 - 2*0.05)
        fz += 0.05
        for j in range(n_points):
            axes[i,j].imshow(fz[j])
            axes[i,j].set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    temp_comparison_path = os.path.join(results_folder, 'plots', name)
    plt.savefig(temp_comparison_path, bbox_inches='tight')
    plt.close()


def get_embeddings_test(key, data_loader, model, n_samples_per_batch=4):
    """
    Save reconstructions
    """
    names, output_shape, params, state, forward, inverse = model
    inital_key = key
    embeddings = []
    labels = []
    for j in range(10000//n_samples_per_batch):
        key, *keys = random.split(key, 3)
        _x, _y = data_loader((n_samples_per_batch,), None, j*n_samples_per_batch, False, True)
        keys = np.array(random.split(key, 64))
        log_px, finvx, _ = forward(params, state, np.zeros(n_samples_per_batch), _x, (), key=keys[0])
        embeddings.append(finvx)
        labels.extend(_y)
        if(j % 100 == 1):
            print(j)
    final_labels = np.array(labels)
    final_embeddings = np.concatenate(embeddings, axis = 0)
    return final_embeddings, final_labels
def get_embeddings_training(key, data_loader, model, n_samples_per_batch=4):
    """
    Save reconstructions
    """
    names, output_shape, params, state, forward, inverse = model
    inital_key = key
    embeddings = []
    labels = []
    for j in range(50000//n_samples_per_batch):
        key, *keys = random.split(key, 3)
        _x, _y = data_loader((n_samples_per_batch,), None, j*n_samples_per_batch, True, True)
        keys = np.array(random.split(key, 64))
        log_px, finvx, _ = forward(params, state, np.zeros(n_samples_per_batch), _x, (), key=keys[0])
        embeddings.append(finvx)
        labels.extend(_y)
        if(j % 100 == 1):
            print(j)
    final_labels = np.array(labels)
    final_embeddings = np.concatenate(embeddings, axis = 0)
    return final_embeddings, final_labels

def save_embeddings(key, data_loader, nf_model, nif_model, path, test = True, n_samples_per_batch=4):
    if(test):
        test_nf_embeddings, y = get_embeddings_test(key, data_loader, nf_model, n_samples_per_batch=4)
        test_nif_embeddings, y = get_embeddings_test(key, data_loader, nif_model, n_samples_per_batch=4)
        test_nf_embeddings, test_nif_embeddings, y = onp.array(test_nf_embeddings), onp.array(test_nif_embeddings), onp.array(y)
        onp.save(os.path.join(path, 'test_nif_embeddings'), test_nif_embeddings)
        onp.save(os.path.join(path, 'test_nf_embeddings'), test_nf_embeddings)
        onp.save(os.path.join(path, 'test_y'), y)
    else:
        training_nf_embeddings, y = get_embeddings_training(key, data_loader, nf_model, n_samples_per_batch=4)
        training_nif_embeddings, y = get_embeddings_training(key, data_loader, nif_model, n_samples_per_batch=4)
        training_nf_embeddings, training_nif_embeddings, training_y = onp.array(training_nf_embeddings), onp.array(training_nif_embeddings), onp.array(y)
        onp.save(os.path.join(path, 'training_nif_embeddings'), training_nif_embeddings)
        onp.save(os.path.join(path, 'training_nf_embeddings'), training_nf_embeddings)
        onp.save(os.path.join(path, 'training_y'), training_y)

def print_reduced_embeddings(key, data_loader, nf_model, nif_model, path, test=True, n_samples_per_batch=4):
    if(test):
        test_nif_embeddings = onp.array(onp.load(os.path.join(path, 'test_nif_embeddings.npy')))
        test_nf_embeddings = onp.array(onp.load(os.path.join(path, 'test_nf_embeddings.npy')))
        y = onp.array(onp.load(os.path.join(path, 'test_y.npy')))
    else:
        test_nif_embeddings = onp.array(onp.load(os.path.join(path, 'training_nif_embeddings.npy')))
        test_nf_embeddings = onp.array(onp.load(os.path.join(path, 'training_nf_embeddings.npy')))
        y = onp.array(onp.load(os.path.join(path, 'training_y.npy')))
    print(test_nif_embeddings == test_nf_embeddings)
    print(y.shape)
    nf_2d_embeddings = umap.UMAP(random_state=0).fit_transform(test_nf_embeddings, y=y)
    nif_2d_embeddings = umap.UMAP(random_state=0).fit_transform(test_nif_embeddings, y=y)
    colors = y

    def outlier_mask(data, m=2):
        return np.all(np.abs(data - np.mean(data)) < m * np.std(data), axis=1)

    #colorsnf = colors[outlier_mask(nf_2d_embeddings)]
    #colorsnif = colors[outlier_mask(nif_2d_embeddings)]
    #nf_2d_embeddings = nf_2d_embeddings[outlier_mask(nf_2d_embeddings)]
    #nif_2d_embeddings = nif_2d_embeddings[outlier_mask(nif_2d_embeddings)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(nif_2d_embeddings[:,0], nif_2d_embeddings[:,1], s=3.0, c=y, cmap='Spectral', alpha=0.6)
    scatter = axes[1].scatter(nf_2d_embeddings[:,0], nf_2d_embeddings[:,1], s=3.0, c=y, cmap='Spectral', alpha=0.6)

    axes[0].set_title('Our Method', fontdict={'fontsize': 18})
    axes[1].set_title('GLOW', fontdict={'fontsize': 18})

    #axes[0].xaxis.set_visible(False)
    #axes[0].yaxis.set_visible(False)
    #axes[1].xaxis.set_visible(False)
    #axes[1].yaxis.set_visible(False)
    #axes[0].set_xlim(1, 11)
    #axes[0].set_ylim(-4, 5)

    #axes[1].set_xlim(-5, 2)
    #axes[1].set_ylim(-5, 2)

    cbar = fig.colorbar(scatter, boundaries=np.arange(11) - 0.5)
    cbar.set_ticks(np.arange(10))
    cbar.ax.set_yticklabels(['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'])
    cbar.ax.tick_params(labelsize=12)
    plt.savefig('subplot.png')
    plt.close()







def compute_fid_score(model,
                      key,
                      quantize_level_bits,
                      temp=1.0,
                      sigma=1.0,
                      TTUR_path='TTUR/',
                      stats_path='FID/fid_stats_celeba.npz',
                      n_samples=10000,
                      n_samples_per_batch=128,
                      checkpoint_folder='results',
                      fid_score_folder_name='fid_scores',
                      sample_fid_folder_name='sample_fid_folder',
                      check_for_stats=True,
                      name='fid.txt'):

    results_folder = os.path.join(checkpoint_folder, fid_score_folder_name)
    sample_fid_folder = os.path.join(checkpoint_folder, sample_fid_folder_name)

    if(os.path.exists(results_folder) == False):
        os.makedirs(results_folder)

    if(os.path.exists(sample_fid_folder) == False):
        os.makedirs(sample_fid_folder)

    # See if we have already computed the stats for this experiment
    using_stats = False
    if(check_for_stats == True):
        our_stats_path = glob.glob(sample_fid_folder+'/*.npz')
        if(len(our_stats_path) > 0):
            our_stats_path = our_stats_path[0]
            sample_fid_folder = our_stats_path
            using_stats = True

    # If we're not using stats, then we need to compute them
    if(using_stats == False):

        # Add in the kwargs to the model
        _, _, _, _, forward, inverse = model
        model = model._replace(forward=jit(partial(forward, sigma=sigma)))
        model = model._replace(inverse=jit(partial(inverse, sigma=sigma)))

        # Generate the samples from the noisy injective flow
        number_invalid = 0
        zs = random.normal(key, (n_samples,) + model.output_shape)*temp
        for j in tqdm(np.arange(n_samples//n_samples_per_batch), leave=False):
            z = zs[j*n_samples_per_batch:(j + 1)*n_samples_per_batch]
            _, fz, _ = model.inverse(model.params, model.state, np.zeros(n_samples_per_batch), z, (), key=key, test=TEST)
            fz /= (2.0**quantize_level_bits)
            fz *= (1.0 - 2*0.05)
            fz += 0.05
            for k in range(j*n_samples_per_batch, (j + 1)*n_samples_per_batch):
                # Save each of the images
                sample_name = 'sample_%d.jpg'%k
                if(np.any(np.isnan(fz[k-j*n_samples_per_batch]))):
                    number_invalid += 1
                    continue
                if(np.any(np.isinf(fz[k-j*n_samples_per_batch]))):
                    number_invalid += 1
                    continue
                matplotlib.image.imsave(os.path.join(sample_fid_folder, sample_name), fz[k-j*n_samples_per_batch])

    # If a bunch of the samples are infinity or nan, then don't even bother with the FID
    if(using_stats == False and number_invalid > n_samples*0.2):
        fid_score = np.nan
    else:
        # Compute the fid score
        try:
            TTUR_command = ['python', os.path.join(TTUR_path, 'fid.py'), sample_fid_folder, stats_path, '--gpu', '0']
            proc = subprocess.Popen(TTUR_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate()
            fid_score = float(str(out).split('FID:')[1].strip()[:-3])
        except:
            TTUR_command = ['python', os.path.join(TTUR_path, 'fid.py'), sample_fid_folder, stats_path]
            proc = subprocess.Popen(TTUR_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate()
            try:
                fid_score = float(str(out).split('FID:')[1].strip()[:-3])
            except:
                print('out:', out)
                print('\n\n')
                print('err:', err)
                assert 0

    # Save the results to a text file
    fid_save_path = os.path.join(results_folder, name)
    onp.savetxt(fid_save_path, np.array([fid_score]), delimiter=",")

def save_fid_scores(nf_model,
                    nif_model,
                    key,
                    quantize_level_bits,
                    temp=1.0,
                    TTUR_path='TTUR/',
                    stats_path='FID/fid_stats_celeba.npz',
                    n_samples=10000,
                    n_samples_per_batch=128,
                    results_folder='results',
                    name='fid.txt'):
    # stats_name = 'fid_stats_cifar10_train.npz'

    nf_fid_folder = os.path.join(results_folder, 'tmp_nf_fid_folder/')
    nif_fid_folder = os.path.join(results_folder, 'tmp_nif_fid_folder/')

    if(os.path.exists(nf_fid_folder) == False):
        os.makedirs(nf_fid_folder)
    if(os.path.exists(nif_fid_folder) == False):
        os.makedirs(nif_fid_folder)

    # Generate the samples from the noisy injective flow
    zs = random.normal(key, (n_samples,) + nif_model.output_shape)*temp
    for j in tqdm(np.arange(n_samples//n_samples_per_batch)):
        break
        z = zs[j*n_samples_per_batch:(j + 1)*n_samples_per_batch]
        _, fz, _ = nif_model.inverse(nif_model.params, nif_model.state, np.zeros(n_samples_per_batch), z, (), key=key, test=TEST)
        fz /= (2.0**quantize_level_bits)
        fz *= (1.0 - 2*0.05)
        fz += 0.05
        for k in range(j*n_samples_per_batch, (j + 1)*n_samples_per_batch):
            # Save each of the images
            sample_name = 'nif_sample_%d.jpg'%k
            if(np.any(np.isnan(fz[k-j*n_samples_per_batch]))):
                continue
            matplotlib.image.imsave(os.path.join(nif_fid_folder, sample_name), fz[k-j*n_samples_per_batch])

    # Compute the fid score
    try:
        proc = subprocess.Popen(['python', os.path.join(TTUR_path, 'fid.py'), nif_fid_folder, stats_path, '--gpu', '0'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        nif_fid_score = float(str(out).split('FID:')[1].strip()[:-3])
    except:
        proc = subprocess.Popen(['python', os.path.join(TTUR_path, 'fid.py'), nif_fid_folder, stats_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        nif_fid_score = float(str(out).split('FID:')[1].strip()[:-3])

    # Generate the samples from the normalizing flow
    zs = random.normal(key, (n_samples,) + nf_model.output_shape)*temp
    for j in tqdm(np.arange(n_samples//n_samples_per_batch)):
        z = zs[j*n_samples_per_batch:(j + 1)*n_samples_per_batch]
        _, fz, _ = nf_model.inverse(nf_model.params, nf_model.state, np.zeros(n_samples_per_batch), z, (), key=key, test=TEST)
        fz /= (2.0**quantize_level_bits)
        fz *= (1.0 - 2*0.05)
        fz += 0.05
        for k in range(j*n_samples_per_batch, (j + 1)*n_samples_per_batch):
        # Save each of the images
            sample_name = 'nf_sample_%d.jpg'%k
            if(np.any(np.isnan(fz[k-j*n_samples_per_batch]))):
                continue
            matplotlib.image.imsave(os.path.join(nf_fid_folder, sample_name), fz[k-j*n_samples_per_batch])

    # Compute the fid score
    try:
        proc = subprocess.Popen(['python', os.path.join(TTUR_path, 'fid.py'), nf_fid_folder, stats_path, '--gpu', '0'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        nf_fid_score = float(str(out).split('FID:')[1].strip()[:-3])
    except:
        proc = subprocess.Popen(['python', os.path.join(TTUR_path, 'fid.py'), nf_fid_folder, stats_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        nf_fid_score = float(str(out).split('FID:')[1].strip()[:-3])

    # Save the results to a text file
    fid_scores = np.array([nif_fid_score, nf_fid_score])
    onp.savetxt(os.path.join(results_folder, name), fid_scores, delimiter=",")
