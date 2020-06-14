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
from jax.tree_util import tree_map,tree_leaves,tree_structure,tree_unflatten
from collections import namedtuple
from datasets import get_celeb_dataset
import argparse
from jax.lib import xla_bridge
import pickle
import jax.nn.initializers as jaxinit
import jax.numpy as np
import glob
clip_grads = jit(optimizers.clip_grads)

n_gpus = xla_bridge.device_count()
print('n_gpus:', n_gpus)

parser = argparse.ArgumentParser(description='Training Noisy Injective Flows')

parser.add_argument('--name', action='store', type=str,
                   help='Name of model', default='CelebADefault')
parser.add_argument('--batchsize', action='store', type=int,
                   help='Batch Size, default=64', default=64)
parser.add_argument('--dataset', action='store', type=str,
                   help='Dataset to load, default=CelebA', default='CelebA')
parser.add_argument('--numimage', action='store', type=int,
                   help='Number of images to load from selected dataset, default=50000', default=200000)
parser.add_argument('--quantize', action='store', type=int,
                   help='Sets the value of quantize_level_bits, default=5', default=5)
parser.add_argument('--startingit', action ='store', type=int,
                   help='Sets the training iteration to start on. There must be a stored file for this to occure', default=0)
parser.add_argument('--model', action ='store', type=str,
                   help='Sets the model to use', default='CelebADefault')
parser.add_argument('--printevery', action='store', type=int,
                   help='Sets the number of iterations between each test', default=500)

args = parser.parse_args()

batch_size = args.batchsize
dataset = args.dataset
n_images = args.numimage
quantize_level_bits = args.quantize
start_it = args.startingit
experiment_name = args.name
model_type = args.model
print_every = args.printevery

experiment_folder = os.path.join('Experiments', experiment_name)
start_iter_folder = os.path.join(experiment_folder, str(start_it))

# Make sure that the starting iteration is valid
if(start_it > 0):
    if(os.path.exists(start_iter_folder) == False):
        assert 0, 'Invalid starting iteration'

# Get the most recent training iteration
if(start_it == -1):
    completed_iterations = []
    for root, dirs, _ in os.walk(experiment_folder):
        for d in dirs:
            try:
                completed_iterations.append(int(d))
            except:
                pass
    completed_iterations = sorted(completed_iterations)
    if(len(completed_iterations) == 0):
        start_it = 0
    else:
        start_it = completed_iterations[-1]
    start_iter_folder = os.path.join(experiment_folder, str(start_it))

print('Start iteration is', start_it)

print('Done Parsing Arguments')

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

from CelebA_512 import CelebA512
from CelebA_256 import CelebA256
from CelebA_128 import CelebA128

from CIFAR10_512 import CIFAR512
from CIFAR10_256 import CIFAR256

from CelebAImportanceSample128 import CelebAIS128
from CelebAImportanceSample256 import CelebAIS256


from STL10_default_model import STL10Default
# from STL10_512 import STL10512
from upsample_vs_multiscale import CelebAUpscale
from CelebAImportanceSample import CelebAImportanceSample

if(model_type == 'CelebA512'):
    assert dataset == 'CelebA', 'Dataset mismatch'
    nf, nif = CelebA512(False, quantize_level_bits), CelebA512(True, quantize_level_bits)
elif(model_type == 'CelebA256'):
    assert dataset == 'CelebA', 'Dataset mismatch'
    nf, nif = CelebA256(False, quantize_level_bits), CelebA256(True, quantize_level_bits)
elif(model_type == 'CelebA128'):
    assert dataset == 'CelebA', 'Dataset mismatch'
    nf, nif = CelebA128(False, quantize_level_bits), CelebA128(True, quantize_level_bits)
elif(model_type == 'CelebAIS128'):
    assert dataset == 'CelebA', 'Dataset mismatch'
    nf, nif = CelebAIS128(False, quantize_level_bits), CelebAIS128(True, quantize_level_bits)
elif(model_type == 'CelebAIS256'):
    assert dataset == 'CelebA', 'Dataset mismatch'
    nf, nif = CelebAIS256(False, quantize_level_bits), CelebAIS256(True, quantize_level_bits)
elif(model_type == 'CIFAR512'):
    assert dataset == 'CIFAR', 'Dataset mismatch'
    nf, nif = CIFAR512(False, quantize_level_bits), CIFAR512(True, quantize_level_bits)
elif(model_type == 'CIFAR256'):
    assert dataset == 'CIFAR', 'Dataset mismatch'
    nf, nif = CIFAR256(False, quantize_level_bits), CIFAR256(True, quantize_level_bits)
elif(model_type == 'STL10Default'):
    assert dataset == 'STL10', 'Dataset mismatch'
    nf, nif = STL10Default(False, quantize_level_bits), STL10Default(True, quantize_level_bits)
elif(model_type == 'STL512'):
    assert dataset == 'STL10', 'Dataset mismatch'
    nf, nif = STL10512(False, quantize_level_bits), STL10512(True, quantize_level_bits)
elif(model_type == 'CelebAUpsample'):
    nf, nif = CelebAUpscale(False, quantize_level_bits), CelebAUpscale(True, quantize_level_bits)
elif(model_type == 'CelebAImportanceSample'):
    nf, nif = CelebAImportanceSample(False, quantize_level_bits), CelebAImportanceSample(True, quantize_level_bits)
else:
    assert 0, 'Invalid model type.'

print('Done Loading Model')

Model = namedtuple('model', 'names output_shape params state forward inverse')

models = []
for flow in [nf, nif]:
    init_fun, forward, inverse = flow
    key = random.PRNGKey(0)
    names, output_shape, params, state = init_fun(key, x_shape, ())
    n_params = ravel_pytree(params)[0].shape[0]
    print('n_params', n_params)
    z_dim = output_shape[-1]
    flow_model = ((names, output_shape, params, state), forward, inverse)
    actnorm_names = [name for name in tree_flatten(names)[0] if 'act_norm' in name]
    if(start_it == 0):
        params = multistep_flow_data_dependent_init(None,
                                                    actnorm_names,
                                                    flow_model,
                                                    (),
                                                    'actnorm_seed',
                                                    key,
                                                    data_loader=data_loader,
                                                    n_seed_examples=1000,
                                                    batch_size=8,
                                                    notebook=False)
    n_params = ravel_pytree(params)[0].shape[0]
    models.append(Model(names, output_shape, params, state, forward, inverse))
nf_model, nif_model = models
print('Done With Data Dependent Init')

@partial(jit, static_argnums=(0,))
def nll(forward, params, state, x, **kwargs):
    log_px, z, updated_state = forward(params, state, np.zeros(x.shape[0]), x, (), **kwargs)
    return -np.mean(log_px), updated_state

@partial(pmap, static_broadcasted_argnums=(0, 1, 2), axis_name='batch')
def spmd_update(forward, opt_update, get_params, i, opt_state, state, x_batch, key):
    params = get_params(opt_state)
    (val, state), grads = jax.value_and_grad(partial(nll, forward), has_aux=True)(params, state, x_batch, key=key, test=TRAIN)
    g = jax.lax.psum(grads, 'batch')
    g = clip_grads(g, 5.0)
    opt_state = opt_update(i, g, opt_state)
    return val, state, opt_state

# Create the optimizer

def lr_schedule(i, lr_decay=1.0, max_lr=2e-5):
    warmup_steps = 10000
    return np.where(i < warmup_steps, max_lr*i/warmup_steps, max_lr*(lr_decay**(i - warmup_steps)))

opt_init, opt_update, get_params = optimizers.adam(lr_schedule)
opt_update = jit(opt_update)
opt_state_nf = opt_init(nf_model.params)
opt_state_nif = opt_init(nif_model.params)

def load_pytree(treedef, dir_save):
    with open(dir_save,'rb') as f: leaves = pickle.load(f)
    return tree_unflatten(treedef, leaves)


if(start_it != 0):
    opt_state_nf = load_pytree(tree_structure(opt_state_nf), os.path.join(start_iter_folder, 'opt_state_nf_leaves.p'))
    opt_state_nif = load_pytree(tree_structure(opt_state_nif), os.path.join(start_iter_folder, 'opt_state_nif_leaves.p'))
    state_nf = load_pytree(tree_structure(nf_model.state), os.path.join(start_iter_folder, 'state_nf_leaves.p'))
    state_nif = load_pytree(tree_structure(nif_model.state), os.path.join(start_iter_folder, 'state_nf_leaves.p'))

    nf_model = nf_model._replace(state=state_nf)
    nif_model = nif_model._replace(state=state_nif)
    nf_model = nf_model._replace(params=opt_state_nf)
    nif_model = nif_model._replace(params=opt_state_nif)
    with open(os.path.join(start_iter_folder, 'misc.p'),'rb') as f:
        misc = pickle.load(f)
    key = misc['key']

    start_it += 1
else:
    state_nf, state_nif = nf_model.state, nif_model.state
    misc = dict()

print('Done Loading Checkpoint')

# Fill the update function with the optimizer params
filled_spmd_update_nf = partial(spmd_update, nf_model.forward, opt_update, get_params)
filled_spmd_update_nif = partial(spmd_update, nif_model.forward, opt_update, get_params)

losses_nf, losses_nif = [], []

# Need to copy the optimizer state and network state before it gets passed to pmap
replicate_array = lambda x: onp.broadcast_to(x, (n_gpus,) + x.shape)
replicated_state_nf = tree_map(replicate_array, state_nf)
replicated_opt_state_nf = tree_map(replicate_array, opt_state_nf)
replicated_opt_state_nif, replicated_state_nif = tree_map(replicate_array, opt_state_nif), tree_map(replicate_array, state_nif)

def savePytree(pytree, dir_save):
    with open(dir_save,'wb') as f: pickle.dump(tree_leaves(pytree), f)

if(os.path.exists('Experiments') == False):
    os.mkdir('Experiments')

if(os.path.exists(experiment_folder) == False):
    os.mkdir(experiment_folder)

print('About to Start Experiments')

pbar = tqdm(np.arange(start_it, 500000))
for i in pbar:
    key, *keys = random.split(key, 3)

    # Take the next batch of data and random keys
    x_batch = data_loader(keys[0], n_gpus, batch_size)
    train_keys = np.array(random.split(keys[1], n_gpus))
    replicated_i = np.ones(n_gpus)*i

    if(i == start_it):
        replicated_val_nf, replicated_state_nf, replicated_opt_state_nf = filled_spmd_update_nf(replicated_i, replicated_opt_state_nf, replicated_state_nf, x_batch, train_keys)
    replicated_val_nif, replicated_state_nif, replicated_opt_state_nif = filled_spmd_update_nif(replicated_i, replicated_opt_state_nif, replicated_state_nif, x_batch, train_keys)

    # Convert to bits/dimension
    val_nf, val_nif = replicated_val_nf[0], replicated_val_nif[0]
    val_nf, val_nif = val_nf/np.prod(x_shape)/np.log(2), val_nif/np.prod(x_shape)/np.log(2)

    losses_nf.append(val_nf)
    losses_nif.append(val_nif)

    progress_str = f'Bits/Dim: NF: {val_nf:.3f}, NIF: {val_nif:.3f}'
    pbar.set_description(progress_str)

    if(i%print_every == 0):

        # Save Model
        # Get the trained parameters and the state
        opt_state_nf, opt_state_nif = tree_map(lambda x:x[0], replicated_opt_state_nf), tree_map(lambda x:x[0], replicated_opt_state_nif)
        state_nf, state_nif = tree_map(lambda x:x[0], replicated_state_nf), tree_map(lambda x:x[0], replicated_state_nif)

        opt_state_nf_leaves, opt_state_nif_leaves = tree_leaves(opt_state_nf), tree_leaves(opt_state_nif)
        state_nf_leaves, state_nif_leaves = tree_leaves(state_nf), tree_leaves(state_nif)

        iteration_folder = os.path.join(experiment_folder, str(i))
        if not os.path.exists(iteration_folder):
            os.mkdir(iteration_folder)

        savePytree(opt_state_nf_leaves, os.path.join(iteration_folder, 'opt_state_nf_leaves.p'))
        savePytree(opt_state_nif_leaves, os.path.join(iteration_folder, 'opt_state_nif_leaves.p'))
        savePytree(state_nf_leaves, os.path.join(iteration_folder, 'state_nf_leaves.p'))
        savePytree(state_nif_leaves, os.path.join(iteration_folder, 'state_nif_leaves.p'))

        onp.savetxt(os.path.join(iteration_folder, 'losses_nf.txt'), losses_nf, delimiter=",")
        onp.savetxt(os.path.join(iteration_folder, 'losses_nif.txt'), losses_nif, delimiter=",")
        misc['key'] = key

        with open(os.path.join(iteration_folder, 'misc.p'), 'wb') as f:
            pickle.dump(misc, f)
