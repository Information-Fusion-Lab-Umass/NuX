import os
from tqdm import tqdm
from jax import random, vmap, jit, value_and_grad
from jax.experimental import optimizers, stax
import jax.numpy as np
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

n_gpus = xla_bridge.device_count()

parser = argparse.ArgumentParser(description='Training Noisy Injective Flows')

parser.add_argument('--name', action='store', type=str,
                   help='Name of model', default = '0')
parser.add_argument('--batchsize', action='store', type=int,
                   help='Batch Size, default = 64', default = 64)
parser.add_argument('--dataset', action='store', type=str,
                   help='Dataset to load, default = CelebA', default = 'CelebA')
parser.add_argument('--numimage', action='store', type=int,
                   help='Number of images to load from selected dataset, default = 50000', default = 50000)
parser.add_argument('--quantize', action='store', type=int,
                   help='Sets the value of quantize_level_bits, default = 5', default = 5)
parser.add_argument('--startingit', action ='store', type=int,
                   help = 'Sets the training iteration to start on. There must be a stored file for this to occure', default = 0)


parser.add_argument('--printevery', action = 'store', type=int,
                   help='Sets the number of iterations between each test', default = 2)

args = parser.parse_args()

batch_size = args.batchsize
dataset = args.dataset 
n_images = args.numimage
quantize_level_bits = args.quantize
start_it = args.startingit
experiment_name = args.name

print_every = args.printevery


if(dataset == 'CelebA'):
    x_train = get_celeb_dataset(quantize_level_bits=quantize_level_bits, strides=(2, 2), crop=(29, 9), n_images=n_images)
elif(dataset == 'CIFAR'):
    x_train, train_labels, test_images, test_labels = get_cifar10_data(quantize_level_bits=quantize_level_bits)

def GLOWNet(out_shape, n_filters=512):
    _, _, channels = out_shape
    return spp.sequential(spp.Conv(n_filters, filter_shape=(3, 3), padding=((1, 1), (1, 1)), bias=True, weightnorm=False),
#                           spp.InstanceNorm(),
                          spp.Relu(),
                          spp.Conv(n_filters, filter_shape=(1, 1), padding=((0, 0), (0, 0)), bias=True, weightnorm=False),
#                           spp.InstanceNorm(),
                          spp.Relu(),
                          spp.Conv(2*channels, filter_shape=(3, 3), padding=((1, 1), (1, 1)), bias=True, weightnorm=False, W_init=jaxinit.zeros, b_init=jaxinit.zeros),
                          spp.Split(2, axis=-1), 
                          spp.parallel(spp.Tanh(), spp.Identity()))  # log_s, t
def FlatTransform(out_shape, n_hidden_layers=4, layer_size=1024):
    dense_layers = [spp.Dense(layer_size), spp.Relu()]*n_hidden_layers
    return spp.sequential(*dense_layers,
                          spp.Dense(out_shape[-1]*2),
                          spp.Split(2, axis=-1), 
                          spp.parallel(spp.Tanh(), spp.Identity())) # log_s, t
def GLOW(name_iter, norm_type='instance', conditioned_actnorm=False):
    layers = [GLOWBlock(GLOWNet, masked=False, name=next(name_iter))]*1 #32
    return sequential_flow(Squeeze(), Debug(''), *layers, UnSqueeze())
def RealNDP(z_dim=50):
    debug_kwargs = dict(print_init_shape=True, print_forward_shape=False, print_inverse_shape=False, compare_vals=False)
    
    an_names = iter(['act_norm_%d'%i for i in range(100)])
    name_iter = iter(['glow_%d'%i for i in range(100)])
    def multi_scale(flow):
        return sequential_flow(GLOW(name_iter),
                               Squeeze(),
                               FactorOut(2),
                               factored_flow(flow, Identity()),
                               FanInConcat(2),
                               UnSqueeze())
    flow = GLOW(name_iter)
    #flow = multi_scale(flow)
    #flow = multi_scale(flow)
    #flow = multi_scale(flow)
    if(z_dim is not None):
        prior_layers = [AffineCoupling(FlatTransform), ActNorm(name=next(an_names)), Reverse()]*2 #10
        prior_flow = sequential_flow(*prior_layers, AffineGaussianPriorFullCov(z_dim))
        prior_flow = TallAffineDiagCov(prior_flow, z_dim)
#         prior_flow = AffineGaussianPriorFullCov(z_dim)
    else:
        prior_flow = UnitGaussianPrior()
    flow = sequential_flow(Dequantization(scale=2**quantize_level_bits), 
                           Logit(), 
                           flow,
                           Flatten(),
                           prior_flow)
    return flow

nf, nif = RealNDP(None), RealNDP(512)
Model = namedtuple('model', 'names output_shape params state forward inverse')

models = []
for flow in [nf, nif]:
    init_fun, forward, inverse = flow
    key = random.PRNGKey(0)
    names, output_shape, params, state = init_fun(key, x_train.shape[1:], ())
    z_dim = output_shape[-1]
    flow_model = ((names, output_shape, params, state), forward, inverse) 
    actnorm_names = [name for name in tree_flatten(names)[0] if 'act_norm' in name]
    if(start_it != 0):
        params = multistep_flow_data_dependent_init(x_train,
                                           actnorm_names,
                                           flow_model,
                                           (),
                                           'actnorm_seed',
                                           key,
                                           n_seed_examples=8,
                                           batch_size=8,
                                           notebook=False)

    models.append(Model(names, output_shape, params, state, forward, inverse))
nf_model, nif_model = models

@partial(jit, static_argnums=(0,))
def nll(forward, params, state, x, **kwargs):
    log_px, z, updated_state = forward(params, state, np.zeros(x.shape[0]), x, (), **kwargs)
    return -np.mean(log_px), updated_state

@partial(pmap, static_broadcasted_argnums=(0, 1, 2), axis_name='batch')
def spmd_update(forward, opt_update, get_params, i, opt_state, state, x_batch, key):
    params = get_params(opt_state)
    (val, state), grads = jax.value_and_grad(partial(nll, forward), has_aux=True)(params, state, x_batch, key=key, test=TRAIN)
    g = jax.lax.psum(grads, 'batch')
    opt_state = opt_update(i, g, opt_state)
    return val, state, opt_state

# Create the optimizer

def lr_schedule(i, lr_decay=1.0, max_lr=1e-4):
    warmup_steps = 2000
    return np.where(i < warmup_steps, max_lr*i/warmup_steps, max_lr*(lr_decay**(i - warmup_steps)))

opt_init, opt_update, get_params = optimizers.adam(lr_schedule)
opt_update = jit(opt_update)
opt_state_nf = opt_init(nf_model.params)
opt_state_nif = opt_init(nif_model.params)


def load_pytree(treedef, dir_save):
    with open(dir_save,'rb') as f: leaves = pickle.load(f)
    return tree_unflatten(treedef, leaves)


if(start_it != 0):
    opt_state_nf = load_pytree(tree_structure(opt_state_nf), 'Experiments/' + str(experiment_name) + '/' + str(start_it) + '/' + 'opt_state_nf_leaves.p')
    opt_state_nif = load_pytree(tree_structure(opt_state_nif), 'Experiments/' + str(experiment_name) + '/' + str(start_it) + '/' + 'opt_state_nif_leaves.p')
    state_nf = load_pytree(tree_structure(nf_model.state), 'Experiments/' + str(experiment_name) + '/' + str(start_it) + '/' + 'state_nf_leaves.p')
    state_nif = load_pytree(tree_structure(nif_model.state), 'Experiments/' + str(experiment_name) + '/' + str(start_it) + '/' + 'state_nf_leaves.p')

    nf_model.state, nif_model.state = state_nf.state_nif
    nf_model.params, nif_model.params = get_params(opt_state_nf), get_params(opt_state_nif)

    start_it += 1



# Fill the update function with the optimizer params
filled_spmd_update_nf = partial(spmd_update, nf_model.forward, opt_update, get_params)
filled_spmd_update_nif = partial(spmd_update, nif_model.forward, opt_update, get_params)

losses_nf, losses_nif = [], []

# Need to copy the optimizer state and network state before it gets passed to pmap
replicate_array = lambda x: onp.broadcast_to(x, (n_gpus,) + x.shape)
replicated_opt_state_nf, replicated_state_nf = tree_map(replicate_array, opt_state_nf), tree_map(replicate_array, state_nf)
replicated_opt_state_nif, replicated_state_nif = tree_map(replicate_array, opt_state_nif), tree_map(replicate_array, state_nif)


def savePytree(pytree, dir_save):
    with open(dir_save,'wb') as f: pickle.dump(tree_leaves(pytree), f)


if not os.path.exists('Experiments/' + str(experiment_name)):
    os.mkdir('Experiments/' + str(experiment_name))


for i in np.arange(start_it, 100000):
    key, *keys = random.split(key, 3)
    
    # Take the next batch of data and random keys
    batch_idx = random.randint(keys[0], (n_gpus, batch_size), minval=0, maxval=x_train.shape[0])
    x_batch = x_train[batch_idx,:]
    train_keys = np.array(random.split(keys[1], n_gpus))
    replicated_i = np.ones(n_gpus)*i
    
    replicated_val_nf, replicated_state_nf, replicated_opt_state_nf = filled_spmd_update_nf(replicated_i, replicated_opt_state_nf, replicated_state_nf, x_batch, train_keys)
    replicated_val_nif, replicated_state_nif, replicated_opt_state_nif = filled_spmd_update_nif(replicated_i, replicated_opt_state_nif, replicated_state_nif, x_batch, train_keys)
    
    # Convert to bits/dimension
    val_nf, val_nif = replicated_val_nf[0], replicated_val_nif[0]
    val_nf, val_nif = val_nf/np.prod(x_train.shape[1:])/np.log(2), val_nif/np.prod(x_train.shape[1:])/np.log(2)

    losses_nf.append(val_nf)
    losses_nif.append(val_nif)
    print(f'Negative Log Likelihood: NF: {val_nf:.3f}, NIF: {val_nif:.3f}') 
    
    if(i%print_every == 0):

        #Save Model
        # Get the trained parameters and the state
        opt_state_nf, opt_state_nif = tree_map(lambda x:x[0], replicated_opt_state_nf), tree_map(lambda x:x[0], replicated_opt_state_nif)
        state_nf, state_nif = tree_map(lambda x:x[0], replicated_state_nf), tree_map(lambda x:x[0], replicated_state_nif)

        opt_state_nf_leaves, opt_state_nif_leaves = tree_leaves(opt_state_nf), tree_leaves(opt_state_nif)
        state_nf_leaves, state_nif_leaves = tree_leaves(state_nf), tree_leaves(state_nif)

        if not os.path.exists('Experiments/' + str(experiment_name) + '/' + str(i) + '/'):
            os.mkdir('Experiments/' + str(experiment_name) + '/' + str(i) + '/')
        savePytree(opt_state_nf_leaves, 'Experiments/' + str(experiment_name) + '/' + str(i) + '/' + 'opt_state_nf_leaves.p')
        savePytree(opt_state_nif_leaves, 'Experiments/' + str(experiment_name) + '/' + str(i) + '/' + 'opt_state_nif_leaves.p')
        savePytree(state_nf_leaves, 'Experiments/' + str(experiment_name) + '/' + str(i) + '/' + 'state_nf_leaves.p')
        savePytree(state_nif_leaves, 'Experiments/' + str(experiment_name) + '/' + str(i) + '/' + 'state_nif_leaves.p')
















