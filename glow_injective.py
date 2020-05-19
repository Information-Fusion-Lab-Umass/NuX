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

parser.add_argument('--name', action='store', type=str, nargs = 1,
                   help='Name of model', default = '0')
parser.add_argument('--batchsize', action='store', type=int, nargs = 1,
                   help='Batch Size, default = 64', default = 64)
parser.add_argument('--dataset', action='store', type=str, nargs = 1,
                   help='Dataset to load, default = CelebA', default = 'CelebA')
parser.add_argument('--numimage', action='store', type=int, nargs = 1,
                   help='Number of images to load from selected dataset, default = 50000', default = 50000)
parser.add_argument('--quantize', action='store', type=int, nargs = 1,
                   help='Sets the value of quantize_level_bits, default = 5', default = 5)
parser.add_argument('--startingit', action ='store', type=int, nargs=1,
                   help = 'Sets the training iteration to start on. There must be a stored file for this to occure', default = 0)


parser.add_argument('--printevery', action = 'store', type=int, nargs = 1,
                   help='Sets the number of iterations between each test', default = 2)

args = parser.parse_args()

batch_size = args.batchsize
dataset = args.dataset 
n_images = args.numimage
quantize_level_bits = args.quantize
start_it = args.startingit[0]
experiment_name = args.name

print_every = args.printevery


if(dataset == 'CelebA'):
    x_train = get_celeb_dataset(quantize_level_bits=quantize_level_bits, strides=(2, 2), crop=(29, 9), n_images=n_images)
elif(dataset == 'CIFAR'):
    x_train, train_labels, test_images, test_labels = get_cifar10_data(quantize_level_bits=quantize_level_bits)

def ResidualBlock(n_filters, norm_type='instance', use_wn=False):
    if(norm_type == 'batch_norm'):
        norm = spp.BatchNorm()
    elif(norm_type == 'instance'):
        norm = spp.InstanceNorm()
    else:
        assert 0
        
    one_by_one = lambda bias, **kwargs: spp.Conv(n_filters, filter_shape=(1, 1), padding=((0, 0), (0, 0)), bias=bias, **kwargs)
    three_by_three = lambda bias, **kwargs: spp.Conv(n_filters, filter_shape=(3, 3), padding=((1, 1), (1, 1)), bias=bias, **kwargs)
        
    network = spp.sequential(three_by_three(bias=True),
                             norm,
                             spp.Relu(), 
                             one_by_one(bias=True),
                             norm,
                             spp.Relu(),
                             three_by_three(bias=True))
    return spp.Residual(network)
def ResNet(out_shape, norm_type='instance', n_filters=64, n_blocks=2):
    _, _, channels = out_shape
    return spp.sequential(spp.Conv(n_filters, filter_shape=(1, 1), padding=((0, 0), (0, 0)), bias=True),
                          spp.Relu(),
                          *[ResidualBlock(n_filters, norm_type=norm_type) for i in range(n_blocks)],
                          spp.Conv(2*channels, filter_shape=(3, 3), padding=((1, 1), (1, 1)), bias=True, W_init=jaxinit.zeros, b_init=jaxinit.zeros),
                          spp.Split(2, axis=-1), 
                          spp.parallel(spp.Tanh(), spp.Identity()))  # log_s, t
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
    layers = [GLOWBlock(GLOWNet, masked=False, name=next(name_iter))]*1
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
        prior_layers = [AffineCoupling(FlatTransform), ActNorm(name=next(an_names)), Reverse()]*2
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
    if(True):
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
opt_init, opt_update, get_params = optimizers.adam(1e-4)
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

    start_it += 1



# Fill the update function with the optimizer params
filled_spmd_update_nf = partial(spmd_update, nf_model.forward, opt_update, get_params)
filled_spmd_update_nif = partial(spmd_update, nif_model.forward, opt_update, get_params)

losses_nf, losses_nif = [], []

# Need to copy the optimizer state and network state before it gets passed to pmap
replicate_array = lambda x: onp.broadcast_to(x, (n_gpus,) + x.shape)
replicated_opt_state_nf, replicated_state_nf = tree_map(replicate_array, opt_state_nf), tree_map(replicate_array, nf_model.state)
replicated_opt_state_nif, replicated_state_nif = tree_map(replicate_array, opt_state_nif), tree_map(replicate_array, nif_model.state)


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






        '''

        #Tests
        
        # Create new models
        nf_model = Model(nf_model.names, nf_model.output_shape, get_params(opt_state_nf), state_nf, nf_model.forward, nf_model.inverse)
        nif_model = Model(nif_model.names, nif_model.output_shape, get_params(opt_state_nif), state_nif, nif_model.forward, nif_model.inverse)
        
        # Pull noise samples
        n_samples = 18
        n_samples_per_batch = 2
        eval_key = random.PRNGKey(0)
        temperatures = np.linspace(0.5, 3.0, n_samples)
        
        # Create the axes
        n_cols = n_samples
        n_rows = 2

        # Make the subplots
        fig, axes = plt.subplots(n_rows, n_cols); axes = axes.ravel()
        axes_iter = iter(axes)
        fig.set_size_inches(2*n_cols, 2*n_rows)

        # Generate the samples from the noisy injective flow
        zs = random.normal(eval_key, (n_samples, z_dim))
        zs *= temperatures[:,None]
        temp_iter = iter(temperatures)
        
        for j in range(n_samples//n_samples_per_batch):
            z = zs[j*n_samples_per_batch:(j + 1)*n_samples_per_batch]
            _, fz, _ = nif_model.inverse(nif_model.params, nif_model.state, np.zeros(n_samples_per_batch), z, (), key=eval_key, test=TEST)
            fz /= (2.0**quantize_level_bits) # Put the image (mostly) between 0 and 1
            fz *= (1.0 - 2*0.05)
            fz += 0.05

            for k in range(n_samples_per_batch):
                ax = next(axes_iter)
                ax.imshow(fz[k])
                ax.set_title('T = %5.3f'%(next(temp_iter)))
                ax.set_axis_off()
                
        # Generate the samples from the normalizing flow
        zs = random.normal(eval_key, (n_samples,) + x_train.shape[1:])
        zs *= temperatures[:,None,None,None]
        temp_iter = iter(temperatures)
                
        for j in range(n_samples//n_samples_per_batch):
            z = zs[j*n_samples_per_batch:(j + 1)*n_samples_per_batch]
            _, fz, _ = nf_model.inverse(nf_model.params, nf_model.state, np.zeros(n_samples_per_batch), z, (), key=eval_key, test=TEST)
            fz /= (2.0**quantize_level_bits) # Put the image (mostly) between 0 and 1
            fz *= (1.0 - 2*0.05)
            fz += 0.05

            for k in range(n_samples_per_batch):
                ax = next(axes_iter)
                ax.imshow(fz[k])
                ax.set_axis_off()
                
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)        
        plt.show()


fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10, 10)
ax.plot(pd.DataFrame(losses_nf).ewm(alpha=0.001).mean()[500:], label='Normalizing Flow')
ax.plot(pd.DataFrame(losses_nif).ewm(alpha=0.001).mean()[500:], label='Noisy Injective Flow')
ax.set_title('Bits/Dimension During Training')
ax.set_xlabel('Iterations')
ax.set_ylabel('Bits/Dim')
ax.legend()

'''

















