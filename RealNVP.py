from tqdm import tqdm
from jax import random, vmap, jit, value_and_grad
from jax.experimental import optimizers, stax
import jax.numpy as np
import staxplusplus as spp
from normalizing_flows import *
import matplotlib.pyplot as plt
from datasets import get_celeb_dataset
import jax.nn.initializers
import os
import pickle
import argparse
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten, tree_map
from jax.lib import xla_bridge

def ResidualBlock(n_channels):
    """
    Create the residual component of the full residual network.  The input and output shapes should be the same.
    """
    network = spp.sequential(spp.Conv(n_channels, filter_shape=(1, 1), padding=((0, 0), (0, 0)), bias=False, weightnorm=True),
                             spp.Relu(),
                             spp.Conv(n_channels, filter_shape=(3, 3), padding=((1, 1), (1, 1)), bias=False, weightnorm=True),
                             spp.Relu(),
                             spp.Conv(n_channels, filter_shape=(1, 1), padding=((0, 0), (0, 0)), bias=True, weightnorm=True))
    return spp.Residual(network)

def ResNet(out_shape, n_filters=32, n_blocks=3):
    """
    Create the residual network.  The output is split and passed to the Affine Coupling layers.  We initialize
    the last convolution to be 0 so that the transformation ends up being the identity transformation initially.
    """
    _, _, channel = out_shape
    network = spp.sequential(spp.Conv(n_filters, filter_shape=(3, 3), padding=((1, 1), (1, 1)), bias=True, weightnorm=True),
                             spp.Relu(),
                             *[ResidualBlock(n_filters) for i in range(n_blocks)],
                             spp.Relu(),
                             spp.Conv(2*channel, filter_shape=(3, 3), padding=((1, 1), (1, 1)), bias=True, weightnorm=False,
                                      W_init=jax.nn.initializers.zeros, b_init=jax.nn.initializers.zeros))

    return spp.sequential(network, spp.Split(2, axis=-1), spp.parallel(spp.Tanh(), spp.Identity()))

def Transform(out_shape, n_hidden_layers=4, layer_size=1024):
    """
    Transformation for 1D affine coupling layers.
    """
    layer_sizes = [layer_size for _ in range(n_hidden_layers)]
    log_s_out = spp.sequential(spp.Dense(out_shape[-1]), spp.Tanh())
    t_out = spp.sequential(spp.Dense(out_shape[-1]))
    dense_layers = [spp.Dense(layer_size), spp.Relu()]*n_hidden_layers
    return spp.sequential(*dense_layers, spp.FanOut(2), spp.parallel(log_s_out, t_out))

def RealNVP(an_names):
    """
    Real NVP architecture
    """
    MAC = lambda tlz: MaskedAffineCoupling(ResNet, mask_type='checkerboard', top_left_zero=tlz)

    checker_transforms1 = sequential_flow(MAC(False),
                                          ActNorm(name=next(an_names)),
                                          MAC(True),
                                          ActNorm(name=next(an_names)),
                                          MAC(False),
                                          ActNorm(name=next(an_names)),
                                          MAC(True),
                                          ActNorm(name=next(an_names)))

    channel_transforms = sequential_flow(AffineCoupling(ResNet),
                                         Reverse(),
                                         ActNorm(name=next(an_names)),
                                         AffineCoupling(ResNet),
                                         Reverse(),
                                         ActNorm(name=next(an_names)),
                                         AffineCoupling(ResNet),
                                         ActNorm(name=next(an_names)))

    checker_transforms2 = sequential_flow(MAC(False),
                                          ActNorm(name=next(an_names)),
                                          MAC(True),
                                          ActNorm(name=next(an_names)),
                                          MAC(False),
                                          ActNorm(name=next(an_names)))

    real_nvp = sequential_flow(checker_transforms1,
                               Reverse(),
                               Squeeze(),
                               Reverse(),
                               channel_transforms,
                               Reverse(),
                               UnSqueeze(),
                               checker_transforms2)

    return real_nvp

def MultiScaleRealNVP(z_dim):
    """
    Multiscale Real NVP architecture with a low dimensional prior
    """
    an_names = iter(['an%d'%i for i in range(100)])

    def multi_scale(flow):
        return sequential_flow(RealNVP(an_names),
                               Squeeze(),
                               FactorOut(2),
                               factored_flow(flow, Identity()),
                               FanInConcat(2),
                               UnSqueeze())

    # Build the Multiscale Architecture
    flow = RealNVP(an_names)
    # flow = multi_scale(flow)
    # flow = multi_scale(flow)

    # Create a prior in a lower dimension
    prior = sequential_flow(AffineCoupling(Transform),
                            ActNorm(name=next(an_names)),
                            Reverse(),
                            AffineCoupling(Transform),
                            ActNorm(name=next(an_names)),
                            Reverse(),
                            AffineCoupling(Transform),
                            ActNorm(name=next(an_names)),
                            Reverse(),
                            AffineCoupling(Transform),
                            ActNorm(name=next(an_names)),
                            Reverse(),
                            AffineCoupling(Transform),
                            ActNorm(name=next(an_names)),
                            Reverse(),
                            AffineCoupling(Transform),
                            ActNorm(name=next(an_names)),
                            Reverse(),
                            AffineCoupling(Transform),
                            ActNorm(name=next(an_names)),
                            Reverse(),
                            AffineCoupling(Transform),
                            ActNorm(name=next(an_names)),
                            Reverse(),
                            AffineCoupling(Transform),
                            ActNorm(name=next(an_names)),
                            Reverse(),
                            UnitGaussianPrior())

    # Put the whole thing together
    flow = sequential_flow(Dequantization(scale=2**quantize_level_bits),
                           Logit(),
                           flow,
                           Reshape((np.prod(x_train.shape[1:]),)),
                           CoupledDimChange(Transform, prior, z_dim))

    flow = sequential_flow(Dequantization(scale=2**quantize_level_bits),
                           Logit(),
                           Reshape((np.prod(x_train.shape[1:]),)),
                           CoupledDimChange(Transform, prior, z_dim))


    return flow

def get_data():
    """
    Retrieve the CelebA dataset
    """
    quantize_level_bits = 3
    x_train = get_celeb_dataset(quantize_level_bits=quantize_level_bits, n_images=100000)
    return x_train

def initialize_model(x_train, z_dim=1000, batch_size=64, n_seed_examples=1000, notebook=False):
    """
    Initialize the model and seed it
    """
    flow = MultiScaleRealNVP(z_dim)
    init_fun, forward, inverse = flow

    # Intialize
    key = random.PRNGKey(0)
    names, output_shape, params, state = init_fun(key, x_train.shape[1:], ())
    model = (names, output_shape, params, state), forward, inverse

    # Data dependent init
    actnorm_names = [name for name in tree_flatten(names)[0] if 'an' in name]

    params = multistep_flow_data_dependent_init(x_train,
                                                actnorm_names,
                                                model,
                                                (),
                                                'actnorm_seed',
                                                key,
                                                n_seed_examples=n_seed_examples,
                                                batch_size=batch_size,
                                                notebook=notebook)
    return (names, output_shape, params, state), forward, inverse

@partial(jit, static_argnums=(0,))
def nll(forward, params, state, x, **kwargs):
    """
    Create the loss function
    """
    log_px, z, updated_state = forward(params, state, np.zeros(x.shape[0]), x, (), **kwargs)
    return -np.mean(log_px), updated_state

@partial(pmap, static_broadcasted_argnums=(0, 1, 2), axis_name='batch')
def spmd_update(forward, opt_update, get_params, i, opt_state, state, x_batch, key):
    """
    We will be using multiple gpus to increase the batch size
    """
    params = get_params(opt_state)
    (val, state), grads = jax.value_and_grad(partial(nll, forward), has_aux=True)(params, state, x_batch, key=key, test=TRAIN)
    g = jax.lax.psum(grads, 'batch')
    opt_state = opt_update(i, g, opt_state)
    return val, state, opt_state

def train(x_train, z_dim=100, batch_size=64, n_iters=20000, checkpoint_iters=5000, notebook=False, n_samples=16, n_samples_per_batch=16, results_folder='./realnvp_results'):
    """
    Train the model
    """

    # Make the results folder if it doesn't exist
    if(os.path.exists(results_folder) == False):
        os.makedirs(results_folder)

    # Initialize the model
    model = initialize_model(x_train, z_dim, notebook=notebook)
    (names, output_shape, params, state), forward, inverse = model
    jitted_inverse = jit(inverse)

    # Create the gradient function
    valgrad = jax.value_and_grad(partial(nll, forward), has_aux=True)
    valgrad = jit(valgrad)

    # Create the optimizer
    opt_init, opt_update, get_params = optimizers.adam(5e-4)
    opt_update = jit(opt_update)
    opt_state = opt_init(params)

    # Fill the update function with the optimizer params
    filled_spmd_update = partial(spmd_update, forward, opt_update, get_params)

    # Get set up to use multiple gpus
    n_gpus = xla_bridge.device_count()
    replicate_array = lambda x: onp.broadcast_to(x, (n_gpus,) + x.shape)
    replicated_opt_state = tree_map(replicate_array, opt_state)
    replicated_state = tree_map(replicate_array, state)

    key = random.PRNGKey(0)

    # Train
    losses = []
    batch_size = 2
    pbar = tnrange(n_iters) if notebook else tqdm(range(n_iters))
    for i in pbar:
        key, *keys = random.split(key, 3)

        # Take the next batch and make sure we've copied things correctly before its handed off to pmap
        batch_idx = random.randint(keys[0], (n_gpus, batch_size), minval=0, maxval=x_train.shape[0])
        x_batch = x_train[batch_idx,:]
        train_keys = np.array(random.split(keys[1], n_gpus))
        replicated_i = np.ones(n_gpus)*i

        # Take a gradient step
        replicated_val, replicated_state, replicated_opt_state = filled_spmd_update(replicated_i,
                                                                                    replicated_opt_state,
                                                                                    replicated_state,
                                                                                    x_batch,
                                                                                    train_keys)
        # Convert to bits/dimension
        val = replicated_val[0]
        val = val/np.prod(x_train.shape[1:])/np.log(2)
        losses.append(val)
        pbar.set_description('Negative Log Likelihood: %5.3f'%(val))

        # Checkpoint
        if(i%checkpoint_iters == 0):

            # Get the parameters and state
            state = tree_map(lambda x:x[0], replicated_state)
            opt_state = tree_map(lambda x:x[0], replicated_opt_state)
            params = get_params(opt_state)

            # Pull noise samples
            eval_key = random.PRNGKey(0)
            zs = random.normal(eval_key, (n_samples, z_dim))

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
                _, fz, _ = jitted_inverse(params, state, np.zeros(n_samples_per_batch), z, (), test=TEST)
                fz /= (2.0**quantize_level_bits) # Put the image (mostly) between 0 and 1
                fz *= (1.0 - 2*0.05)
                fz += 0.05

                for k in range(n_samples_per_batch):
                    next(axes_iter).imshow(fz[k])

            # Save the image samples
            samples_folder = os.path.join(results_folder, 'checkpoint')

            if(os.path.exists(samples_folder) == False):
                os.makedirs(samples_folder)

            iters_folder = os.path.join(samples_folder, 'iter_%d'%(i))
            if(os.path.exists(iters_folder) == False):
                os.makedirs(iters_folder)

            samples_path = os.path.join(iters_folder, 'sample')
            plt.savefig(samples_path)

            # Save the current losses
            loss_path = os.path.join(iters_folder, 'train_loss')
            fig, ax = plt.subplots(1, 1)
            ax.plot(losses)
            plt.savefig(loss_path)

            # Save the parameters
            save_model = (params, state)
            model_path = os.path.join(iters_folder, 'model')
            with open(model_path, 'wb') as f:
                pickle.dump(save_model, f)

    state = tree_map(lambda x:x[0], replicated_state)
    params = get_params(tree_map(lambda x:x[0], replicated_opt_state))

    return (names, output_shape, params, state), forward, inverse

def save_final_samples(model, z_dim, n_samples=64, n_samples_per_batch=4, results_folder='./realnvp_results'):
    """
    Save a bunch of final sampels
    """
    # Pull noise samples
    eval_key = random.PRNGKey(0)
    zs = random.normal(eval_key, (n_samples, z_dim))

    (names, output_shape, params, state), forward, inverse = model

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
            next(axes_iter).imshow(fz[k])

    # Save the image samples
    samples_folder = os.path.join(results_folder, 'samples')

    if(os.path.exists(samples_folder) == False):
        os.makedirs(samples_folder)

    samples_path = os.path.join(samples_folder, 'final_samples.png')
    plt.savefig(samples_path)

def save_reconstructions(x, model, n_samples=16, n_samples_per_batch=4, results_folder='./realnvp_results'):
    """
    Save reconstructions
    """
    eval_key = random.PRNGKey(0)

    (names, output_shape, params, state), forward, inverse = model

    # Create the axes
    n_cols = n_samples
    n_rows = 2

    # Make the subplots
    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(2*n_cols, 2*n_rows)

    for j in range(n_samples//n_samples_per_batch):
        _x = x_train[j*n_samples_per_batch:(j + 1)*n_samples_per_batch]

        log_px, finvx, _ = forward(params, state, np.zeros(n_samples_per_batch), _x, (), test=TEST, sigma=0.0)
        _, fz, _ = inverse(params, state, np.zeros(n_samples_per_batch), finvx, (), test=TEST)
        fz /= (2.0**quantize_level_bits)
        fz *= (1.0 - 2*0.05)
        fz += 0.05

        for i in range(n_samples_per_batch):
            axes[0,j*n_samples_per_batch + i].imshow(fz[i])
            axes[1,j*n_samples_per_batch + i].imshow(_x[i]/(2.0**quantize_level_bits))

    # Save the image samples
    reconstruction_folder = os.path.join(results_folder, 'samples')

    if(os.path.exists(reconstruction_folder) == False):
        os.makedirs(reconstruction_folder)

    reconstruction_path = os.path.join(reconstruction_folder, 'reconstructions.png')
    plt.savefig(reconstruction_path)

if(__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', help='Where to download data to', type=str)
    parser.add_argument('save_root', help='Where to save the results of the experiments to', type=str)
    parser.add_argument('z_dim', help='Latent state size', type=int)

    args = parser.parse_args()

    # Load the data
    quantize_level_bits = 3
    x_train = get_celeb_dataset(quantize_level_bits=quantize_level_bits, n_images=100)

    # Train
    trained_model = train(x_train, z_dim=args.z_dim, results_folder=args.save_root)

    # Save the final samples and reconstructions
    save_final_samples(trained_model, args.z_dim, n_samples=64, n_samples_per_batch=4, results_folder=args.save_root)

    # Save the reconstructions
    save_reconstructions(x_train, trained_model, n_samples=16, n_samples_per_batch=4, results_folder=args.save_root)
