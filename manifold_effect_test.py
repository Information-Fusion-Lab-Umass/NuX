from tqdm import tqdm
from jax import random, vmap, jit, value_and_grad
from jax.experimental import optimizers, stax
import jax.numpy as np
import staxplusplus as spp
from normalizing_flows import *
import matplotlib.pyplot as plt
from datasets import *
import pandas as pd
import os
import pickle
import argparse

def create_network_head(train_data, noise_scale):
    """
    Handle dequantization and normalization
    """
    # Seed the first actnorm to normalize the data
    data_mean = np.mean(train_data, axis=0)
    data_std = np.std(train_data, axis=0)
    log_s_init = lambda key, shape: np.log(data_std)
    b_init = lambda key, shape: data_mean

    return sequential_flow(Dequantization(noise_scale=noise_scale, scale=1.0),
                           ActNorm(log_s_init=log_s_init, b_init=b_init))

def create_manifold_comparison_network(train_data, noise_scale, latent_dim, baseline=False):
    flow = create_network_head(train_data, noise_scale)

    top_flow = sequential_flow(flow,
                               MAF([1024]*5),
                               Reverse(),
                               MAF([1024]*5),
                               Reverse(),
                               MAF([1024]*5),
                               Reverse(),
                               MAF([1024]*5),
                               Reverse(),
                               MAF([1024]*5))

    if(latent_dim == None or latent_dim == 'baseline'):
        baseline = True

    if(baseline):
        flow = sequential_flow(top_flow, Affine(), UnitGaussianPrior())
    else:
        assert latent_dim is not None
        flow = sequential_flow(top_flow, AffineGaussianPriorFullCov(latent_dim))

    return flow

@partial(jit, static_argnums=(0,))
def neg_log_likelihood(forward, params, state, x, **kwargs):
    """
    Training objective
    """
    log_px, z, updated_state = forward(params, state, np.zeros(x.shape[0]), x, (), **kwargs)
    return -np.mean(log_px), updated_state

def experiment(x_train, x_test, noise_scale, data_name, key, latent_dim, baseline=False):
    """
    Run an experiment
    """
    # Create the flow
    flow = create_manifold_comparison_network(x_train, noise_scale, latent_dim=latent_dim, baseline=baseline)

    # Initialize it
    init_fun, forward, inverse = flow
    input_shape = (x_train.shape[-1],)
    names, output_shape, params, state = init_fun(key, input_shape, ())

    # Create the gradient function
    valgrad = jax.value_and_grad(partial(neg_log_likelihood, forward), has_aux=True)
    valgrad = jit(valgrad)

    # Create the optimizer
    opt_init, opt_update, get_params = optimizers.adam(0.0005)
    opt_update = jit(opt_update)
    opt_state = opt_init(params)

    # Train
    batch_size = 1024
    n_iters = 50000
    test_interval = 10
    train_nll = []
    test_nll = []
    pbar = tqdm(range(n_iters), leave=False)
    for i in pbar:
        key, *keys = random.split(key, 5)

        # Take batches for the train data
        batch_idx = random.randint(keys[0], (batch_size,), minval=0, maxval=x_train.shape[0])
        x_train_batch = x_train[batch_idx[0],:]

        params = get_params(opt_state)
        (nll, state), grad = valgrad(params, state, x_train_batch, key=keys[1])
        opt_state = opt_update(i, grad, opt_state)
        pbar.set_description('nll: %5.3f'%(nll))

        train_nll.append(nll)

        # Evaluate the test log likelihood
        if(i%test_interval == 0):
            params = get_params(opt_state)
            batch_idx = random.randint(keys[2], (batch_size,), minval=0, maxval=x_test.shape[0])
            x_test_batch = x_test[batch_idx,:]
            nll, _ = neg_log_likelihood(forward, params, state, x_test_batch, key=keys[3], test=TEST)
            test_nll.append(nll)

    return (flow, params, state), (np.array(train_nll), np.array(test_nll))

def train_iter_for_dataset(full_dim):
    """
    Iterate through the different latent state sizes we will use
    """
    z_dims = [int(p*full_dim) for p in [0.25, 0.5, 1.0]]
    for i, latent_dim in enumerate(z_dims):
        yield i, latent_dim
    yield i + 1, None

def run_experiment(dataset_names=['hepmass', 'gas', 'miniboone', 'power'], data_root='/tmp/', save_root='./manifold_effect_results/'):
    """
    Run and save the experiment
    """
    # Run the experiments
    key = random.PRNGKey(0)
    results = {}
    for train_data, test_data, noise_scale, dataset_name in uci_loader(datasets=dataset_names, data_root=data_root):
        for i, latent_dim in train_iter_for_dataset(train_data.shape[1]):
            (_, params, state), (train_nll, test_nll) = experiment(train_data, test_data, noise_scale, dataset_name, key, latent_dim, baseline=(latent_dim is None))
            save_name = '%s_%s'%(dataset_name, str(latent_dim))
            results[save_name] = (params, state, train_nll, test_nll)

    # Save the results
    for save_name, val in results.items():
        save_path = os.path.join(save_root, '%s.p'%(save_name))
        if(os.path.exists(save_path) == False):

            # If the folder doesn't exist, make it
            if(os.path.exists(save_root) == False):
                os.makedirs(save_root)

            # Save using pickle
            with open(save_path, 'wb') as f:
                pickle.dump(val, f)

def retrieve_results(results_folder='./manifold_results/', dataset_names=['hepmass', 'gas', 'miniboone', 'power']):
    """
    Retrieve the pickled results
    """
    results = {}
    for dataset_name in dataset_names:
        results_files = glob.glob('%s/*%s*'%(results_folder, dataset_name))
        for file_name in results_files:
            test_name =  os.path.basename(file_name)[:-2]
            with open(file_name, 'rb') as f:
                res = pickle.load(f)
                results[test_name] = res
    return results

def evaluate_results(results, save_folder='./manifold_results/'):
    """
    Plot the results
    """
    names = ['hepmass', 'gas', 'miniboone', 'power']

    fig, axes = plt.subplots(2, 4)
    fig.set_size_inches(8*4, 8*2)

    for run_str in results.keys():
        name, latent_dim = run_str.split('_')
        _, _, train_nll, test_nll = results[run_str]
        suffix_to_plot = 1000

        # Plot the smoothed train results
        smoothed_test_nll = pd.Series(train_nll).ewm(alpha=0.01).mean()
        axes[0,names.index(name)].plot(smoothed_test_nll[-(suffix_to_plot*10):], label='%s'%run_str)
        axes[0,names.index(name)].set_title('%s train set negative log likelihoods'%(name))
        axes[0,names.index(name)].legend()

        # Plot the smoothed test results
        smoothed_test_nll = pd.Series(test_nll).ewm(alpha=0.01).mean()
        axes[1,names.index(name)].plot(smoothed_test_nll[-suffix_to_plot:], label='%s'%run_str)
        axes[1,names.index(name)].set_title('%s test set negative log likelihoods'%(name))
        axes[1,names.index(name)].legend()


    plt.savefig(os.path.join(save_folder, 'full_results.png'))

parser = argparse.ArgumentParser()
parser.add_argument('data_root', help='Where to download data to', type=str)
parser.add_argument('save_root', help='Where to save the results of the experiments to', type=str)
parser.add_argument('--plot_results', help='Plot the results', action='store_true')
parser.add_argument('--hepmass', help='Use the hepmass dataset', action='store_true')
parser.add_argument('--gas', help='Use the gas dataset', action='store_true')
parser.add_argument('--miniboone', help='Use the miniboone dataset', action='store_true')
parser.add_argument('--power', help='Use the power dataset', action='store_true')

args = parser.parse_args()

if(args.plot_results):
    results = retrieve_results(results_folder=args.save_root)
    evaluate_results(results, save_folder=args.save_root)
else:
    dataset_names = []
    if(args.hepmass):
        dataset_names.append('hepmass')
    if(args.gas):
        dataset_names.append('gas')
    if(args.miniboone):
        dataset_names.append('miniboone')
    if(args.power):
        dataset_names.append('power')

    run_experiment(dataset_names, data_root=args.data_root, save_root=args.save_root)


# ./data ./manifold_results --hepmass --gas --miniboone --power
# ./data ./manifold_results --plot_results
