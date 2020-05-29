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
from util import TRAIN, TEST
clip_grads = jit(optimizers.clip_grads)

n_gpus = xla_bridge.device_count()
print('n_gpus:', n_gpus)


data_loader = cifar10_reduced_data_loader()
test_nif_embeddings = np.load(os.path.join('Experiments/CIFAR512/100000', 'test_nif_embeddings.npy'))
test_nf_embeddings = np.load(os.path.join('Experiments/CIFAR512/100000', 'test_nf_embeddings.npy'))
test_y = np.load(os.path.join('Experiments/CIFAR512/100000', 'test_y.npy'))
training_nif_embeddings = np.load(os.path.join('Experiments/CIFAR512/100000', 'training_nif_embeddings.npy'))
training_nf_embeddings = np.load(os.path.join('Experiments/CIFAR512/100000', 'training_nf_embeddings.npy'))
training_y = np.load(os.path.join('Experiments/CIFAR512/100000', 'training_y.npy'))

test_y = (test_y == np.arange(10)[:,None]).astype(np.float32).T
training_y = (training_y == np.arange(10)[:,None]).astype(np.float32).T


init_random_params_nf, predict_nf = spp.sequential(*([spp.Dense(512, keep_prob=0.7), spp.Relu(), spp.BatchNorm()]*4), spp.Dense(10, keep_prob=0.7), spp.Softmax())
init_random_params_nif, predict_nif = spp.sequential(*([spp.Dense(512, keep_prob=0.7), spp.Relu(), spp.BatchNorm()]*4), spp.Dense(10, keep_prob=0.7), spp.Softmax())



opt_init_nf, opt_update_nf, get_params_nf = optimizers.adam(1e-3)
opt_init_nif, opt_update_nif, get_params_nif = optimizers.adam(1e-3)

@jit
def loss_nf(params, state, batch, key):
    inputs, targets = batch
    preds, updated_state = predict_nf(params, state, inputs, key=key, test=TRAIN)
    return -np.mean(np.sum(preds * targets, axis=1)), updated_state

@jit
def loss_nif(params, state, batch, key):
    inputs, targets = batch
    preds, updated_state = predict_nif(params, state, inputs, key=key, test=TRAIN)
    return -np.mean(np.sum(preds * targets, axis=1)), updated_state


@jit
def update_nf(i, opt_state, state, batch, key):
    params = get_params_nf(opt_state)
    g, updated_state = grad(loss_nf, has_aux=True)(params, state, batch, key)
    return opt_update_nf(i, g, opt_state), updated_state


@jit
def update_nif(i, opt_state, state, batch, key):
    params = get_params_nif(opt_state)
    g, updated_state = grad(loss_nif, has_aux=True)(params, state, batch, key)
    return opt_update_nif(i, g, opt_state), updated_state



@jit
def accuracy_nf(params, state, batch):
    inputs, targets = batch
    target_class = np.argmax(targets, axis=1)
    preds, updated_state = predict_nf(params, state, inputs, test=TEST)
    predicted_class = np.argmax(preds, axis=1)
    return np.mean(predicted_class == target_class)

@jit
def accuracy_nif(params, state, batch):
    inputs, targets = batch
    target_class = np.argmax(targets, axis=1)
    preds, updated_state = predict_nif(params, state, inputs, test=TEST)
    predicted_class = np.argmax(preds, axis=1)
    return np.mean(predicted_class == target_class)

key = random.PRNGKey(0)
_,_, init_params_nf, state_nf = init_random_params_nf(key, (3072,))
_,_, init_params_nif, state_nif = init_random_params_nif(key, (512,))

opt_state_nf = opt_init_nf(init_params_nf)
opt_state_nif = opt_init_nif(init_params_nif)


pbar = tqdm(np.arange(0, 500000))
for i in pbar:
    key, *keys = random.split(key, 3)

    nif_batch, nf_batch, y_batch  = data_loader((32,), keys[0], None, True, True)

    y_batch = (y_batch == np.arange(10)[:,None]).astype(np.float32).T


    opt_state_nf, state_nf = update_nf(i, opt_state_nf, state_nf, (nf_batch, y_batch), keys[1])
    opt_state_nif, state_nif= update_nif(i, opt_state_nif, state_nif, (nif_batch, y_batch), keys[1])


    if(i % 1000 == 1):
        params_nf = get_params_nf(opt_state_nf)
        params_nif = get_params_nif(opt_state_nif)
        train_acc_nf = accuracy_nf(params_nf, state_nf, (training_nf_embeddings, training_y))
        train_acc_nif = accuracy_nif(params_nif, state_nif, (training_nif_embeddings, training_y))
        test_acc_nf = accuracy_nf(params_nf, state_nf, (test_nf_embeddings, test_y))
        test_acc_nif = accuracy_nif(params_nif, state_nif, (test_nif_embeddings, test_y))

        print(f"Training set accuracy nf: {train_acc_nf} nif: {train_acc_nif} ")
        print(f"Test set accuracy nf: {test_acc_nf} nif: {test_acc_nif}")















