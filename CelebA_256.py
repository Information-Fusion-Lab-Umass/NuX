from jax import random, vmap, jit, value_and_grad
from jax.experimental import optimizers, stax
import jax.numpy as np
import staxplusplus as spp
from normalizing_flows import *
import jax.nn.initializers as jaxinit

def GLOWNet(out_shape, n_filters=512):
    _, _, channels = out_shape
    return spp.sequential(spp.Conv(n_filters, filter_shape=(3, 3), padding=((1, 1), (1, 1)), bias=True, weightnorm=False),
                          spp.Relu(),
                          spp.Conv(n_filters, filter_shape=(1, 1), padding=((0, 0), (0, 0)), bias=True, weightnorm=False),
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
    layers = [GLOWBlock(GLOWNet, masked=False, name=next(name_iter), additive_coupling=False)]*16
    return sequential_flow(Squeeze(), Debug(''), *layers, UnSqueeze())

def CelebA256(injective=True, quantize_level_bits=3):
    if(injective):
        z_dim = 256
    else:
        z_dim = None

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
    flow = multi_scale(flow)
    flow = multi_scale(flow)
    flow = multi_scale(flow)
    # flow = multi_scale(flow)
    # flow = multi_scale(flow)
    if(z_dim is not None):
        prior_layers = [AffineCoupling(FlatTransform), ActNorm(name=next(an_names)), Reverse()]*10
        prior_flow = sequential_flow(*prior_layers, AffineGaussianPriorDiagCov(z_dim))
        prior_flow = TallAffineDiagCov(prior_flow, z_dim)
    else:
        prior_flow = UnitGaussianPrior()
    flow = sequential_flow(Dequantization(scale=2**quantize_level_bits),
                           Logit(),
                           flow,
                           Flatten(),
                           prior_flow)
    return flow

# python glow_injective.py --name=CelebA128 --batchsize=8 --dataset=CelebA --numimage=-1 --quantize=3 --model=CelebADefault --startingit=0 --printevery=500