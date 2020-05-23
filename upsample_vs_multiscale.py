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

def GLOW(name_iter, norm_type='instance', conditioned_actnorm=False):
    layers = [GLOWBlock(GLOWNet, masked=False, name=next(name_iter), additive_coupling=False)]*16
    return sequential_flow(Squeeze(), Debug(''), *layers, UnSqueeze())

def CelebAUpscale(injective=True, quantize_level_bits=3):
    if(injective):
        z_dim = 128
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

    if(z_dim is not None):
        flow = sequential_flow(Dequantization(scale=2**quantize_level_bits),
                               Logit(),
                               Squeeze(),
                               CoupledUpSample(GLOWNet, (2, 2)),
                               GLOW(name_iter),
                               Debug(''),
                               CoupledUpSample(GLOWNet, (2, 2)),
                               GLOW(name_iter),
                               Debug(''),
                               CoupledUpSample(GLOWNet, (2, 2)),
                               GLOW(name_iter),
                               Debug(''),
                               Flatten(),
                               UnitGaussianPrior())
    else:
        flow = GLOW(name_iter)
        flow = multi_scale(flow)
        flow = multi_scale(flow)
        flow = multi_scale(flow)
        flow = sequential_flow(Dequantization(scale=2**quantize_level_bits),
                               Logit(),
                               flow,
                               Flatten(),
                               UnitGaussianPrior())
    return flow

# python glow_injective.py --name=UpSampleTest --batchsize=8 --dataset=CelebA --numimage=-1 --quantize=3 --model=CelebAUpsample --startingit=0 --printevery=2000