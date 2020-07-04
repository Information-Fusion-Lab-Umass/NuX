import src.flows as flows
from src.tests.nf_test import standard_layer_tests, image_layer_test, unit_test, flow_test
from src.tests.nif_test import nif_test
import jax.numpy as jnp
from debug import *

import src.flows as nux
import jax
from jax import jit, vmap, random
import haiku as hk
import tensorflow_datasets as tfds
import tensorflow as tf
from functools import partial
import src.util as util

################################################################################################################

class ConvBlock(hk.Module):

    # nux.Coupling expects an uninitialized Haiku network that accepts an output_shape
    def __init__(self, out_shape, n_hidden_channels=256, name=None):
        _, _, out_channels = out_shape
        super().__init__(name=name)
        self.out_channels = out_channels
        self.n_hidden_channels = n_hidden_channels

        self.last_channels = 2*out_channels

    def __call__(self, x, **kwargs):
        H, W, C = x.shape

        x = hk.Conv2D(output_channels=self.n_hidden_channels, kernel_shape=(3, 3), stride=(1, 1))(x[None])[0]
        x = jax.nn.relu(x)
        x = hk.Conv2D(output_channels=self.n_hidden_channels, kernel_shape=(1, 1), stride=(1, 1))(x[None])[0]
        x = jax.nn.relu(x)
        x = hk.Conv2D(output_channels=self.last_channels, kernel_shape=(3, 3), stride=(1, 1))(x[None])[0]

        mu, alpha = jnp.split(x, 2, axis=-1)
        alpha = jnp.tanh(alpha)
        return mu, alpha

def GLOWBlock(name):
    return nux.sequential(nux.ActNorm(name='an_%s'%name),
                          nux.OnebyOneConvLAX(),
                          nux.Coupling(ConvBlock))

def GLOW(i, num_blocks=1):
    base_idx = i*num_blocks
    layers = [GLOWBlock('glow_%d'%(base_idx + j)) for j in range(num_blocks)]
    # return nux.Identity()
    return nux.sequential(*layers)

def MultiscaleGLOW(quantize_bits=3):

    flow = nux.Identity()
    flow = nux.multi_scale(GLOW(0), flow)
#     flow = nux.multi_scale(GLOW(1), flow)
#     flow = nux.multi_scale(GLOW(2), flow)
#     flow = nux.multi_scale(GLOW(3), flow)
#     flow = nux.multi_scale(GLOW(4), flow)

    flow = nux.sequential(nux.UniformDequantization(scale=2**quantize_bits),
                          nux.Logit(),
                          nux.Squeeze(), # So that the channel is divisible by 2
                          flow,
                          nux.Flatten())#,
                          # nux.UnitGaussianPrior())
                          # nux.AffineGaussianPriorDiagCov(128))
    return flow

################################################################################################################

def random_crop(x):
#     x['image'] = tf.image.random_crop(x['image'][::4,::4], size=[32, 32, 3])
    x['image'] = x['image'][::2,::2][26:-19,12:-13]
    return x

def quantize(x, quantize_bits):
    quantize_factor = 256/(2**quantize_bits)
    x['image'] = x['image']//tf.cast(quantize_factor, dtype=tf.uint8)
    return x

################################################################################################################

def load_dataset(split='train', is_training=True, batch_size=32, quantize_bits=3):
    ds = tfds.load('celeb_a', split=split).repeat()
    ds = ds.map(random_crop)
    ds = ds.map(partial(quantize, quantize_bits=quantize_bits))
    if is_training:
        ds = ds.shuffle(10*batch_size, seed=0)
    ds = ds.batch(batch_size)
    return tfds.as_numpy(ds)

################################################################################################################

if(__name__ == '__main__'):

    # ds = load_dataset(quantize_bits=3)

    # x = next(ds)['image']*1.0

    # key = random.PRNGKey(0)
    # x = jnp.arange(16).reshape((8, 2))
    # inputs = {'x': x}

    # init_fun = nux.Reverse()
    # outputs, flow = init_fun(key, inputs, batched=True)

    # assert 0

    # key = random.PRNGKey(0)
    # quantize_bits = 3
    # init_fun = MultiscaleGLOW()

    # init_fun(key, inputs, batched=True)

    # flow_test(MultiscaleGLOW(), {'x': x[0]}, key)

    standard_layer_tests()
    image_layer_test()
    unit_test()

    nif_test()