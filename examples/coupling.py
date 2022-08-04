import jax
import jax.numpy as jnp
import nux
import nux.util as util
from jax import random
from functools import partial

class RealNVP():
  # This is implemented in affine_coupling.py
  def __init__(self,
               working_channel=16,
               hidden_channel=16,
               nonlinearity=util.square_swish,
               dropout_prob=0.2,
               n_layers=4):
    # Conditioner will be a resnet.  It will be initialized
    # when we know the shape of x.
    # This neural network does not have to come from NuX!
    self.make_coupling_net = lambda out_dim: nux.CouplingResNet(out_channel=out_dim,
                                                                working_channel=working_channel,
                                                                filter_shape=(3, 3),
                                                                hidden_channel=hidden_channel,
                                                                nonlinearity=nonlinearity,
                                                                dropout_prob=dropout_prob,
                                                                n_layers=n_layers)
    self.transformer = nux.ShiftScale()

  def get_params(self):
    return dict(scale=self.scale_params,
                conditioner=self.conditioner.get_params())

  def __call__(self, x, params=None, aux=None, rng_key=None, is_training=True, inverse=False, **kwargs):
    if params is None:
      # Initialize the parameters
      self.scale_params = random.normal(rng_key, ())*0.01
      self.conditioner_params = None
    else:
      # Retrieve the parameters
      self.scale_params = params["scale"]
      self.conditioner_params = params["conditioner"]

    k1, k2 = random.split(rng_key, 2)

    # Split the input
    split_dim = x.shape[-1]//2
    x1, x2 = x[...,:split_dim], x[...,split_dim:]

    # Apply the conditioner network
    dim = x1.shape[-1]
    self.conditioner = self.make_coupling_net(2*dim)
    theta = self.conditioner(x2, aux=aux, params=self.conditioner_params, rng_key=k1, is_training=is_training)

    # The initial parameters should be close to 0 for training to be stable.
    # Most of the layers in NuX will be close to the identity function when
    # given parameters with value 0.
    theta *= self.scale_params

    # Split the parameters for the transformer
    s, b = jnp.split(theta, 2, axis=-1)
    params = dict(s=s, b=b)

    # Apply the transformer to the input
    z1, log_det = self.transformer(x1, params=params, rng_key=k2, inverse=inverse, **kwargs)

    # Concatenate and return
    z = jnp.concatenate([z1, x2], axis=-1)
    return z, log_det

################################################################################################################

if __name__ == "__main__":
  # from debug import *
  rng_key = random.PRNGKey(0)

  # Generate some dummy data and a data iterator
  H, W, C = (32, 32, 3)
  x = random.normal(rng_key, (1000, H, W, C))
  class data_iterator():
    def __init__(self, x, rng_key, batch_size=16):
      self.x = x
      self.N = x.shape[0]
      self.batch_size = batch_size
      self.rng_key = rng_key

    def __iter__(self):
      return self

    def __next__(self):
      key, self.rng_key = random.split(self.rng_key, 2)
      indices = random.randint(key, minval=0, maxval=self.N, shape=(self.batch_size,))
      return self.x[indices]

  di = data_iterator(x, rng_key, batch_size=16)
  x_iter = iter(di)

  # Create the flow
  def make_realnvp():
    return RealNVP(working_channel=8,
                   hidden_channel=16,
                   nonlinearity=util.square_swish,
                   dropout_prob=0.0,
                   n_layers=2)
  glow = nux.Sequential([make_realnvp(),
                         nux.OneByOneConv(),
                         make_realnvp(),
                         nux.UnitGaussianPrior()])

  # Initialize with some data and get the parameters
  glow(next(x_iter), rng_key=rng_key)
  params = glow.get_params()

  # Create a loss function to train with
  def loss(params, x, rng_key):
    z, log_px = glow(x, params=params, rng_key=rng_key)
    return -log_px.mean()

  # Use gradient descent to train the flow
  valgrad = jax.value_and_grad(loss)
  valgrad = jax.jit(valgrad)

  n_train_iters = 10
  lr = 1e-4
  keys = random.split(rng_key, n_train_iters)
  for i, rng_key in enumerate(keys):
    nll, grad = valgrad(params, next(x_iter), rng_key)
    params = jax.tree_util.tree_map(lambda x, y: x - lr*y, params, grad)

  import pdb; pdb.set_trace()