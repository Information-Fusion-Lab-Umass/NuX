# from jax.config import config
# config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
import nux.util as util
from nux.nn.mlp import CouplingResNet1D, ResNet1D
from nux.nn.resnet import CouplingResNet
from nux.flows.bijective.affine import ShiftScale, PLUMVP
from nux.flows.bijective.spline import RationalQuadraticSpline
from nux.flows.bijective.logistic_cdf_mixture_logit import LogisticCDFMixtureLogit
from nux.flows.bijective.conv import OneByOneConv
from nux.flows.base import Sequential, Repeat
from abc import ABC, abstractmethod
import einops

__all__ = ["NonlinearCoupling",
           "NonlinearCouplingImage"]

class _coupling(ABC):

  def __init__(self, K=8, kind="spline", with_affine_coupling=True, **kwargs):
    layers = []
    if kind == "spline":
      layers.append(RationalQuadraticSpline(K=K, **kwargs))
      self.split = 3*K - 1
    else:
      layers.append(LogisticCDFMixtureLogit(K=K, **kwargs))
      self.split = 3*K

    self.mul = self.split

    self.with_affine_coupling = with_affine_coupling
    if with_affine_coupling:
      layers.append(ShiftScale())
      self.mul += 2

    self.transformer = Sequential(layers)

    self._check_aux = kwargs.get("_check_aux", True)

  def get_params(self):
    return dict(scale=self.scale_params,
                conditioner=self.conditioner.get_params())

  def __call__(self, x, params=None, aux=None, rng_key=None, is_training=True, inverse=False, **kwargs):
    if params is None:
      self.scale_params = random.normal(rng_key, ())*0.01
      self.conditioner_params = None
    else:
      self.scale_params = params["scale"]
      self.conditioner_params = params["conditioner"]

    k1, k2 = random.split(rng_key, 2)

    # Split the input
    split_dim = x.shape[-1]//2
    x1, x2 = x[...,:split_dim], x[...,split_dim:]

    # The auxiliary input and x1 must have the same spatial dimensions
    if aux is not None and self._check_aux:
      x_spatial_shape, aux_spatial_shape = x.shape[1:-1], aux.shape[1:-1]
      if x_spatial_shape != aux_spatial_shape:
        assert len(x_spatial_shape) == 2
        aux = einops.reduce(aux, "... (h h1) (w w1) c -> ... h w c", h=x_spatial_shape[0], w=x_spatial_shape[1], reduction="mean")

    # Apply the conditioner network
    dim = x1.shape[-1]
    self.conditioner = self.make_coupling_net(self.mul*dim)
    theta = self.conditioner(x2, aux=aux, params=self.conditioner_params, rng_key=k1, is_training=is_training, **kwargs)
    theta *= self.scale_params

    # Split the parameters for the transformer
    if self.with_affine_coupling == False:
      params = [dict(theta=theta)]
    else:
      _theta, sb = theta[...,:self.split*dim], theta[...,self.split*dim:]
      s, b = jnp.split(sb, 2, axis=-1)
      params = [dict(theta=_theta), dict(s=s, b=b)]

    # Apply the transformer to the input
    z1, log_det = self.transformer(x1, params=params, rng_key=k2, inverse=inverse, **kwargs)

    # Concatenate and return
    z = jnp.concatenate([z1, x2], axis=-1)
    return z, log_det

class Block(_coupling):

  def __init__(self,
               working_dim=16,
               hidden_dim=16,
               nonlinearity=util.square_swish,
               dropout_prob=0.2,
               n_layers=4,
               K=8,
               kind="spline",
               with_affine_coupling=True,
               make_coupling_net=None,
               **kwargs):
    super().__init__(K=K, kind=kind, with_affine_coupling=with_affine_coupling, **kwargs)
    if make_coupling_net is None:
      self.make_coupling_net = lambda out_dim: CouplingResNet1D(out_dim,
                                                                working_dim,
                                                                hidden_dim,
                                                                nonlinearity,
                                                                dropout_prob,
                                                                n_layers,
                                                                **kwargs)
    else:
      self.make_coupling_net = make_coupling_net

class ImageBlock(_coupling):

  def __init__(self,
               working_channel=16,
               hidden_channel=16,
               nonlinearity=util.square_swish,
               dropout_prob=0.2,
               n_layers=4,
               K=8,
               kind="spline",
               with_affine_coupling=True,
               make_coupling_net=None,
               **kwargs):
    super().__init__(K=K, kind=kind, with_affine_coupling=with_affine_coupling, **kwargs)
    if make_coupling_net is None:
      self.make_coupling_net = lambda out_dim: CouplingResNet(out_dim,
                                                              working_channel,
                                                              (3, 3),
                                                              hidden_channel,
                                                              nonlinearity,
                                                              dropout_prob,
                                                              n_layers,
                                                              **kwargs)
    else:
      self.make_coupling_net = make_coupling_net

################################################################################################################

class NonlinearCoupling(Repeat):

  def __init__(self,
               n_layers=4,
               working_dim=16,
               hidden_dim=16,
               nonlinearity=util.square_swish,
               dropout_prob=0.2,
               n_resnet_layers=4,
               K=8,
               kind="spline",
               with_affine_coupling=False,
               **kwargs):
    coupling_layer = Block(working_dim=working_dim,
                           hidden_dim=hidden_dim,
                           nonlinearity=nonlinearity,
                           dropout_prob=dropout_prob,
                           n_layers=n_resnet_layers,
                           K=K,
                           kind=kind,
                           with_affine_coupling=with_affine_coupling,
                           **kwargs)
    layers = []
    layers.append(coupling_layer)
    layers.append(PLUMVP())
    layers.append(ShiftScale())
    self.flow = Sequential(layers)
    super().__init__(flow=self.flow, n_repeats=n_layers, checkerboard=False, **kwargs)

class NonlinearCouplingImage(Repeat):

  def __init__(self,
               n_layers=4,
               working_channel=16,
               hidden_channel=16,
               nonlinearity=util.square_swish,
               dropout_prob=0.2,
               n_resnet_layers=4,
               K=8,
               kind="spline",
               with_affine_coupling=True,
               checkerboard=True,
               **kwargs):
    coupling_layer = ImageBlock(working_channel=working_channel,
                                hidden_channel=hidden_channel,
                                nonlinearity=nonlinearity,
                                dropout_prob=dropout_prob,
                                n_layers=n_resnet_layers,
                                K=K,
                                kind=kind,
                                with_affine_coupling=with_affine_coupling,
                                **kwargs)
    layers = []
    layers.append(coupling_layer)
    layers.append(OneByOneConv())
    layers.append(ShiftScale())
    self.flow = Sequential(layers)
    super().__init__(flow=self.flow, n_repeats=n_layers, checkerboard=checkerboard, **kwargs)

################################################################################################################

def regular_test():
  from jax.flatten_util import ravel_pytree

  rng_key = random.PRNGKey(0)
  x = random.normal(rng_key, shape=(2, 2))
  flow = NonlinearCoupling(n_layers=10,  dropout_prob=0.0, kind="logistic_mixture", K=3, with_affine_coupling=False, n_resnet_layers=1)
  z, log_det = flow(x, rng_key=rng_key)
  params = flow.get_params()

  zz, _ = flow(x[:1], params=params, rng_key=rng_key)
  assert jnp.allclose(zz, z[:1])

  reconstr, _ = flow(z[:1], params=params, rng_key=rng_key, inverse=True)
  assert jnp.allclose(x[:1], reconstr)

  z2, _ = flow(reconstr[:1], params=params, rng_key=rng_key, inverse=False)
  assert jnp.allclose(z[:1], z2)

  def jac(x, blah=False):
    flat_x, unflatten = ravel_pytree(x)
    def flat_call(flat_x):
      x = unflatten(flat_x)
      z, _ = flow(x[None], params=params, rng_key=rng_key)
      return z.ravel()
    z = flat_call(flat_x)
    if blah:
      return z
    return jax.jacobian(flat_call)(flat_x)

  jac(x[0], blah=True)
  J = jax.vmap(jac)(x)
  true_log_det = jnp.linalg.slogdet(J)[1]
  assert jnp.allclose(log_det, true_log_det)

if __name__ == "__main__":
  from debug import *

  regular_test()
