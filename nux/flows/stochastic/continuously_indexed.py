import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
from typing import Optional, Mapping, Callable, Sequence
from nux.flows.base import ZeroInitWrapper
from nux.nn.mlp import CouplingResNet1D
from nux.priors.gaussian import ParametrizedGaussianPrior, UnitGaussianPrior

__all__ = ["ContinuouslyIndexed"]

def make_feature_net1D(out_dim):
  net = CouplingResNet1D(out_dim,
                         working_dim=32,
                         hidden_dim=64,
                         nonlinearity=jax.nn.relu,
                         dropout_prob=0.0,
                         n_layers=8)
  return ZeroInitWrapper(net)

def make_conditioned_flow(create_network):
  return ParametrizedGaussianPrior(create_network)

class ContinuouslyIndexed():

  def __init__(self, flow, u_dist=None, v_dist=None, make_feature_net=None):
    """ Continuously indexed flow https://arxiv.org/pdf/1909.13833v3.pdf
        Main idea is that extra noise can significantly help form complicated
        marginal distributions that don't have the topological problems of
        bijective functions
    Args:
      flow        : The flow to use for the transform
      name        : Optional name for this module.
    """
    # p(u|z) and q(u|x) will share the same network and all of the feature networks will also be shared
    self.flow = flow
    self.make_feature_net = make_feature_net1D if make_feature_net is None else make_feature_net
    self.u_dist = UnitGaussianPrior() if u_dist is None else u_dist
    self.v_dist = make_conditioned_flow(self.make_feature_net) if v_dist is None else v_dist

  def get_params(self):
    return dict(p_ugz=self.p_ugz_params,
                q_ugx=self.q_ugx_params,
                f=self.flow.get_params(),
                f_feature=self.f_feature_net_params,
                q_feature=self.q_feature_net_params)

  def __call__(self, x, params=None, aux=None, inverse=False, rng_key=None, **kwargs):
    k1, k2, k3, k4, k5, k6 = random.split(rng_key, 6)

    if params is None:
      params = dict(p_ugz=None,
                    q_ugx=None,
                    f=None,
                    f_feature=None,
                    q_feature=None)

    out_dim = x.shape[-1]
    self.feature_net = self.make_feature_net(x.shape[-1])

    if inverse == False:
      # Sample u ~ q(u|ϕ(x))
      phi_x = self.feature_net(x, params=params["q_feature"], aux=None, rng_key=k1, **kwargs)
      self.q_feature_net_params = self.feature_net.get_params()
      u, log_qugx = self.v_dist(jnp.zeros_like(x), aux=phi_x, params=params["q_ugx"], inverse=True, rng_key=k2, **kwargs)
      self.q_ugx_params = self.v_dist.get_params()

      # Compute z = f(x;ϕ(u)) and p(x|u).
      phi_u = self.feature_net(u, params=params["f_feature"], aux=None, rng_key=k3, **kwargs)
      self.f_feature_net_params = self.feature_net.get_params()
      z, log_pxgu = self.flow(x, aux=phi_u, params=params["f"], inverse=False, rng_key=k4, **kwargs)

      # Compute p(u)
      _, log_pugz = self.u_dist(u, params=params["p_ugz"], inverse=False, rng_key=k6, **kwargs)
      self.p_ugz_params = self.u_dist.get_params()

    else:
      z = x # Rename

      # Sample u ~ p(u)
      u, log_pugz = self.u_dist(jnp.zeros_like(z), params=params["p_ugz"], inverse=True, rng_key=k6, **kwargs)
      self.p_ugz_params = self.u_dist.get_params()

      # Compute x = f^{-1}(z;u)
      phi_u = self.feature_net(u, params=params["f_feature"], aux=None, rng_key=k3, **kwargs)
      self.f_feature_net_params = self.feature_net.get_params()
      x, log_pxgu = self.flow(z, aux=phi_u, params=params["f"], inverse=True, rng_key=k4, **kwargs)

      # Predict q(u|x)
      phi_x = self.feature_net(x, params=params["q_feature"], aux=None, rng_key=k1, **kwargs)
      self.q_feature_net_params = self.feature_net.get_params()
      _, log_qugx = self.v_dist(u, aux=phi_x, params=params["q_ugx"], inverse=False, rng_key=k2, **kwargs)
      self.q_ugx_params = self.v_dist.get_params()

      # Rename again
      z = x

    llc = log_pxgu + log_pugz - log_qugx

    return z, llc

################################################################################################################

if __name__ == "__main__":
  from debug import *
  import nux

  rng_key = random.PRNGKey(0)
  x_shape = (4, 4, 2)
  x = random.normal(rng_key, shape=(2,) + x_shape)

  flow = nux.GLOWImage(n_layers=2,
                       working_channel=4,
                       hidden_channel=8,
                       nonlinearity=util.square_swish,
                       dropout_prob=0.0,
                       n_resnet_layers=1,
                       additive=False)

  def make_feature_net(out_dim):
    net = nux.CouplingResNet(out_dim,
                         working_channel=32,
                         hidden_channel=64,
                         filter_shape=(3, 3),
                         nonlinearity=jax.nn.relu,
                         dropout_prob=0.0,
                         n_layers=1)
    return nux.ZeroInitWrapper(net)

  flow = ContinuouslyIndexed(flow, make_feature_net=make_feature_net)

  z, log_det = flow(x, rng_key=rng_key)
  params = flow.get_params()

  x_reconstr, log_det2 = flow(z, params=params, rng_key=rng_key, inverse=True, reconstruction=True)
  import pdb; pdb.set_trace()
