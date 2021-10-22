from abc import ABC, abstractmethod
import jax
from jax import random
import jax.numpy as jnp
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable
from collections import OrderedDict
import nux.util as util
import einops

__all__ = ["Condition"]

class Condition(ABC):

  def __init__(self, make_coupling_net):
    self.make_coupling_net = make_coupling_net
    self.flow = self.make_flow()

  @property
  @abstractmethod
  def output_multiplier(self):
    pass

  @abstractmethod
  def make_flow(self):
    pass

  @abstractmethod
  def split_theta(self, theta, x_dim):
    pass

  def get_params(self):
    return dict(scale=self.scale_params,
                flow=self.flow.get_params(),
                conditioner=self.conditioner.get_params())

  def __call__(self, x, params=None, aux=None, rng_key=None, inverse=False, **kwargs):
    if params is None:
      self.scale_params = random.normal(rng_key, ())*0.01
      self.flow_params = None
      self.conditioner_params = None
    else:
      self.scale_params = params["scale"]
      self.flow_params = params["flow"]
      self.conditioner_params = params["conditioner"]

    k1, k2 = random.split(rng_key, 2)

    x_dim = x.shape[-1]
    self.conditioner = self.make_coupling_net(self.output_multiplier*x_dim)
    theta = self.conditioner(aux, params=self.conditioner_params, rng_key=k1)
    theta *= self.scale_params

    params = self.split_theta(theta, x_dim)

    return self.flow(x, params=params, rng_key=k2, inverse=inverse, **kwargs)

################################################################################################################

if __name__ == "__main__":
  from debug import *
  import nux
  from nux.tests.basic_unit_test import exact_test

  rng_key = random.PRNGKey(0)
  x, aux = random.normal(rng_key, shape=(2, 5, 4))

  class ConditionedNonlinearity(Condition):

    def make_flow(self):
      self.K = 8
      self.split_dim = 3*self.K
      flow = nux.Sequential([nux.LogisticCDFMixtureLogit(K=self.K), nux.ShiftScale()])
      self._out_multiplier = sum([x.param_multiplier for x in flow.layers])
      return flow

    @property
    def output_multiplier(self):
      return self._out_multiplier

    def split_theta(self, theta, x_dim):
      _theta, sb = theta[...,:self.split_dim*x_dim], theta[...,self.split_dim*x_dim:]
      s, b = jnp.split(sb, 2, axis=-1)
      params = [dict(theta=_theta), dict(s=s, b=b)]
      return params

  make_coupling_net = lambda out_dim: nux.CouplingResNet1D(out_dim,
                                                           working_dim=8,
                                                           hidden_dim=8,
                                                           nonlinearity=jax.nn.relu,
                                                           dropout_prob=0.0,
                                                           n_layers=2)
  flow = ConditionedNonlinearity(make_coupling_net=make_coupling_net)

  # Initialize the flow
  flow(x, aux=aux, rng_key=rng_key)
  params = flow.get_params()

  # Scramble the parameters to undo the data dependent init
  flat_params, unflatten = jax.flatten_util.ravel_pytree(params)
  flat_params = random.normal(rng_key, flat_params.shape)
  params = unflatten(flat_params)

  # Compute the log likelihood contribution of flow
  z, log_det = flow(x, params=params, aux=aux, rng_key=rng_key)

  # Reconstruct x
  x_reconstr, log_det2 = flow(z, params=params, aux=aux, rng_key=rng_key, inverse=True)
  assert jnp.allclose(x, x_reconstr)
  assert jnp.allclose(log_det, log_det2)

  # Compute the exact jacobian
  def unbatched_apply_fun(x, aux):
    z, _ = flow(x[None], params=params, aux=aux[None], rng_key=rng_key)
    return z[0]

  J = jax.vmap(jax.jacobian(unbatched_apply_fun))(x, aux)
  total_dim = util.list_prod(x.shape[1:])
  J_flat = J.reshape((-1, total_dim, total_dim))
  log_det_exact = jnp.linalg.slogdet(J_flat)[1]

  assert jnp.allclose(log_det_exact, log_det)
  print(f"{str(flow)} passed the reconstruction and log det test")

  import pdb; pdb.set_trace()