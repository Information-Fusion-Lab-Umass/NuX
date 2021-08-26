import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
from typing import Optional, Mapping, Callable
from nux.flows.bijective.nonlinearities import SquarePlus

__all__ = ["OnehotArgmaxFlow"]

class OnehotArgmaxFlow():
  """ Thresholding based argmax flows https://arxiv.org/pdf/2102.05379.pdf """
  def __init__(self,
               qugx: Callable,
               feature_network: Callable):
    self.qugx            = qugx
    self.feature_network = feature_network
    def _sp(x, inverse=False):
      z, _ = SquarePlus()(x, inverse=inverse)
      return z
    self.sp = _sp

  def get_params(self):
    return {"feature_network": self.feature_network.get_params(),
            "qugx": self.qugx.get_params()}

  def __call__(self, x, params=None, aux=None, inverse=False, rng_key=None, **kwargs):
    if params is None:
      self.f_params = None
      self.q_params = None
    else:
      self.f_params = params["feature_network"]
      self.q_params = params["qugx"]

    k1, k2 = random.split(rng_key, 2)
    sum_axes = util.last_axes(x.shape[1:])

    if inverse == False:
      # x is the one-hot vector.  Want to return a random vector whose argmax is x

      # Extract features to condition on
      f = self.feature_network(x, aux=None, params=self.f_params, rng_key=k1, **kwargs)

      # Sample an unconstrained vector
      flow_in = jnp.zeros(x.shape)
      u, log_qugx = self.qugx(flow_in, aux=f, params=self.q_params, inverse=True, rng_key=k2, **kwargs)

      # Apply thresholding.  This is elementwise so use autodiff for the logdet
      def threshold(u):
        T = (u*x).sum(axis=sum_axes, keepdims=True)
        v = jnp.where(x, u, T - self.sp(T - u))
        return v
      v, dvdu = jax.jvp(threshold, (u,), (jnp.ones_like(u),))
      log_dvdu = jnp.log(jnp.abs(dvdu)).sum(axis=sum_axes)
      return v, log_qugx - log_dvdu
    else:
      # The naming changes here
      v = x

      # Find the argmax
      def nd_onehot_argmax(v):
        flat_idx = v.ravel().argmax(axis=-1)
        full_idx = jnp.unravel_index(flat_idx, v.shape)
        x = jnp.zeros_like(v)
        x = x.at[full_idx].set(1.0)
        return x
      x = jax.vmap(nd_onehot_argmax)(v)

      # Undo thresholding.
      def threshold_inv(v):
        T = (v*x).sum(axis=sum_axes, keepdims=True)
        u = jnp.where(x, v, T - self.sp(T - v, inverse=True))
        return u
      u, dudv = jax.jvp(threshold_inv, (v,), (jnp.ones_like(v),))
      log_dvdu = -jnp.log(jnp.abs(dudv)).sum(axis=sum_axes)

      # Extract features to condition on
      f = self.feature_network(x, aux=None, params=self.f_params, rng_key=k1, **kwargs)

      # Evaluate the unconstrained vector
      eps, log_qugx = self.qugx(u, aux=f, params=self.q_params, inverse=False, rng_key=k2, **kwargs)
      return x, log_qugx - log_dvdu

################################################################################################################

if __name__ == "__main__":
  from debug import *
  import nux
  import matplotlib.pyplot as plt

  rng_key = random.PRNGKey(0)
  dim = 4
  batch_size = 3
  x = random.randint(rng_key, minval=0, maxval=dim, shape=(batch_size,))
  x = x[:,None] == jnp.arange(dim)
  x = x*1.0

  # flow = nux.GLOW(n_layers=2,
  #                 working_dim=4,
  #                 hidden_dim=8,
  #                 nonlinearity=util.square_swish,
  #                 dropout_prob=0.0,
  #                 n_resnet_layers=1,
  #                 additive=False)
  # flow = nux.Sequential([flow, nux.UnitGaussianPrior()])

  # feature_net = nux.ResNet1D(hidden_dim=8,
  #                            nonlinearity=util.square_swish,
  #                            dropout_prob=0.0,
  #                            n_layers=1)

  rng_key = random.PRNGKey(0)
  H, W, C = 2, 2, 2
  dim = H*W*C
  batch_size = 3
  x = random.randint(rng_key, minval=0, maxval=dim, shape=(batch_size,))
  x = x[:,None] == jnp.arange(dim)
  x = x*1.0
  x = x.reshape((batch_size, H, W, C))

  def make_flow():
    flow = nux.GLOWImage(n_layers=2,
                         working_channel=4,
                         hidden_channel=8,
                         nonlinearity=util.square_swish,
                         dropout_prob=0.0,
                         n_resnet_layers=1,
                         additive=False)
    return nux.Sequential([flow, nux.UnitGaussianPrior()])

  feature_net = nux.ResNet((3, 3),
                           hidden_channel=8,
                           nonlinearity=util.square_swish,
                           dropout_prob=0.0,
                           n_layers=1)

  # Initialize the argmax flow
  argmax_flow = OnehotArgmaxFlow(make_flow(), feature_net)

  # Get the initial parameters
  argmax_flow(x, rng_key=rng_key)
  params = argmax_flow.get_params()

  # Test reconstruction
  z, log_det = argmax_flow(x, params=params, rng_key=rng_key)
  reconstr, log_det2 = argmax_flow(z, params=params, rng_key=rng_key, inverse=True, reconstruction=True)

  import pdb; pdb.set_trace()

# import jax
# import jax.numpy as jnp
# import nux.util as util
# from jax import random
# from functools import partial
# from typing import Optional, Mapping, Callable
# from nux.flows.bijective.nonlinearities import SquarePlus

# __all__ = ["OnehotArgmaxFlow"]

# class OnehotArgmaxFlow():
#   """ Thresholding based argmax flows """
#   def __init__(self,
#                prior: Callable,
#                qugx: Callable,
#                feature_network: Callable):
#     self.prior           = prior
#     self.qugx            = qugx
#     self.feature_network = feature_network
#     def _sp(x, inverse=False):
#       z, _ = SquarePlus()(x, inverse=inverse)
#       return z
#     self.sp = _sp

#   def get_params(self):
#     return {"feature_network": self.feature_network.get_params(),
#             "qugx": self.qugx.get_params(),
#             "prior": self.prior.get_params()}

#   def __call__(self, x, params=None, aux=None, inverse=False, rng_key=None, **kwargs):
#     if params is None:
#       self.f_params = None
#       self.q_params = None
#       self.p_params = None
#     else:
#       self.f_params = params["feature_network"]
#       self.q_params = params["qugx"]
#       self.p_params = params["prior"]

#     k1, k2, k3 = random.split(rng_key, 3)
#     sum_axes = util.last_axes(x.shape[1:])

#     if inverse == False:
#       # x is the one-hot vector.  Want to return a random vector whose argmax is x

#       # Extract features to condition on
#       f = self.feature_network(x, aux=None, params=self.f_params, rng_key=k1, **kwargs)

#       # Sample an unconstrained vector
#       flow_in = jnp.zeros(x.shape)
#       u, log_qugx = self.qugx(flow_in, aux=f, params=self.q_params, inverse=True, rng_key=k2, **kwargs)

#       # Apply thresholding.  This is elementwise so use autodiff for the logdet
#       def threshold(u):
#         T = (u*x).sum(axis=sum_axes, keepdims=True)
#         v = jnp.where(x, u, T - self.sp(T - u))
#         return v
#       v, dvdu = jax.jvp(threshold, (u,), (jnp.ones_like(u),))
#       log_dvdu = jnp.log(jnp.abs(dvdu)).sum(axis=sum_axes)

#       # Evaluate the prior
#       _, log_pv = self.prior(v, params=self.p_params, inverse=False, rng_key=k3, **kwargs)

#       self.x = x
#       self.f = f
#       self.u = u
#       self.v = v
#       self.log_pv = log_pv
#       self.log_qugx = log_qugx
#       self.log_dvdu = log_dvdu

#       return v, log_pv - log_qugx + log_dvdu
#     else:
#       # Sample from the prior
#       v, log_pv = self.prior(x, params=self.p_params, inverse=True, rng_key=k3, **kwargs)

#       # Find the argmax
#       def nd_onehot_argmax(v):
#         flat_idx = v.ravel().argmax(axis=-1)
#         full_idx = jnp.unravel_index(flat_idx, v.shape)
#         x = jnp.zeros_like(v)
#         x = x.at[full_idx].set(1.0)
#         return x
#       x = jax.vmap(nd_onehot_argmax)(v)

#       # Undo thresholding.
#       def threshold_inv(v):
#         T = (v*x).sum(axis=sum_axes, keepdims=True)
#         u = jnp.where(x, v, T - self.sp(T - v, inverse=True))
#         return u
#       u, dudv = jax.jvp(threshold_inv, (v,), (jnp.ones_like(v),))
#       log_dvdu = -jnp.log(jnp.abs(dudv)).sum(axis=sum_axes)

#       # Extract features to condition on
#       f = self.feature_network(x, aux=None, params=self.f_params, rng_key=k1, **kwargs)

#       # Evaluate the unconstrained vector
#       eps, log_qugx = self.qugx(u, aux=f, params=self.q_params, inverse=False, rng_key=k2, **kwargs)
#       import pdb; pdb.set_trace()
#       return x, log_pv - log_qugx + log_dvdu

# ################################################################################################################

# if __name__ == "__main__":
#   from debug import *
#   import nux
#   import matplotlib.pyplot as plt

#   rng_key = random.PRNGKey(0)
#   dim = 4
#   batch_size = 3
#   x = random.randint(rng_key, minval=0, maxval=dim, shape=(batch_size,))
#   x = x[:,None] == jnp.arange(dim)
#   x = x*1.0

#   # flow = nux.GLOW(n_layers=2,
#   #                 working_dim=4,
#   #                 hidden_dim=8,
#   #                 nonlinearity=util.square_swish,
#   #                 dropout_prob=0.0,
#   #                 n_resnet_layers=1,
#   #                 additive=False)
#   # flow = nux.Sequential([flow, nux.UnitGaussianPrior()])

#   # feature_net = nux.ResNet1D(hidden_dim=8,
#   #                            nonlinearity=util.square_swish,
#   #                            dropout_prob=0.0,
#   #                            n_layers=1)

#   rng_key = random.PRNGKey(0)
#   H, W, C = 2, 2, 2
#   dim = H*W*C
#   batch_size = 3
#   x = random.randint(rng_key, minval=0, maxval=dim, shape=(batch_size,))
#   x = x[:,None] == jnp.arange(dim)
#   x = x*1.0
#   x = x.reshape((batch_size, H, W, C))

#   def make_flow():
#     flow = nux.GLOWImage(n_layers=2,
#                          working_channel=4,
#                          hidden_channel=8,
#                          nonlinearity=util.square_swish,
#                          dropout_prob=0.0,
#                          n_resnet_layers=1,
#                          additive=False)
#     return nux.Sequential([flow, nux.UnitGaussianPrior()])

#   feature_net = nux.ResNet((3, 3),
#                            hidden_channel=8,
#                            nonlinearity=util.square_swish,
#                            dropout_prob=0.0,
#                            n_layers=1)

#   # Initialize the argmax flow
#   argmax_flow = OnehotArgmaxFlow(make_flow(), make_flow(), feature_net)

#   # Get the initial parameters
#   argmax_flow(x, rng_key=rng_key)
#   params = argmax_flow.get_params()

#   # Test reconstruction
#   z, log_det = argmax_flow(x, params=params, rng_key=rng_key)
#   reconstr, log_det2 = argmax_flow(z, params=params, rng_key=rng_key, inverse=True, reconstruction=True)

#   import pdb; pdb.set_trace()