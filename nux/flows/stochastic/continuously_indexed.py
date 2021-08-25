import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
from typing import Optional, Mapping, Callable, Sequence

__all__ = ["ContinuouslyIndexed"]

class ContinuouslyIndexed():

  def __init__(self,
               flow: Optional[Callable]=None,
               p_ugz: Callable=None,
               q_ugx: Callable=None,
  ):
    """ Continuously indexed flow https://arxiv.org/pdf/1909.13833v3.pdf
        Main idea is that extra noise can significantly help form complicated
        marginal distributions that don't have the topological problems of
        bijective functions
    Args:
      flow        : The flow to use for the transform
      name        : Optional name for this module.
    """
    self.flow = flow
    self.p_ugz = p_ugz
    self.q_ugx = q_ugx

  def get_params(self):
    return {"p_ugz": self.p_ugz.get_params(),
            "q_ugx": self.q_ugx.get_params(),
            "flow": self.flow.get_params()}

  def __call__(self, x, params=None, inverse=False, rng_key=None, **kwargs):
    k1, k2, k3 = random.split(rng, 3)

    if params is None:
      self.q_params = None
      self.p_params = None
      self.flow_params = None
    else:
      self.q_params = params["p_ugz"]
      self.p_params = params["q_ugx"]
      self.flow_params = params["flow"]

    if inverse == False:

      u, log_qugx = self.q_ugx(jnp.zeros_like(x), aux=x, params=self.q_params, inverse=True, rng_key=k1)
      z, log_det = self.flow(x, aux=u, params=self.flow_params, inverse=False, rng_key=k2)
      _, log_pygx = self.p_ugz(u, aux=z, params=self.p_params, inverse=False, rng_key=k3)

      log_det += log_pugx - log_qugx

    else:
      u, log_pygx = self.p_ugz(jnp.zeros_like(x), aux=x, params=self.p_params, inverse=True, rng_key=k1)
      f_inputs = {"x": x, "condition": u}
      z, log_det = self.flow(x, aux=u, params=self.flow_params, inverse=True, rng_key=k2)

    return z, log_det