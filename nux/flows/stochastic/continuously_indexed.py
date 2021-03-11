import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence
from nux.internal.layer import InvertibleLayer
import nux.util as util
import nux
import nux.networks as net

__all__ = ["ContinuouslyIndexed"]

class ContinuouslyIndexed(InvertibleLayer):

  def __init__(self,
               flow: Optional[Callable]=None,
               p_ugz: Callable=None,
               q_ugx: Callable=None,
               create_network: Optional[Callable]=None,
               network_kwargs: Optional=None,
               name: str="continuously_indexed"
  ):
    """ Continuously indexed flow https://arxiv.org/pdf/1909.13833v3.pdf
        Main idea is that extra noise can significantly help form complicated
        marginal distributions that don't have the topological problems of
        bijective functions
    Args:
      flow        : The flow to use for the transform
      name        : Optional name for this module.
    """
    super().__init__(name=name)
    self.flow = flow
    if p_ugz is not None:
      self._pugz = p_ugz

    if q_ugx is not None:
      self._q_ugx = q_ugx

    self.network_kwargs = network_kwargs
    self.create_network = create_network

  @property
  def p_ugz(self):
    if hasattr(self, "_pugz"):
      return self._pugz

    # Keep this simple!
    self._pugz = nux.ParametrizedGaussianPrior(network_kwargs=self.network_kwargs,
                                               create_network=self.create_network)
    return self._pugz

  @property
  def q_ugx(self):
    if hasattr(self, "_qugx"):
      return self._qugx

    # Keep this simple, but a bit more complicated than p(u|z).
    self._qugx = nux.sequential(nux.reverse_flow(nux.LogisticMixtureLogit(n_components=8,
                                                                          with_affine_coupling=False,
                                                                          coupling=False)),
                                nux.ParametrizedGaussianPrior(network_kwargs=self.network_kwargs,
                                                              create_network=self.create_network))
    return self._qugx

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    k1, k2, k3 = random.split(rng, 3)

    if sample == False:
      x = inputs["x"]

      q_inputs = {"x": jnp.zeros_like(x), "condition": x}
      q_outputs = self.q_ugx(q_inputs, k1, sample=True)
      u, log_qugx = q_outputs["x"], q_outputs["log_pz"] + q_outputs["log_det"]

      f_inputs = {"x": x, "condition": u}
      f_outputs = self.flow(f_inputs, k2, sample=False, **kwargs)
      z, log_det = f_outputs["x"], f_outputs["log_det"]

      p_inputs = {"x": u, "condition": z}
      p_outputs = self.p_ugz(p_inputs, k3, sample=False, **kwargs)
      log_pugx = p_outputs["log_pz"] + p_outputs.get("log_det", 0.0)

      log_det += log_pugx - log_qugx

      outputs = {"x": z,
                 "log_det": log_det}

    else:

      z = inputs["x"]

      p_inputs = {"x": jnp.zeros_like(z), "condition": z}
      p_outputs = self.p_ugz(p_inputs, k1, sample=True, **kwargs)
      u = p_outputs["x"]

      f_inputs = {"x": z, "condition": u}
      f_outputs = self.flow(f_inputs, k2, sample=True, **kwargs)
      x, log_det = f_outputs["x"], f_outputs["log_det"]

      outputs = {"x": x,
                 "log_det": log_det}

    return outputs
