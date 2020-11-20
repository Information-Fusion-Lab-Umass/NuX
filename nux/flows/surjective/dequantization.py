import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Callable
from nux.flows.base import *
import nux.util as util
import nux
from haiku._src.typing import PRNGKey

__all__ = ["UniformDequantization",
           "VariationalDequantization"]

class UniformDequantization(Layer):

  def __init__(self,
               scale: float,
               name: str="uniform_dequantization"
  ):
    """ Uniform dequantization.  See section 3.1 here https://arxiv.org/pdf/1511.01844.pdf
    Args:
      scale: This is usually the first layer of image pipelines, so for convenience also
             scale by the max value a pixel can take.
      name : Optional name for this module.
    """
    super().__init__(name=name)
    self.scale = scale

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           no_dequantization=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    x_shape = self.get_unbatched_shapes(sample)["x"]

    if sample == False:
      if no_dequantization == False:
        noise = random.uniform(rng, x.shape)
        z = (x + noise)/self.scale
      else:
        z = x/self.scale
    else:
      z = x*self.scale
      # z = jnp.floor(x*self.scale)

    log_det = -jnp.log(self.scale)*util.list_prod(x_shape)*jnp.ones(self.batch_shape)
    return {"x": z, "log_det": log_det}

################################################################################################################

class VariationalDequantization(Layer):
  def __init__(self,
               scale: float,
               flow: Optional[Callable]=None,
               network_kwargs: Optional=None,
               name: str="variational_dequantization"
  ):
    """ Variational dequantization https://arxiv.org/pdf/1902.00275.pdf
    Args:
      scale         : This is usually the first layer of image pipelines, so for convenience also
                      scale by the max value a pixel can take.
      flow          : The flow to use for dequantization
      network_kwargs: Dictionary with settings for the default network (see get_default_network in util.py)
      name          : Optional name for this module.
    """
    super().__init__(name=name)
    self.scale          = scale
    self.flow           = flow
    self.network_kwargs = network_kwargs

  def default_flow(self):
    return nux.sequential(nux.Logit(scale=None),
                          nux.OneByOneConv(),
                          nux.CouplingLogitsticMixtureLogit(n_components=8,
                                                            network_kwargs=self.network_kwargs,
                                                            reverse=True,
                                                            use_condition=True),
                          nux.OneByOneConv(),
                          nux.CouplingLogitsticMixtureLogit(n_components=8,
                                                            network_kwargs=self.network_kwargs,
                                                            reverse=True,
                                                            use_condition=True),
                          nux.UnitGaussianPrior())

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    x_shape = self.get_unbatched_shapes(sample)["x"]

    log_det = -jnp.log(self.scale)*util.list_prod(x_shape)*jnp.ones(self.batch_shape)
    flow = self.flow if self.flow is not None else self.default_flow()

    if sample == False:
      flow_inputs = {"x": jnp.zeros(x.shape), "condition": x}
      outputs = flow(flow_inputs, rng, sample=True)

      noise = outputs["x"]
      z = (x + noise)/self.scale

      log_qugx = outputs["log_det"] + outputs["log_pz"]
      log_det -= log_qugx
    else:
      z_continuous = x*self.scale
      z = jnp.floor(z_continuous).astype(jnp.int32)
      noise = z_continuous - z
      flow_inputs = {"x": noise, "condition": x}
      outputs = flow(flow_inputs, rng, sample=False)
      log_qugx = outputs["log_det"] + outputs["log_pz"]
      log_det -= log_qugx

    return {"x": z, "log_det": log_det}

