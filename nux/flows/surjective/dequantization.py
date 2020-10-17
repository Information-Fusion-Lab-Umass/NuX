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

class UniformDequantization(AutoBatchedLayer):

  def __init__(self, scale: float, name: str="uniform_dequantization", **kwargs):
    super().__init__(name=name, **kwargs)
    self.scale = scale

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           no_dequantization=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    x = inputs["x"]

    if(sample == False):
      if(no_dequantization == False):
        noise = random.uniform(rng, x.shape)
        z = (x + noise)/self.scale
      else:
        z = x/self.scale
    else:
      z = x*self.scale
      # z = jnp.floor(x*self.scale)

    log_det = -jnp.log(self.scale)*jnp.prod(jnp.array(x.shape))
    return {"x": z, "log_det": log_det}

################################################################################################################

class VariationalDequantization(AutoBatchedLayer):
  # This takes up a ton of memory!

  def __init__(self,
               scale: float,
               flow: Optional[Callable]=None,
               name: str="variational_dequantization",
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.scale = scale
    self.flow  = flow

  def default_flow(self):
    return nux.sequential(nux.Logit(),
                          nux.ActNorm(),
                          nux.OneByOneConv(),
                          nux.ConditionedCoupling(),
                          nux.ActNorm(),
                          nux.OneByOneConv(),
                          nux.ConditionedCoupling(),
                          nux.ActNorm(),
                          nux.OneByOneConv(),
                          nux.ConditionedCoupling(),
                          nux.UnitGaussianPrior())

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    x = inputs["x"]
    log_det = -jnp.log(self.scale)*jnp.prod(jnp.array(x.shape))

    if sample == False:
      flow = self.flow if self.flow is not None else self.default_flow()
      flow_inputs = {"x": jnp.zeros(x.shape), "condition": x}
      import pdb; pdb.set_trace()
      outputs = flow(flow_inputs, rng, sample=True, no_batching=True)

      noise = outputs["x"]
      z = (x + noise)/self.scale

      log_qugx = outputs["log_det"] + outputs["log_pz"]
      log_det -= log_qugx
    else:
      z = jnp.floor(x*self.scale).astype(jnp.int32)

    return {"x": z, "log_det": log_det}

