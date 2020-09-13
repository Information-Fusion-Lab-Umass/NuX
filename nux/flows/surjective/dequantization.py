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

class VariationalDequantization(Layer):

  def __init__(self, scale: float, flow: Optional[Callable]=None, name: str="variational_dequantization", **kwargs):
    super().__init__(name=name, **kwargs)
    self.scale = scale
    self.flow  = flow

  def default_flow(self):
      return nux.sequential(nux.Logit(),
                            nux.OneByOneConv(),
                            nux.ConditionedCoupling(),
                            nux.OneByOneConv(),
                            nux.ConditionedCoupling(),
                            nux.OneByOneConv(),
                            nux.ConditionedCoupling(),
                            nux.UnitGaussianPrior())

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    x = inputs["x"]
    log_det = -jnp.log(self.scale)*jnp.prod(jnp.array(self.expected_shapes["x"]))

    # We don't want to auto batch this function because the rest of the flow might be autobatched
    expected_ndim = len(self.expected_shapes["x"])
    if expected_ndim < x.ndim:
      batch_dimensions = x.shape[:x.ndim - expected_ndim]
      log_det = jnp.ones(batch_dimensions)*log_det

    if sample == False:
      flow = self.flow if self.flow is not None else self.default_flow()
      flow_inputs = {"x": jnp.zeros(x.shape), "condition": x}
      outputs = flow(flow_inputs, sample=True)

      noise = outputs["x"]
      z = (x + noise)/self.scale

      log_qugx = outputs["log_det"] + outputs["log_pz"]
      log_det -= log_qugx
    else:
      z = jnp.floor(x*self.scale).astype(jnp.int32)

    return {"x": z, "log_det": log_det}

