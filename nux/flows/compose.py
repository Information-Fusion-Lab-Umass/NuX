from functools import partial
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
import haiku as hk
from typing import Optional, Mapping, Type, Callable, Iterable, Any, Sequence, Union, Tuple
import nux.util as util
from nux.flows.base import *
import nux

__all__ = ["sequential",
           "factored",
           "multi_scale",
           "track"]

################################################################################################################

class sequential(Layer):

  def __init__(self, *layers: Iterable[Callable], name: str="sequential", **kwargs):
    super().__init__(name=name, **kwargs)
    self.layers = tuple(layers)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           sample: Optional[bool]=False,
           accumulate: Iterable[str]=["log_det", "flow_norm"],
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    iter_layers = self.layers if sample == False else self.layers[::-1]

    # Want to make sure that we're passing all inputs/outputs to the next layer
    final_outputs = inputs.copy()
    accumulated_outputs = dict([(name, jnp.array(0.0)) for name in accumulate])

    # Run the rest of the layers
    layer_inputs = inputs.copy()
    for layer in iter_layers:
      outputs = layer(layer_inputs, sample=sample, **kwargs)
      layer_inputs.update(outputs)
      final_outputs.update(outputs)

      # Remember to accumulate the outputs
      for name in accumulated_outputs.keys():
        accumulated_outputs[name] += outputs.get(name, jnp.array(0.0))

    # Swap in the accumulated outputs
    for name, val in accumulated_outputs.items():
      final_outputs[name] = val

    return final_outputs

################################################################################################################

class factored(Layer):

  def __init__(self, *layers: Iterable[Callable], axis: Optional[int]=-1, name: str="sequential", **kwargs):
    super().__init__(name=name, **kwargs)
    self.layers = tuple(layers)
    self.axis   = axis

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           sample: Optional[bool]=False,
           accumulate: Iterable[str]=["log_det", "flow_norm"],
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    iter_layers = self.layers if sample == False else self.layers[::-1]

    # Want to make sure that we're passing all inputs/outputs to the next layer
    final_outputs = inputs.copy()
    accumulated_outputs = dict([(name, jnp.array(0.0)) for name in accumulate])

    # Split x
    xs = jnp.split(inputs["x"], len(self.layers), self.axis)
    zs = []

    # Run each of the flows on a part of x
    layer_inputs = inputs.copy()
    for x, layer in zip(xs, self.layers):
      layer_inputs["x"] = x
      outputs = layer(layer_inputs, sample=sample, **kwargs)
      final_outputs.update(outputs)
      zs.append(outputs["x"])

      # Remember to accumulate the outputs
      for name in accumulated_outputs.keys():
        accumulated_outputs[name] += outputs.get(name, jnp.array(0.0))

    # Swap in the accumulated outputs
    for name, val in accumulated_outputs.items():
      final_outputs[name] = val

    # Recombine the data
    z = jnp.concatenate(zs, self.axis)
    final_outputs["x"] = z
    return final_outputs

################################################################################################################

class multi_scale(Layer):

  def __init__(self, flow, name: str="multi_scale", **kwargs):
    super().__init__(name=name, **kwargs)
    self.flow = flow

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    flow = sequential(nux.Squeeze(),
                      factored(self.flow, nux.Identity()),
                      nux.UnSqueeze())

    return flow(inputs, sample=sample)

################################################################################################################

class track(Layer):

  def __init__(self, flow, name: str, reducer: Callable=None, **kwargs):
    self.name = name
    self.flow = flow
    self.reducer = reducer
    super().__init__(name=name, **kwargs)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    flow_outputs = self.flow(inputs, sample=sample, **kwargs)
    outputs = flow_outputs.copy()
    if(self.reducer is None):
      outputs[self.name] = flow_outputs
    else:
      outputs[self.name] = self.reducer(flow_outputs)
    return outputs