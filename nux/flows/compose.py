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
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           accumulate: Iterable[str]=["log_det", "flow_norm"],
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    n_layers = len(self.layers)
    iter_layers = self.layers if sample == False else self.layers[::-1]

    # Want to make sure that we're passing all inputs/outputs to the next layer
    final_outputs = inputs.copy()
    accumulated_outputs = dict([(name, jnp.array(0.0)) for name in accumulate])
    accumulated_found = dict([(name, False) for name in accumulate])

    # Split the random key
    rngs = random.split(rng, n_layers) if rng is not None else [None]*n_layers

    # Run the rest of the layers
    layer_inputs = inputs.copy()
    for layer, rng in zip(iter_layers, rngs):
      outputs = layer(layer_inputs, rng, sample=sample, **kwargs)
      layer_inputs["x"] = outputs["x"]
      final_outputs.update(outputs)

      # Remember to accumulate the outputs
      for name in accumulated_outputs.keys():
        if(name in outputs):
          accumulated_outputs[name] += outputs[name]
          accumulated_found[name] = True

    # Swap in the accumulated outputs
    for name, val in accumulated_outputs.items():
      if(accumulated_found[name]):
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
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           accumulate: Iterable[str]=["log_det", "flow_norm"],
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    n_layers = len(self.layers)
    iter_layers = self.layers if sample == False else self.layers[::-1]

    # Want to make sure that we're passing all inputs/outputs to the next layer
    final_outputs = inputs.copy()
    accumulated_outputs = dict([(name, jnp.array(0.0)) for name in accumulate])
    accumulated_found = dict([(name, False) for name in accumulate])

    # Split x
    xs = jnp.split(inputs["x"], n_layers, self.axis)
    zs = []

    # Split the random key
    rngs = random.split(rng, n_layers) if rng is not None else [None]*n_layers

    # Run each of the flows on a part of x
    layer_inputs = inputs.copy()
    for x, layer, rng in zip(xs, self.layers, rngs):
      layer_inputs["x"] = x
      outputs = layer(layer_inputs, rng, sample=sample, **kwargs)
      final_outputs.update(outputs)
      zs.append(outputs["x"])

      # Remember to accumulate the outputs
      for name in accumulated_outputs.keys():
        if(name in outputs):
          accumulated_outputs[name] += outputs[name]
          accumulated_found[name] = True

    # Swap in the accumulated outputs
    for name, val in accumulated_outputs.items():
      if(accumulated_found[name]):
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
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    flow = sequential(nux.Squeeze(),
                      factored(self.flow, nux.Identity()),
                      nux.UnSqueeze())

    return flow(inputs, rng, sample=sample)

################################################################################################################

class track(Layer):

  def __init__(self, flow, name: str, reducer: Callable=None, **kwargs):
    self.name = name
    self.flow = flow
    self.reducer = reducer
    super().__init__(name=name, **kwargs)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    flow_outputs = self.flow(inputs, rng, sample=sample, **kwargs)
    outputs = flow_outputs.copy()
    if(self.reducer is None):
      outputs[self.name] = flow_outputs
    else:
      outputs[self.name] = self.reducer(flow_outputs)
    return outputs