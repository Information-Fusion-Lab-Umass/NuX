from functools import partial
import jax.numpy as jnp
import numpy as np
import jax
from jax import random, jit, vmap
import haiku as hk
from typing import Optional, Mapping, Type, Callable, Iterable, Any, Sequence, Union, Tuple
import nux.util as util
from nux.internal.layer import Layer, InvertibleLayer
import nux
from nux.internal.functional import make_functional_modules
from haiku._src import data_structures
from nux.internal.base import get_constant
import haiku._src.base as hk_base
from haiku._src.base import ThreadLocalStack, \
                            MutableParams, \
                            MutableState, \
                            PRNGSequence, \
                            Frame, \
                            frame_stack, \
                            extract_state, \
                            Stack, \
                            ModuleState, \
                            StatePair, \
                            current_frame, \
                            current_bundle_name

__all__ = ["repeat",
           "sequential",
           "factored",
           "multi_scale",
           "as_flat",
           "reverse_flow",
           "track"]

################################################################################################################

def _batch_repeated_layers(params, param_hashes):
  # Generate a mapping from hash to parameter name
  param_hash_map = {hash(k): k for k in params.keys()}

  # Go through the parameter hashes and find the order that we should
  # batch the parameters together.
  batched_params = {}
  for h_base, all_hashes in param_hashes.items():
    base_layer_name = param_hash_map[h_base]
    batched_params[base_layer_name] = []
    for h in all_hashes:
      layer_name = param_hash_map[h]
      batched_params[base_layer_name].append(params[layer_name])

    # Batch the parameters
    batched_params[base_layer_name] = jax.tree_multimap(lambda *xs: jnp.stack([*xs]), *batched_params[base_layer_name])

  return batched_params

class repeat(InvertibleLayer):

  def __init__(self,
               layer_create_fun: Iterable[Callable],
               n_repeats: int,
               name: str="repeat"
  ):
    """ Create a flow sequentially
    Args:
      layers: An iterable that contains flow layers
      name  : Optional name for this module.
    """
    super().__init__(name=name)
    self.layer_create_fun = layer_create_fun
    self.n_repeats = n_repeats

  @hk.transparent
  def get_layer(self):
    # Don't want to compile a new object every time we call this function.
    # Also the hk.transparent is necessary for the names to be consistent!
    if hasattr(self, "_layer"):
      return self._layer
    self._layer = self.layer_create_fun()
    return self._layer

  def get_parameter_and_state_names(self, layer):

    # Store the names of the parameters for the scan loop
    with make_functional_modules([layer]) as ([apply_fun], \
                                               params, \
                                               (state, constants, rng_seq), \
                                               finalize):
      bundle_name = current_bundle_name()

      # Filter out the params and states that aren't a part of this repeat
      filtered_params = {key: val for (key, val) in params.items() if key.startswith(bundle_name)}
      filtered_state  = {key: val for (key, val) in state.items() if key.startswith(bundle_name)}

      # Order the parameters correctly and separate the keys from values
      sorted_params = sorted(filtered_params.items(), key=lambda x: x[0])
      sorted_state  = sorted(filtered_state.items(), key=lambda x: x[0])

      param_names, param_vals = zip(*sorted_params)
      param_shapes = util.tree_shapes(param_vals)
      if len(sorted_state) == 0:
        state_names = ()
        state_shapes = ()
      else:
        state_names, state_vals = zip(*sorted_state)
        state_shapes = util.tree_shapes(state_vals)

      finalize(params, (state, constants, rng_seq))

    return (param_names, state_names), (param_shapes, state_shapes)

  @hk.transparent
  def call_no_scan(self,
                   inputs: Mapping[str, jnp.ndarray],
                   rng: jnp.ndarray=None,
                   sample: Optional[bool]=False,
                   **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    # Want to make sure that we're passing all inputs/outputs to the next layer
    final_outputs = inputs.copy()
    log_det = jnp.array(0.0)

    # Split the random key
    rngs = random.split(rng, self.n_repeats) if rng is not None else [None]*self.n_repeats

    # We might need to sort out what parameter names correspond to each other
    # across repeated layers
    init_names = False
    if get_constant("param_state_name_hashes", value=None) is None:
      init_names = True

      # Keep track of the parameter names for a single layer call
      layer_param_names = {}
      layer_state_names = {}

      # Keep track of the parameter names for each layer
      used_param_names = set()
      used_state_names = set()

    # Run the rest of the layers
    layer_inputs = inputs.copy()
    for i, rng in enumerate(rngs):
      layer = self.layer_create_fun()
      outputs = layer(layer_inputs, rng, sample=sample, **kwargs)
      layer_inputs["x"] = outputs["x"]
      final_outputs.update(outputs)

      # Remember to accumulate the outputs
      log_det += outputs.get("log_det", 0.0)

      # Match the new layers parameters with the original layer's parameters
      if init_names:
        names, shapes = self.get_parameter_and_state_names(layer)
        param_names, state_names = names
        param_shapes, state_shapes = shapes
        param_names_and_shapes = dict(zip(param_names, param_shapes))
        state_names_and_shapes = dict(zip(state_names, state_shapes))

        # Find the new names
        new_param_names = set(param_names).difference(used_param_names)
        new_state_names = set(state_names).difference(used_state_names)

        if i == 0:
          layer_param_names = {k:[k] for k in new_param_names}
          layer_state_names = {k:[k] for k in new_state_names}
        else:
          # Match the new names to the existing ones
          # param_matching = util.match_strings(layer_param_names.keys(), new_param_names)
          # state_matching = util.match_strings(layer_state_names.keys(), new_state_names)
          param_matching = util.match_strings_using_shapes(layer_param_names.keys(), new_param_names, param_names_and_shapes)
          state_matching = util.match_strings_using_shapes(layer_state_names.keys(), new_state_names, state_names_and_shapes)

          # Update the layer lists
          for k, v in param_matching.items():
            expected_shape, new_shape = param_names_and_shapes[k], param_names_and_shapes[v]
            assert util.tree_equal(expected_shape, new_shape)
            layer_param_names[k].append(v)

          for k, v in state_matching.items():
            expected_shape, new_shape = state_names_and_shapes[k], state_names_and_shapes[v]
            assert util.tree_equal(expected_shape, new_shape)
            layer_state_names[k].append(v)

        # Update the set of names that have been used
        used_param_names.update(new_param_names)
        used_state_names.update(new_state_names)

    if init_names:
      # Turn the strings into hashes so that they can be used in JAX
      param_hashes = {hash(k): v for (k, v) in jax.tree_map(hash, layer_param_names).items()}
      state_hashes = {hash(k): v for (k, v) in jax.tree_map(hash, layer_state_names).items()}
      get_constant("param_state_name_hashes", (param_hashes, state_hashes))

    # Swap in the accumulated outputs
    final_outputs["log_det"] = log_det

    return final_outputs

  @hk.transparent
  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           no_scan: bool=False,
           accumulate: Iterable[str]=["log_det", "aux_loss"],
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    if Layer._is_initializing:
      return self.call_no_scan(inputs, rng, sample=sample, **kwargs)

    # Want to make sure that we're passing all inputs/outputs to the next layer
    final_outputs = inputs.copy()

    # Need to get the funcitonal apply fun
    with make_functional_modules([self.get_layer()]) as ([apply_fun], \
                                                          params, \
                                                          (state, constants, rng_seq), \
                                                          finalize):
      # Retrieve the hashes of the names of the parameters and states for the layer call
      param_hashes, state_hashes = get_constant("param_state_name_hashes", None)

      # Batch together the parameters and state across the repeated layers
      scan_params = _batch_repeated_layers(params, param_hashes)
      scan_params = data_structures.to_immutable_dict(scan_params)
      scan_state = _batch_repeated_layers(state, state_hashes)

      # Reverse the order if we are sampling
      if sample == True:
        scan_params = jax.tree_map(lambda x: x[::-1], scan_params)
        scan_state = jax.tree_map(lambda x: x[::-1], scan_state)

      # Pass other inputs we might have through the network
      shared_inputs = inputs.copy()
      del shared_inputs["x"]

      # Use a scan loop so that we only need to compile layer once!
      def scan_body(carry, scan_inputs):
        x = carry
        params, state, rng = scan_inputs

        # Bundle the non-parameter state together
        bundled_state = (state, constants, rng_seq)

        # Make sure that we're passing all of the inputs (such as labels) to the layer
        inputs = shared_inputs.copy()
        inputs["x"] = x

        # Run the function
        outputs, bundled_state = apply_fun(params, bundled_state, inputs, rng, sample=sample, **kwargs)

        # Retrieve the state because it might have changed
        state, _, _ = bundled_state

        # Return the stuff we need
        x = outputs["x"]
        del outputs["x"]
        return x, (outputs, state)

      # Run the scan function
      rngs = random.split(rng, self.n_repeats) if rng is not None else [None]*self.n_repeats
      x, (batched_outputs, batched_updated_state) = jax.lax.scan(scan_body, inputs["x"], (scan_params, scan_state, rngs))

      # Reverse the updated state if we are sampling
      if sample == True:
        batched_updated_state = jax.tree_map(lambda x: x[::-1], batched_updated_state)

      # Search through the outputs to find things we want to accumulate
      accumulated_outputs = {}
      for name in accumulate:
        if name in batched_outputs:
          accumulated_outputs[name] = batched_outputs[name].sum(axis=0)
          del batched_outputs[name]

      # Convert the output of the scan into the same state data structure that was passed in.
      hash_map = {hash(k): k for k in state.keys()}
      rev_hash_map = {k: hash(k) for k in state.keys()}
      updated_state = state.copy()
      for base_layer_name, pytree in batched_updated_state.items():

        # Retrieve the names of each repeated layer
        layer_names = [hash_map[k] for k in state_hashes[rev_hash_map[base_layer_name]]]

        # Split the batched parameters
        leaves, treedef = jax.tree_flatten(batched_updated_state[base_layer_name])
        split_states = [jax.tree_unflatten(treedef, [l[i] for l in leaves]) for i in range(self.n_repeats)]

        # Update the state dictionary
        updated_state.update(dict(zip(layer_names, split_states)))

      # Just in case
      updated_state = jax.lax.stop_gradient(updated_state)

      # Only state might be different
      bundled_state = (updated_state, constants, rng_seq)
      finalize(params, bundled_state)

    outputs = {"x": x}
    outputs.update(accumulated_outputs)

    return outputs

################################################################################################################

class sequential(InvertibleLayer):

  def __init__(self,
               *layers: Iterable[Callable],
               name: str="sequential"
  ):
    """ Create a flow sequentially
    Args:
      layers: An iterable that contains flow layers
      name  : Optional name for this module.
    """
    super().__init__(name=name)
    self.layers = tuple(layers)

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           accumulate: Iterable[str]=["log_det", "aux_loss"],
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
    for i, (layer, rng) in enumerate(zip(iter_layers, rngs)):
      outputs = layer(layer_inputs, rng, sample=sample, **kwargs)
      layer_inputs["x"] = outputs["x"]
      final_outputs.update(outputs)

      # Remember to accumulate the outputs
      for name in accumulated_outputs.keys():
        if name in outputs:
          accumulated_outputs[name] += outputs[name]
          accumulated_found[name] = True

    # Swap in the accumulated outputs
    for name, val in accumulated_outputs.items():
      if accumulated_found[name]:
        final_outputs[name] = val

    return final_outputs

################################################################################################################

class factored(InvertibleLayer):

  def __init__(self,
               *layers: Iterable[Callable],
               axis: Optional[int]=-1,
               first_factor_ratio: int=2,
               name: str="sequential"
  ):
    """ Create a flow in parallel.  This is basically like using the chain
        rule and applying a flow to each part.
    Args:
      layers            : An iterable that contains flow layers
      axis              : Which axis to factor on
      first_factor_ratio: How large the first component should be compared to the second
      name              : Optional name for this module.
    """
    super().__init__(name=name)
    self.layers             = tuple(layers)
    self.axis               = axis
    self.first_factor_ratio = first_factor_ratio

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           accumulate: Iterable[str]=["log_det", "aux_loss"],
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    n_layers = len(self.layers)
    iter_layers = self.layers if sample == False else self.layers[::-1]

    # Want to make sure that we're passing all inputs/outputs to the next layer
    final_outputs = inputs.copy()
    accumulated_outputs = dict([(name, jnp.array(0.0)) for name in accumulate])
    accumulated_found = dict([(name, False) for name in accumulate])

    # Split x
    # split_size = inputs["x"].shape[self.axis]/n_layers
    # split_size = jnp.ceil(split_size).astype(int)
    # split_idx = jnp.array([i*split_size for i in range(1, n_layers)])
    # xs = jnp.split(inputs["x"], indices_or_sections=split_idx, axis=self.axis)

    if n_layers == 2:
      split_idx = inputs["x"].shape[self.axis]//self.first_factor_ratio
      xs = jnp.split(inputs["x"], indices_or_sections=(split_idx,), axis=self.axis)
      # xs = jnp.split(inputs["x"], n_layers, self.axis)
    else:
      assert 0, "Not implemented"

    zs = []

    # Split the random key
    rngs = random.split(rng, n_layers) if rng is not None else [None]*n_layers

    # Run each of the flows on a part of x
    layer_inputs = inputs.copy()
    for i, (x, layer, rng) in enumerate(zip(xs, self.layers, rngs)):
      layer_inputs["x"] = x
      outputs = layer(layer_inputs, rng, sample=sample, **kwargs)
      final_outputs.update(outputs)
      zs.append(outputs["x"])

      # Remember to accumulate the outputs
      for name in accumulated_outputs.keys():
        if name in outputs:
          accumulated_outputs[name] += outputs[name]
          accumulated_found[name] = True

    # Swap in the accumulated outputs
    for name, val in accumulated_outputs.items():
      if accumulated_found[name]:
        final_outputs[name] = val

    # Recombine the data
    z = jnp.concatenate(zs, self.axis)
    final_outputs["x"] = z
    return final_outputs

################################################################################################################

class multi_scale(InvertibleLayer):

  def __init__(self,
               flow,
               condition: bool=False,
               name: str="multi_scale",
  ):
    """ Use a flow in a multiscale architecture as described in RealNVP https://arxiv.org/pdf/1605.08803.pdf
        Factors half of the dimensions.
    Args:
      flow: The flow to use
      name: Optional name for this module.
    """
    super().__init__(name=name)
    self.flow = flow
    self.condition = condition

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]

    # Split the input vector on the channel dimension
    xa, xb = jnp.split(x, indices_or_sections=jnp.array([x.shape[-1]//2]), axis=-1)

    # Run a flow on only one half
    factored_inputs = inputs.copy()
    factored_inputs["x"] = xb
    if self.condition:
      factored_inputs["condition"] = xa
    outputs = self.flow(factored_inputs, rng, sample=sample, **kwargs)

    z = jnp.concatenate([xa, outputs["x"]], axis=-1)

    outputs["x"] = z
    return outputs

################################################################################################################

class as_flat(InvertibleLayer):

  def __init__(self,
               flow,
               name: str="as_flat",
  ):
    """ Reshape the input to 1d
    Args:
      flow: The flow to use
      name: Optional name for this module.
    """
    super().__init__(name=name)
    self.flow = flow

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    x = inputs["x"]
    x_shape = x.shape
    x_flat = x.reshape(self.batch_shape + (-1,))

    # Run a flow on only one half
    flat_inputs = inputs.copy()
    flat_inputs["x"] = x_flat
    outputs = self.flow(flat_inputs, rng, sample=sample, **kwargs)

    z = outputs["x"].reshape(x_shape)
    outputs["x"] = z
    return outputs

################################################################################################################

class reverse_flow(InvertibleLayer):

  def __init__(self,
               flow,
               name: str="reverse_flow",
  ):
    """ Reverse the direction of a flow.  Useful if one direction is faster than the other.
    Args:
      flow: The flow to use
      name: Optional name for this module.
    """
    super().__init__(name=name)
    self.flow = flow

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    outputs = self.flow(inputs, rng, not sample, **kwargs)

    outputs["log_det"] *= -1

    return outputs

################################################################################################################

class track(InvertibleLayer):

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
    if self.reducer is None:
      outputs[self.name] = flow_outputs
    else:
      outputs[self.name] = self.reducer(flow_outputs)
    return outputs

################################################################################################################

if __name__ == "__main__":
  from debug import *
  from nux.flows.bijective.affine import ShiftScale, AffineDense, Scale

  Dense = partial(AffineDense, weight_norm=False, spectral_norm=True)

  def block():
    # return ShiftScale()
    return nux.sequential(Dense(), ShiftScale())

  def create_fun(should_repeat=True, n_repeats=2):
    if should_repeat:
      repeated = repeat(block, n_repeats=n_repeats)
    else:
      repeated = nux.sequential(*[block() for _ in range(n_repeats)])
    return repeated
    # return sequential(ShiftScale(),
    #                   sequential(repeated),
    #                              Scale(0.2))

  rng = random.PRNGKey(1)
  x = random.normal(rng, (10, 7, 3))
  inputs = {"x": x[0]}
  flow = nux.Flow(create_fun, rng, inputs, batch_axes=(0,))

  outputs1 = flow.apply(rng, inputs)
  outputs2 = flow.apply(rng, inputs, no_scan=True)

  doubly_batched_inputs = {"x": x}
  trainer = nux.MaximumLikelihoodTrainer(flow)

  trainer.grad_step(rng, inputs)
  trainer.grad_step_for_loop(rng, doubly_batched_inputs)
  trainer.grad_step_scan_loop(rng, doubly_batched_inputs)





  rng = random.PRNGKey(1)
  x = random.normal(rng, (10, 3))
  inputs = {"x": x}
  flow = nux.Flow(partial(create_fun, should_repeat=False), rng, inputs, batch_axes=(0,))

  outputs3 = flow.apply(rng, inputs)
  outputs4 = flow.apply(rng, inputs)

  import pdb; pdb.set_trace()
