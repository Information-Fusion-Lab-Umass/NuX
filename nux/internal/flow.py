from functools import partial, wraps
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
import warnings
from typing import Optional, Mapping, Type, Callable, Iterable, Any, Sequence, Union, Tuple, MutableMapping, NamedTuple, Set, TypeVar
import nux.util as util
from haiku._src.typing import PRNGKey, Params, State
from nux.internal.transform import transform_flow

__all__ = ["Flow"]

class Flow():
  """ Convenience class to wrap a Layer class

      Args:
          flow     - A Flow object.
          clip     - How much to clip gradients.  This is crucial for stable training!
          warmup   - How much to warm up the learning rate.
          lr_decay - Learning rate decay.
          lr       - Max learning rate.
  """
  def __init__(self,
               create_fun: Callable,
               key: PRNGKey,
               inputs: Mapping[str,jnp.ndarray],
               batch_axes: Sequence[int],
               check_for_bad_init: Optional[bool]=False,
               **kwargs):

    self._flow = transform_flow(create_fun)
    self.params, self.state, outputs = self._flow.init(key,
                                                       inputs,
                                                       batch_axes=batch_axes,
                                                       return_initial_output=True)

    if check_for_bad_init:
      self.check_init(outputs)

    self.n_params = jax.flatten_util.ravel_pytree(self.params)[0].size

    self.data_shape   = inputs["x"].shape[len(batch_axes):]
    self.latent_shape = outputs["x"].shape[len(batch_axes):]
    self.scan_apply_loop = jit(partial(jax.lax.scan, partial(self.scan_body, is_training=True)))
    self.scan_apply_test_loop = jit(partial(jax.lax.scan, partial(self.scan_body, is_training=False)))

  def to_bits_per_dim(self, log_likelihood):
    return log_likelihood/util.list_prod(self.data_shape)/jnp.log(2)

  def get_batch_shape(self, inputs):
    return inputs["x"].shape[:-len(self.data_shape)]

  #############################################################################

  def check_init(self, outputs):
    import pdb; pdb.set_trace()

  #############################################################################

  def process_outputs(self, outputs):
    # Only set log_px if outputs has both a prior and log_det term
    if "log_pz" not in outputs:
      warnings.warn("Flow does not have a prior")
    if "log_det" not in outputs:
      warnings.warn("Flow does not have a transformation")

    outputs["log_px"] = outputs.get("log_pz", 0.0) + outputs.get("log_det", 0.0)
    return outputs

  #############################################################################

  @property
  def _apply_fun(self):
    return self._flow.apply

  def apply(self,
            key: PRNGKey,
            inputs: Mapping[str, jnp.ndarray],
            **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    outputs, self.state = self._flow.apply(self.params, self.state, key, inputs, **kwargs)
    return self.process_outputs(outputs)

  def stateful_apply(self,
                     key: PRNGKey,
                     inputs: Mapping[str, jnp.ndarray],
                     params: Params,
                     state: State,
                     **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    outputs, state = self._flow.apply(params, state, key, inputs, **kwargs)
    return self.process_outputs(outputs), state

  #############################################################################

  def scan_body(self, carry, scan_inputs, **kwargs):
    key, _inputs = scan_inputs
    params, state = carry
    outputs, state = self.stateful_apply(key, _inputs, params, state, **kwargs)
    return (params, state), outputs

  def scan_apply(self,
                 key: PRNGKey,
                 inputs: Mapping[str, jnp.ndarray],
                 is_training: bool=True
  ) -> Mapping[str, jnp.ndarray]:
    """ Applies a lax.scan loop to the first batch axis
    """
    if len(inputs["x"].shape) == len(self.data_shape):
      assert 0, "Expect a batched or doubly-batched input"

    # Get the inputs for the scan loop
    n_iters = inputs["x"].shape[0]
    keys = random.split(key, n_iters)
    scan_inputs = (keys, inputs)
    scan_carry = (self.params, self.state)
    if is_training:
      (_, self.state), outputs = self.scan_apply_loop(scan_carry, scan_inputs)
    else:
      (_, self.state), outputs = self.scan_apply_test_loop(scan_carry, scan_inputs)
    return self.process_outputs(outputs)

  #############################################################################

  def sample(self,
             key: PRNGKey,
             n_samples: int,
             n_batches: Optional[int]=None,
             labels: Optional=None,
             **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    if n_batches is None:
      dummy_z = jnp.zeros((n_samples,) + self.latent_shape)
      outputs = self.apply(key, {"x": dummy_z}, sample=True, is_training=False, **kwargs)
    else:
      dummy_z = jnp.zeros((n_batches, n_samples) + self.latent_shape)
      outputs = self.scan_apply_test_loop(key, {"x": dummy_z}, sample=True, is_training=False, **kwargs)
    return self.process_outputs(outputs)

  def reconstruct(self,
                  key: PRNGKey,
                  inputs: Mapping[str, jnp.ndarray],
                  scan_loop: bool=False,
                  **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    if scan_loop == False:
      outputs = self.apply(key, inputs, sample=True, reconstruction=True, is_training=False, **kwargs)
    else:
      outputs = self.scan_apply(key, inputs, sample=True, reconstruction=True, is_training=False, **kwargs)
    return self.process_outputs(outputs)

  #############################################################################

  def save(self, path: str=None):
    save_items = {"params": self.params,
                  "state": self.state}
    util.save_pytree(save_items, path, overwrite=True)

  def load(self, path: str=None):
    loaded_items = util.load_pytree(path)
    self.params = loaded_items["params"]
    self.state = loaded_items["state"]
