from functools import partial
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
from nux.internal.flow import Flow
import nux.util as util
from typing import Optional, Mapping, Callable, Sequence, Tuple, Any
from haiku._src.typing import Params, State, PRNGKey
from abc import ABC, abstractmethod
from collections import namedtuple

class CompleteState(ABC):
  pass

class Output(ABC):
  pass

################################################################################################################

class Evaluate(ABC):
  def __init__(self):
    self.compiled_scan_loop = None

  @property
  @abstractmethod
  def complete_state(self) -> CompleteState:
    return self._complete_state

  @complete_state.setter
  @abstractmethod
  def complete_state(self, val: CompleteState):
    pass

  @abstractmethod
  def update_outputs(self, out: Output):
    pass

  @abstractmethod
  def save_items(self):
    pass

  @abstractmethod
  def load_items(self, items):
    pass

  @abstractmethod
  def _step(self,
            carry: CompleteState,
            scan_inputs: Tuple[PRNGKey, Mapping[str, jnp.ndarray]],
            **kwargs
  ) -> Tuple[CompleteState, Output]:
    pass

  def step(self,
           key: PRNGKey,
           inputs: Mapping[str, jnp.ndarray],
           **kwargs
  ) -> Output:
    """ Takes a single step """
    self.complete_state, out = self._step(self.complete_state, (key, inputs), **kwargs)

    # Update the outputs from the step
    self.update_outputs(out)
    return out

  def step_for_loop(self,
                    key: PRNGKey,
                    inputs: Mapping[str, jnp.ndarray],
                    **kwargs
  ) -> Output:
    """ Takes multiple steps using a for loop """

    # Get the inputs for the for loop
    n_iters = inputs["x"].shape[0]
    keys = random.split(key, n_iters)
    scan_inputs = (keys, inputs)

    outs = []
    import tqdm
    pbar = tqdm.tqdm(list(enumerate(keys)), leave=False)
    for i, key in pbar:
      _inputs = jax.tree_map(lambda x: x[i], inputs)
      self.complete_state, out = self._step(self.complete_state, (key, _inputs), **kwargs)
      outs.append(out)

    def concat(*args):
      try:
        return jnp.concatenate(args, axis=0)
      except ValueError:
        return jnp.array(args)

    outs = jax.tree_multimap(concat, *outs)
    self.update_outputs(outs)
    return outs

  def step_scan_loop(self,
                     key: PRNGKey,
                     inputs: Mapping[str, jnp.ndarray],
                     update: bool=True,
                     **kwargs
  ) -> Output:
    # Get the inputs for the scan loop
    n_iters = inputs["x"].shape[0]
    keys = random.split(key, n_iters)
    scan_inputs = (keys, inputs)

    # Run the training steps
    self.complete_state, out = self.compiled_scan_loop(self.complete_state, scan_inputs)
    if update:
      self.update_outputs(out)
    return out
