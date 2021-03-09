from functools import partial
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
from nux.internal.flow import Flow
import nux.util as util
from typing import Optional, Mapping, Callable, Sequence, Tuple, Any
from haiku._src.typing import Params, State, PRNGKey
from collections import namedtuple
from .objective_base import Evaluate

TestState = namedtuple("TestState", ["params", "state"])
TestOut = namedtuple("TestOut", ["loss", "aux"])

class Tester(Evaluate):
  def __init__(self,
               params,
               state,
               loss_fun,
               **kwargs):

    self.params = params
    self.state = state

    self.loss_fun = partial(loss_fun, is_training=False)
    self.compiled_scan_loop = jit(partial(jax.lax.scan, self._step))

    self.losses = jnp.array([])
    self.aux = None

  @property
  def complete_state(self) -> TestState:
    return TestState(self.params, self.state)

  @complete_state.setter
  def complete_state(self, val: TestState):
    self.params    = val.params
    self.state     = val.state

  def update_outputs(self, out: TestOut):
    self.losses = jnp.hstack([self.losses, out.loss])

    # Update the auxiliary loss values and the gradient summaries
    if self.aux is None:
      self.aux = out.aux
    else:
      self.aux = util.tree_hstack(self.aux, out.aux)

  def save_items(self):
    save_items = {"params": self.params,
                  "state": self.state,
                  "test_losses": self.losses,
                  "test_aux": self.aux}
    return save_items

  def load_items(self, items):
    self.params = items["params"]
    self.state = items["state"]
    self.losses = items["test_losses"]
    self.aux = items["test_aux"]

  def _step(self,
            carry,
            scan_inputs,
            **kwargs):
    params, state = carry.params, carry.state
    key, inputs = scan_inputs

    # Evaluate the loss function
    loss, (aux, state) = self.loss_fun(params, state, key, inputs, **kwargs)

    carry = TestState(params, state)
    out = TestOut(loss, aux)

    return carry, out
