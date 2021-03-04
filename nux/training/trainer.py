from functools import partial
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
from nux.internal.flow import Flow
import nux.util as util
from typing import Optional, Mapping, Callable, Sequence, Tuple, Any
from haiku._src.typing import Params, State, PRNGKey
import optax
from optax._src import transform
GradientTransformation = transform.GradientTransformation
from abc import ABC, abstractmethod
from collections import namedtuple
from .objective_base import Evaluate

TrainState = namedtuple("TrainState", ["params", "state", "opt_state"])
TrainOutput = namedtuple("TrainOutput", ["loss", "aux", "grad_summary"])

class Trainer(Evaluate):
  """
  Mixin class that will be combined with the Loss and TesterMixin classes.
  Provides the functionality needed to optimize the loss function (and track
  auxilliary things from the loss and gradients)
  """
  def __init__(self,
               params,
               state,
               loss_fun,
               optimizer: GradientTransformation=None,
               **kwargs):

    self.loss_fun = partial(loss_fun, is_training=True)
    self.params = params
    self.state = state

    # Get the optimizer
    if optimizer is None:
      opt_init, opt_update = self.build_optimizer(**kwargs)
    else:
      opt_init, opt_update = optimizer

    # Initialize the optimizer state
    self.opt_update = jit(opt_update)
    self.opt_state = opt_init(params)
    self.apply_updates = jit(optax.apply_updates)

    # Build the value and grad function
    self.valgrad = jax.value_and_grad(self.loss_fun, has_aux=True)
    self.valgrad = jit(self.valgrad)

    self.compiled_scan_loop = jit(partial(jax.lax.scan, self._step))

    self.losses = jnp.array([])
    self.aux = None
    self.grad_summaries = None

    self.train_key = None

  def build_optimizer(self,
                      clip=15.0,
                      lr=5e-4,
                      warmup=2000,
                      cosine_decay_steps=None,
                      optimizer_name="adabelief"
  ) -> GradientTransformation:
    chain = []
    if optimizer_name == "adabelief":
      chain.append(util.scale_by_belief())
    elif optimizer_name == "adam":
      chain.append(optax.scale_by_adam())
    else:
      assert 0

    # Make sure to use the negative learning rate so that we minimize
    if warmup and warmup > 0:
      warmup_schedule = partial(util.linear_warmup_lr_schedule, warmup=warmup, lr_decay=1.0, lr=-lr)
      chain.append(optax.scale_by_schedule(warmup_schedule))
    else:
      chain.append(optax.scale(-lr))

    if cosine_decay_steps and cosine_decay_steps > 0:
      cosine_lr = optax.cosine_decay_schedule(init_value=1.0, decay_steps=cosine_decay_steps, alpha=1e-1)
      chain.append(optax.scale_by_schedule(cosine_lr))

    if clip and clip > 0:
      chain.append(optax.clip(clip))

    return optax.chain(*chain)

  def grad_hook(self, grad):
    return ()

  @property
  def complete_state(self) -> TrainState:
    return TrainState(self.params, self.state, self.opt_state)

  @complete_state.setter
  def complete_state(self, val: TrainState):
    self.params    = val.params
    self.state     = val.state
    self.opt_state = val.opt_state

  def update_outputs(self, out: TrainOutput):

    # Update the train losses
    self.losses = jnp.hstack([self.losses, out.loss])

    # Update the auxiliary loss values and the gradient summaries
    if self.aux is None:
      self.aux = out.aux
      self.grad_summaries = out.grad_summary
    else:
      self.aux = util.tree_concat(self.aux, out.aux, axis=0)
      self.grad_summaries = util.tree_concat(self.grad_summaries, out.grad_summary, axis=0)

  def save_items(self):
    save_items = {"params": self.params,
                  "state": self.state,
                  "opt_state": self.opt_state,
                  "train_losses": self.losses,
                  "train_aux": self.aux,
                  "grad_summaries": self.grad_summaries}
    return save_items

  def load_items(self, items):
    self.params = items["params"]
    self.state = items["state"]
    self.opt_state = items["opt_state"]
    self.losses = items["train_losses"]
    self.aux = items["train_aux"]
    self.grad_summaries = items["grad_summaries"]

  def _step(self,
            carry: TrainState,
            scan_inputs: Tuple[PRNGKey, Mapping[str, jnp.ndarray]],
            **kwargs
  ) -> Tuple[TrainState, TrainOutput]:
    """ Lowest level gradient step.  Can be used inside a scan loop """
    params, state, opt_state = carry.params, carry.state, carry.opt_state
    key, inputs = scan_inputs

    # Take a gradient step
    (train_loss, (aux, state)), grad = self.valgrad(params, state, key, inputs, **kwargs)

    # Update the parameters and optimizer state
    updates, opt_state = self.opt_update(grad, opt_state, params)
    params = self.apply_updates(params, updates)

    # Monitor the gradients
    grad_summary = self.grad_hook(grad)

    carry = TrainState(params, state, opt_state)
    out = TrainOutput(train_loss, aux, grad_summary)

    return carry, out

################################################################################################################
