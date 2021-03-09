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
from .trainer import Trainer
from .tester import Tester

__all__ = ["MaximumLikelihoodTrainer",
           "JointClassificationTrainer"]

################################################################################################################

class FlowTrainer(ABC):

  def __init__(self, flow: Flow, optimizer: GradientTransformation, **kwargs):
    self.flow = flow
    self.trainer = self.TrainerClass(self.params, self.state, self.loss, optimizer)
    self.tester = self.TesterClass(self.params, self.state, self.loss)

    self.test_eval_times = jnp.array([])

  @property
  def train_losses(self):
    return self.trainer.losses

  @property
  def n_train_steps(self):
    return self.train_losses.size

  @property
  def test_losses(self):
    return self.tester.losses

  @property
  def params(self):
    return self.flow.params

  @params.setter
  def params(self, val):
    self.flow.params = val

  @property
  def state(self):
    return self.flow.state

  @state.setter
  def state(self, val):
    self.flow.state = val

  def update_params_and_state_from_trainer(self):
    # Don't think this should be needed, but keep it in just in case
    self.params, self.state = self.trainer.params, self.trainer.state
    self.tester.params, self.tester.state = self.params, self.state

  def update_params_and_state_from_tester(self):
    self.params, self.state = self.tester.params, self.tester.state
    self.trainer.params, self.trainer.state = self.params, self.state

  @property
  def TrainerClass(cls):
    return Trainer

  @property
  def TesterClass(cls):
    return Tester

  @abstractmethod
  def loss(self, params, state, key, inputs, **kwargs):
    pass

  @abstractmethod
  def summarize_train_out(self, out):
    pass

  @abstractmethod
  def summarize_test_out(self, out):
    pass

  def save(self, path: str=None):

    save_items = {"params": self.params,
                  "state": self.state,
                  "test_eval_times": self.test_eval_times}

    train_items = self.trainer.save_items()
    test_items = self.tester.save_items()

    save_items.update(train_items)
    save_items.update(test_items)

    util.save_pytree(save_items, path, overwrite=True)

  def load(self, path: str=None):
    loaded_items = util.load_pytree(path)
    self.params = loaded_items["params"]
    self.state = loaded_items["state"]
    self.test_eval_times = loaded_items["test_eval_times"]

    self.trainer.load_items(loaded_items)
    self.tester.load_items(loaded_items)

  def grad_step(self,
                key: PRNGKey,
                inputs: Mapping[str, jnp.ndarray],
                **kwargs):
    out = self.trainer.step(key, inputs, **kwargs)
    self.update_params_and_state_from_trainer()
    return out

  def grad_step_for_loop(self,
                         key: PRNGKey,
                         inputs: Mapping[str, jnp.ndarray],
                         **kwargs):
    out = self.trainer.step_for_loop(key, inputs, **kwargs)
    self.update_params_and_state_from_trainer()
    return out

  def grad_step_scan_loop(self,
                          key: PRNGKey,
                          inputs: Mapping[str, jnp.ndarray],
                          **kwargs):
    out = self.trainer.step_scan_loop(key, inputs, **kwargs)
    self.update_params_and_state_from_trainer()
    return out

  def test_step(self,
                key: PRNGKey,
                inputs: Mapping[str, jnp.ndarray],
                **kwargs):
    out = self.tester.step(key, inputs, **kwargs)
    self.update_params_and_state_from_tester()
    return out

  def test_step_for_loop(self,
                         key: PRNGKey,
                         inputs: Mapping[str, jnp.ndarray],
                         **kwargs):
    out = self.tester.step_for_loop(key, inputs, **kwargs)
    self.update_params_and_state_from_tester()
    return out

  def test_step_scan_loop(self,
                          key: PRNGKey,
                          inputs: Mapping[str, jnp.ndarray],
                          **kwargs):
    out = self.tester.step_scan_loop(key, inputs, **kwargs)
    self.update_params_and_state_from_tester()
    return out

  def evaluate_test_set(self,
                        key: PRNGKey,
                        input_iterator,
                        **kwargs):

    outs = []
    i = 0

    try:
      while True:
        key, test_key = random.split(key, 2)
        inputs = next(input_iterator)
        test_out = self.test_step_scan_loop(key, inputs, update=False, **kwargs)
        outs.append(test_out)
        i += 1

    except StopIteration:
      pass

    def concat(*args):
      try:
        return jnp.concatenate(args, axis=0)
      except ValueError:
        return jnp.array(args)

    # Mark when we evaluated the test set
    self.test_eval_times = jnp.hstack([self.test_eval_times, self.n_train_steps])

    outs = jax.tree_multimap(concat, *outs)

    # Condense the outputs over the entire test set
    out = jax.tree_map(jnp.mean, outs)
    self.tester.update_outputs(out)

    return outs

################################################################################################################

class MaximumLikelihoodTrainer(FlowTrainer):
  """ Convenience class for training a flow with maximum likelihood.

      Args:
          flow     - A Flow object.
          clip     - How much to clip gradients.  This is crucial for stable training!
          warmup   - How much to warm up the learning rate.
          lr_decay - Learning rate decay.
          lr       - Max learning rate.
  """
  def __init__(self,
               flow: Flow,
               optimizer: GradientTransformation=None,
               image: bool=False,
               **kwargs):
    super().__init__(flow, optimizer=optimizer, **kwargs)
    self.image = image

  @property
  def accumulate_args(self):
    return ["log_pz", "log_det"]

  def loss(self, params, state, key, inputs, **kwargs):
    outputs, updated_state = self.flow._apply_fun(params, state, key, inputs, accumulate=self.accumulate_args, **kwargs)
    loss = outputs.get("log_pz", 0.0) + outputs.get("log_det", 0.0)

    aux = ()
    return -loss.mean(), (aux, updated_state)

  def summarize_train_out(self, out):
    log_px = out.loss.mean()
    if self.image:
      log_px = self.flow.to_bits_per_dim(log_px)
    return f"loss: {log_px:.2f}"

  def summarize_test_out(self, out):
    log_px = out.loss.mean()
    if self.image:
      log_px = self.flow.to_bits_per_dim(log_px)
    return f"loss: {log_px:.2f}"

################################################################################################################

class JointClassificationTrainer(FlowTrainer):

  def __init__(self,
               flow: Flow,
               optimizer: GradientTransformation=None,
               image: bool=False,
               **kwargs):
    super().__init__(flow, optimizer=optimizer, **kwargs)
    self.image = image

  @property
  def accumulate_args(self):
    return ["log_pz", "log_det", "log_pygx"]

  def loss(self, params, state, key, inputs, **kwargs):
    outputs, updated_state = self.flow._apply_fun(params, state, key, inputs, accumulate=self.accumulate_args, **kwargs)

    # TODO: Stop grouping p(x|y)p(y) into the prior and instead pass them out separately
    log_pyax = outputs.get("log_pz", 0.0) + outputs.get("log_det", 0.0)

    # Compute the data log likelihood
    log_px = log_pyax - outputs.get("log_pygx", 0.0)

    # Compute the accuracy
    y_one_hot = inputs["y"]
    acc = (outputs["prediction_one_hot"]*y_one_hot).sum(axis=-1).mean()

    aux = (acc, -log_px.mean())
    return -log_pyax.mean(), (aux, updated_state)

  def summarize_train_out(self, out):
    loss = out.loss.mean()
    accuracy, nll = jax.tree_map(jnp.mean, out.aux)
    if self.image:
      nll = self.flow.to_bits_per_dim(nll)
    return f"loss: {loss:.2f}, nll: {nll:.2f}, acc: {accuracy:.2f}"

  def summarize_test_out(self, out):
    loss = out.loss.mean()
    accuracy, nll = jax.tree_map(jnp.mean, out.aux)
    if self.image:
      nll = self.flow.to_bits_per_dim(nll)
    return f"loss: {loss:.2f}, nll: {nll:.2f}, acc: {accuracy:.2f}"
