from functools import partial
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
import nux.util as util
from nux.training.trainer import Trainer
from nux.training.tester import Tester
from typing import Optional, Mapping, Callable, Sequence, Tuple, Any
from haiku._src.typing import Params, State, PRNGKey

__all__ = ["GenerativeModel"]

################################################################################################################

class GenerativeModel():
  """ Convenience class for training a generative flow model.

      Args:
          flow     - A Flow object.
          clip     - How much to clip gradients.  This is crucial for stable training!
          warmup   - How much to warm up the learning rate.
          lr_decay - Learning rate decay.
          lr       - Max learning rate.
  """
  def __init__(self,
               flow: Any,
               params: Params,
               state: State,
               loss_fun: Callable,
               test_aggregate_fun: Callable=None,
               **kwargs):
    self.flow   = flow
    self.params = params
    self.state  = state

    loss_fun = partial(loss_fun, self.flow.apply)
    self.trainer = Trainer(loss_fun, self.params, **kwargs)

    if test_aggregate_fun is None:
      def test_aggregate_fun(inputs, outputs):
        log_px = jnp.mean(outputs.get("log_pz", 0.0) + outputs["log_det"])
        return log_px
    self.tester = Tester(self.flow.apply, aggregate_fun=test_aggregate_fun)

  #############################################################################

  @property
  def apply(self):
    return self.flow.apply

  #############################################################################

  @property
  def opt_state(self):
    return self.trainer.opt_state

  @opt_state.setter
  def opt_state(self, val):
    self.trainer.opt_state = val

  #############################################################################

  def grad_step(self,
                key: PRNGKey,
                inputs: Mapping[str, jnp.ndarray],
                **kwargs
    ) -> Tuple[float, Mapping[str, jnp.ndarray]]:
    loss, outputs, params, state = self.trainer.grad_step(key, inputs, self.params, self.state, **kwargs)
    self.params = params
    self.state = state
    return loss, outputs

  def multi_grad_step(self,
                      key: PRNGKey,
                      inputs: Mapping[str, jnp.ndarray],
                      store_outputs: bool=False
    ) -> Tuple[float, Mapping[str, jnp.ndarray]]:

    # This function expects doubly batched inputs!!
    (train_losses, outputs), params, state = self.trainer.multi_grad_step(key, inputs, self.params, self.state, store_outputs=store_outputs)
    self.params = params
    self.state = state
    return train_losses, outputs

  #############################################################################

  def multi_test_step(self,
                      key: PRNGKey,
                      inputs: Mapping[str, jnp.ndarray],
    ) -> Mapping[str, jnp.ndarray]:

    # This function expects doubly batched inputs!!
    test_metrics = self.tester.multi_eval_step(key, inputs, self.params, self.state)

    # Aggregate over the outer batch
    test_metrics = jax.tree_map(lambda x: x.mean(), test_metrics)
    return test_metrics

  #############################################################################

  def forward_apply(self,
                    key: PRNGKey,
                    inputs: Mapping[str, jnp.ndarray],
                    **kwargs
    ) -> Mapping[str, jnp.ndarray]:
    outputs, _ = self.apply(self.params, self.state, key, inputs, **kwargs)
    return outputs

  #############################################################################

  def sample(self,
             key: PRNGKey,
             n_samples: int,
             latent_shape: Sequence[int],
             **kwargs
    ) -> Mapping[str, jnp.ndarray]:
    # dummy_z is a placeholder with the shapes we'll use when we sample in the prior.
    dummy_z = jnp.zeros((n_samples,) + latent_shape)
    outputs, _ = self.apply(self.params, self.state, key, {"x": dummy_z}, sample=True, **kwargs)
    return outputs

  #############################################################################

  def inverse(self,
              key: PRNGKey,
              inputs: Mapping[str, jnp.ndarray],
              **kwargs
    ) -> Mapping[str, jnp.ndarray]:
    outputs, _ = self.apply(self.params, self.state, key, inputs, sample=True, reconstruction=True, **kwargs)
    return outputs

  #############################################################################

  def save_model(self, path: str=None):
    save_items = {"params": self.params,
                  "state": self.state,
                  "opt_state": self.trainer.opt_state,
                  "train_all_outputs": self.trainer.all_outputs,
                  "train_losses": self.trainer.losses,
                  "test_losses": self.tester.losses,
                  "training_steps": self.trainer.training_steps}
    util.save_pytree(save_items, path, overwrite=True)

  def load_model(self, path: str=None):
    loaded_items = util.load_pytree(path)
    self.params = loaded_items["params"]
    self.state = loaded_items["state"]
    self.trainer.opt_state = loaded_items["opt_state"]
    self.trainer.all_outputs = loaded_items.get("train_all_outputs", ())
    self.trainer.losses = loaded_items["train_losses"]
    self.tester.losses = loaded_items["test_losses"]
    self.trainer.training_steps = loaded_items["training_steps"]

  #############################################################################
