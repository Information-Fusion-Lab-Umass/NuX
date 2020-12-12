from functools import partial
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
from nux.internal.layer import Flow
import nux.util as util
from typing import Optional, Mapping, Callable, Sequence, Tuple, Any
from haiku._src.typing import Params, State, PRNGKey
import optax
from optax._src import transform
GradientTransformation = transform.GradientTransformation

__all__ = ["MaximumLikelihoodTrainer"]

################################################################################################################

class MaximumLikelihoodTrainer():
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
               **kwargs):
    self.flow = flow

    # Get the optimizer
    if optimizer is None:
      opt_init, opt_update = self.build_optimizer(**kwargs)
    else:
      opt_init, opt_update = optimizer

    # Initialize the optimizer state
    self.opt_update = jit(opt_update)
    self.opt_state = opt_init(self.flow.params)
    self.apply_updates = jit(optax.apply_updates)

    # Build the value and grad function
    self.valgrad = jax.value_and_grad(self.nll, has_aux=True)
    self.valgrad = jit(self.valgrad)

    self.train_losses = jnp.array([])
    self.test_losses = {}

  @property
  def n_train_steps(self):
    return self.train_losses.size

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
      chain.append(optax.scale_by_schedule(partial(util.linear_lr, lr=-lr)))

    if cosine_decay_steps and cosine_decay_steps > 0:
      cosine_lr = optax.cosine_decay_schedule(init_value=1.0, decay_steps=cosine_decay_steps, alpha=0.0)
      chain.append(optax.scale_by_schedule(cosine_lr))

    if clip and clip > 0:
      chain.append(optax.clip(clip))

    return optax.chain(*chain)

  #############################################################################

  def nll(self, params, state, key, inputs, **kwargs):
    outputs, updated_state = self.flow._apply_fun(params, state, key, inputs, **kwargs)
    log_px = outputs.get("log_pz", 0.0) + outputs.get("log_det", 0.0)
    return -log_px.mean(), updated_state

  #############################################################################

  def scan_grad_step(self, carry, scan_inputs, **kwargs):
    params, state, opt_state = carry
    key, inputs = scan_inputs

    # Take a gradient step
    (train_loss, state), grad = self.valgrad(params, state, key, inputs, **kwargs)

    # Update the parameters and optimizer state
    updates, opt_state = self.opt_update(grad, opt_state, params)
    params = self.apply_updates(params, updates)

    return (params, state, opt_state), train_loss

  def grad_step(self,
                key: PRNGKey,
                inputs: Mapping[str, jnp.ndarray],
                **kwargs):
    carry = (self.flow.params, self.flow.state, self.opt_state)
    carry, train_loss = self.scan_grad_step(carry, (key, inputs), **kwargs)
    self.flow.params, self.flow.state, self.opt_state = carry

    self.train_losses = jnp.hstack([self.train_losses, train_loss])
    return train_loss

  def grad_step_for_loop(self,
                         key: PRNGKey,
                         inputs: Mapping[str, jnp.ndarray],
                         **kwargs):
    if len(inputs["x"].shape) == len(self.flow.data_shape):
      assert 0, "Expect a batched or doubly-batched input"

    # Get the inputs for the for loop
    n_iters = inputs["x"].shape[0]
    keys = random.split(key, n_iters)
    scan_inputs = (keys, inputs)
    carry = (self.flow.params, self.flow.state, self.opt_state)

    train_losses = []
    for i, key in enumerate(keys):
      _inputs = jax.tree_map(lambda x: x[i], inputs)
      carry, train_loss = self.scan_grad_step(carry, (key, _inputs), **kwargs)
      train_losses.append(train_loss)
      self.flow.params, self.flow.state, self.opt_state = carry

    train_losses = jnp.array(train_losses)
    self.train_losses = jnp.hstack([self.train_losses, train_losses])
    return train_losses

  def grad_step_scan_loop(self,
                          key: PRNGKey,
                          inputs: Mapping[str, jnp.ndarray],
                          **kwargs):
    if len(inputs["x"].shape) == len(self.flow.data_shape):
      assert 0, "Expect a batched or doubly-batched input"

    # Get the inputs for the scan loop
    n_iters = inputs["x"].shape[0]
    keys = random.split(key, n_iters)
    scan_inputs = (keys, inputs)
    carry = (self.flow.params, self.flow.state, self.opt_state)

    # Run the training steps
    carry, train_losses = jax.lax.scan(self.scan_grad_step, carry, scan_inputs)
    self.flow.params, self.flow.state, self.opt_state = carry
    self.train_losses = jnp.hstack([self.train_losses, train_losses])
    return train_losses

  #############################################################################

  def evaluate_test(self,
                    key: PRNGKey,
                    input_iterator,
                    **kwargs):

    sum_log_px = 0.0
    total_examples = 0

    try:
      while True:
        key, test_key = random.split(key, 2)
        inputs = next(input_iterator)
        outputs = self.flow.scan_apply(test_key, inputs, **kwargs)

        # Accumulate the total sum of the log likelihoods in case the batch sizes differ
        sum_log_px += outputs["log_px"].sum()
        n_examples_in_batch = util.list_prod(self.flow.get_batch_shape(inputs))
        total_examples += n_examples_in_batch

    except StopIteration:
      pass

    nll = -sum_log_px/total_examples
    self.test_losses[self.n_train_steps] = nll
    return nll

  #############################################################################

  def save(self, path: str=None):
    save_items = {"params": self.flow.params,
                  "state": self.flow.state,
                  "opt_state": self.trainer.opt_state,
                  "train_losses": self.train_losses,
                  "test_losses": self.test_losses}
    util.save_pytree(save_items, path, overwrite=True)

  def load(self, path: str=None):
    loaded_items = util.load_pytree(path)
    self.flow.params = loaded_items["params"]
    self.flow.state = loaded_items["state"]
    self.trainer.opt_state = loaded_items["opt_state"]
    self.trainer.losses = loaded_items["train_losses"]
    self.tester.losses = loaded_items["test_losses"]

  #############################################################################
