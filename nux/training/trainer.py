import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
import nux.util as util
import os
import pathlib
import optax
import pandas as pd

class Trainer():

  def __init__(self, args, init_key=None, train_key=None, eval_key=None, loss_scale=1.0, use_pmap=False):
    self.args = args

    # Make sure that the save path exists
    self.ensure_path(args.save_path)

    self.train_metrics = None # dict(losses, aux, grad_summary)
    self.test_metrics = None  # dict(losses, aux)

    # Keys for training and initialization
    self.init_key = random.PRNGKey(args.init_key_seed) if init_key is None else init_key
    self.train_key = random.PRNGKey(args.train_key_seed) if train_key is None else train_key
    self.eval_key = random.PRNGKey(args.eval_key_seed) if eval_key is None else eval_key

    self.n_train_steps = 0
    self.test_eval_times = None

    self.loss_scale = loss_scale

    self.apply_updates = jax.jit(optax.apply_updates)

    self.n_models = args.n_models
    self.model_mask = jnp.ones(args.n_models).astype(bool)

    self.data_augmentation = args.data_augmentation

    if use_pmap:
      self.map = jax.pmap
    else:
      self.map = jax.vmap

  @property
  def train_losses(self):
    return self.train_metrics["losses"]

  @property
  def test_losses(self):
    return self.test_metrics["losses"]

  def ensure_path(self, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

  def load_data(self, classification=False):
    from datasets import get_dataset
    train_ds, get_test_ds = get_dataset(self.args.dataset,
                                        self.args.train_batch_size,
                                        self.args.train_n_batches,
                                        self.args.test_batch_size,
                                        test_n_batches=None,
                                        quantize_bits=self.args.quantize_bits,
                                        classification=classification,
                                        label_keep_percent=self.args.label_keep_percent,
                                        random_label_percent=self.args.random_label_percent,
                                        train_split=self.args.train_split,
                                        data_augmentation=self.data_augmentation)
    return train_ds, get_test_ds

  @property
  def models_path(self):
    root = self.args.save_path
    model_path = os.path.join(root, "models")
    self.ensure_path(model_path)
    return model_path

  @property
  def initial_model_path(self):
    return os.path.join(self.models_path, "initial.pickle")

  @property
  def best_model_path(self):
    return os.path.join(self.models_path, "best.pickle")

  @property
  def most_recent_model_path(self):
    return os.path.join(self.models_path, "most_recent.pickle")

  def initialize(self, flow, loss_fun):
    self.flow = flow
    self.loss_fun = loss_fun
    self.valgrad = jax.jit(jax.value_and_grad(partial(loss_fun, is_training=True), has_aux=True))

    # Create the optimizer
    chain = []
    chain.append(optax.clip_by_global_norm(self.args.clip))
    chain.append(optax.scale_by_belief())

    # Warm up the learning rate
    warmup_schedule = partial(util.linear_warmup_lr_schedule, warmup=self.args.warmup, lr_decay=1.0, lr=-self.args.lr)
    chain.append(optax.scale_by_schedule(warmup_schedule))

    # Cosine decay schedule
    cosine_lr = optax.cosine_decay_schedule(init_value=1.0, decay_steps=self.args.cosine_decay_steps, alpha=self.args.cosine_decay_amount)
    chain.append(optax.scale_by_schedule(cosine_lr))

    # Initialize the optimizer
    self.opt_init, self.opt_update = optax.chain(*chain)
    self.opt_update = jax.jit(self.opt_update)

  def load_from_file(self):
    if self.args.best_model == True:
      # Load the best model
      model_path = self.best_model_path
    else:
      # Load the most recent model
      model_path = self.most_recent_model_path

    if os.path.exists(model_path) == False:
      assert 0, f"{model_path} does not exist!"

    self.load(model_path)

    # assert self.n_models == self.args.n_models

  def initialize_optimizer(self, params):
    self.params = params
    self.opt_state = self.map(self.opt_init)(self.params)

    # Save off the initial parameters
    self.save(self.initial_model_path)

  def get_model_mask(self, top_k):
    best_losses = pd.read_csv(self.best_losses_path)
    best_test_loss = best_losses["test"]
    self.model_mask = best_test_loss.argsort().replace(-1, jnp.nan) < top_k

  @property
  def n_params(self):
    return jax.flatten_util.ravel_pytree(self.params)[0].size // self.n_models

  @property
  def train_state(self):
    return (self.params, self.opt_state)

  @train_state.setter
  def train_state(self, val):
    self.params, self.opt_state = val

  def grad_hook(self, grad):
    return ()

  def grad_step(self, carry, inputs):
    params, opt_state = carry

    # Evaluate the gradient
    (train_loss, aux), grad = self.valgrad(params, inputs)
    grad_summary = self.grad_hook(grad)

    # Take a gradient step
    updates, opt_state = self.opt_update(grad, opt_state, params)
    params = self.apply_updates(params, updates)

    return (params, opt_state), (train_loss, aux, grad_summary)

  @property
  def train_scan_loop(self):
    if hasattr(self, "_train_scan_loop") == False:
      self._train_scan_loop = partial(jax.lax.scan, self.grad_step)
      if self.data_augmentation:
        self._train_scan_loop = self.map(self._train_scan_loop, in_axes=(0, {"x": None, "x_aug": None, "rng_key": 0}))
      else:
        self._train_scan_loop = self.map(self._train_scan_loop, in_axes=(0, {"x": None, "rng_key": 0}))
      self._train_scan_loop = jax.jit(self._train_scan_loop)
    return self._train_scan_loop

  def train(self, train_iter, train_key=None):
    if train_key is None:
      self.train_key, rng_key = random.split(self.train_key, 2)
    else:
      rng_key = train_key

    data = next(train_iter)
    x = data["x"]
    n_grad_steps = x.shape[0]

    # Get a key for each model
    keys = random.split(rng_key, self.n_models*n_grad_steps)
    keys = keys.reshape((self.n_models, n_grad_steps, -1))

    # Train using the scan loop
    inputs = dict(x=x, rng_key=keys)
    if self.data_augmentation:
      inputs["x_aug"] = data["x_aug"]

    self.train_state, (train_losses, aux, grad_summary) = self.train_scan_loop(self.train_state, inputs)
    self.n_train_steps += n_grad_steps

    # Update the training losses and auxiliary values from the loss function
    train_metrics = dict(losses=train_losses, aux=aux, grad_summary=grad_summary)
    self.shapes_before = util.tree_shapes(self.train_metrics)
    self.train_metrics = util.tree_concat(self.train_metrics, train_metrics, axis=1)
    self.shapes_after = util.tree_shapes(self.train_metrics)
    return self.train_losses.mean()

  @property
  def jitted_test(self):
    if hasattr(self, "_jitted_test") == False:
      loss_fun = partial(self.loss_fun, is_training=False)
      loss_fun = self.map(loss_fun, in_axes=(0, {"x": None, "rng_key": 0}))
      self._jitted_test = jax.jit(loss_fun)
    return self._jitted_test

  def test(self, get_test_iter):
    test_iter = get_test_iter()

    eval_key = self.eval_key

    test_metrics = None
    total_examples = 0
    try:
      i = -1
      while True:
        i += 1
        x = next(test_iter)["x"]
        total_examples += x.shape[0]
        eval_key, rng_key = random.split(eval_key, 2)
        keys = random.split(rng_key, self.n_models)
        inputs = dict(x=x, rng_key=keys)

        # Evaluate a batch of the test set
        test_loss, aux = self.jitted_test(self.params, inputs)

        # Save off the metrics
        t_metrics = dict(losses=test_loss, aux=aux)
        t_metrics = jax.tree_map(lambda x: x[:,None], t_metrics)
        test_metrics_old = test_metrics
        test_metrics = util.tree_concat(test_metrics, t_metrics, axis=1)

    except StopIteration:
      pass

    del test_iter

    # Average the test metrics
    test_metrics = jax.tree_map(lambda x: jnp.mean(x, axis=1), test_metrics)
    test_metrics = jax.tree_map(lambda x: x[:,None], test_metrics)
    self.test_metrics = util.tree_concat(self.test_metrics, test_metrics, axis=1)

    # Mark when we evaluated the test set
    self.test_eval_times = util.tree_hstack(self.test_eval_times, jnp.array(self.n_train_steps))

  @property
  def losses_path(self):
    root = self.args.save_path
    losses_path = os.path.join(root, "losses")
    self.ensure_path(losses_path)
    return losses_path

  @property
  def train_losses_path(self):
    return os.path.join(self.losses_path, "train.csv")

  @property
  def test_losses_path(self):
    return os.path.join(self.losses_path, "test.csv")

  @property
  def best_losses_path(self):
    return os.path.join(self.losses_path, "best_model.csv")

  @property
  def samples_path(self):
    root = self.args.save_path
    samples_path = os.path.join(root, "samples")
    self.ensure_path(samples_path)
    return samples_path

  @property
  def training_curves_path(self):
    root = self.args.save_path
    training_curves_path = os.path.join(root, "training_curves")
    self.ensure_path(training_curves_path)
    return training_curves_path

  def get_internal_state(self):
    items = dict(params=self.params,
                 opt_state=self.opt_state,
                 train_metrics=self.train_metrics,
                 test_metrics=self.test_metrics,
                 init_key=self.init_key,
                 train_key=self.train_key,
                 eval_key=self.eval_key,
                 n_train_steps=self.n_train_steps,
                 test_eval_times=self.test_eval_times,
                 n_models=self.n_models)
    return items

  def set_internal_state(self, items):
    self.params          = items["params"]
    self.opt_state       = items["opt_state"]
    self.train_metrics   = items["train_metrics"]
    self.test_metrics    = items["test_metrics"]
    self.init_key        = items["init_key"]
    self.train_key       = items["train_key"]
    self.eval_key        = items["eval_key"]
    self.n_train_steps   = items["n_train_steps"]
    self.test_eval_times = items["test_eval_times"]
    self.n_models        = items["n_models"]

  def save(self, path):
    # Save everything
    util.save_pytree(self.get_internal_state(), path, overwrite=True)

  def load(self, path):
    items = util.load_pytree(path)
    self.set_internal_state(items)

  def checkpoint(self):
    # Load the losses into a dataframe
    train_losses = pd.DataFrame(self.train_losses.T)
    test_losses  = pd.DataFrame(self.test_losses.T, index=self.test_eval_times)

    # Save the losses
    train_losses.to_csv(self.train_losses_path, index=False, header=False)
    test_losses.to_csv(self.test_losses_path, header=False)

    # Load the losses for the best model
    if os.path.exists(self.best_losses_path):
      best_losses = pd.read_csv(self.best_losses_path)
      best_test_loss = best_losses["test"]
      best_test_loss = best_test_loss.to_numpy().min()

      # Is the current model the best?
      current_is_best = best_test_loss > self.test_losses[:,-1].min()

      # If we are running a new experiment, then override the best model
      best_index = best_losses["index"]
      best_index = best_index.to_numpy()[0]
      if best_index > self.test_eval_times[-1]:
        current_is_best = True
    else:
      current_is_best = True

    # If the current model is the best model, then save it
    if current_is_best:

      # Save the model
      self.save(self.best_model_path)

      # Save the losses
      n_params = self.n_params
      best_test = test_losses.iloc[-1,:]
      best_train = train_losses.iloc[-1,:]
      best_idx = self.test_eval_times[-1]
      df = pd.DataFrame({"train": best_train,
                         "test": best_test,
                         "index": best_idx*jnp.ones(best_train.shape[0]),
                         "n_params": n_params*jnp.ones(best_train.shape[0])})
      df.to_csv(self.best_losses_path, index=False)

    # Save the current model
    self.save(self.most_recent_model_path)

  def save_training_plot(self):
    import matplotlib.pyplot as plt
    import numpy as np

    train_dfs = []
    test_dfs = []

    for model_index in range(self.n_models):
      train_aux = jax.tree_map(lambda x: np.array(x[model_index]), self.train_metrics["aux"])
      test_aux = jax.tree_map(lambda x: np.array(x[model_index]), self.test_metrics["aux"])

      train_df = pd.DataFrame(train_aux)
      train_df = train_df.ewm(alpha=0.1).mean()
      test_df = pd.DataFrame(test_aux, index=self.test_eval_times)

      train_dfs.append(train_df)
      test_dfs.append(test_df)

    train_df = pd.concat(train_dfs, axis=1, keys=list(range(self.n_models)))
    test_df = pd.concat(test_dfs, axis=1, keys=list(range(self.n_models)))

    def make_plot(col, M=2000):
      plot_test = True if col in [name for _, name in test_df.columns] else False

      idx_slice = pd.IndexSlice[:,col]

      train_color = "blue"
      test_color = "red"

      train_slice, test_slice = train_df.loc[:,idx_slice], test_df.loc[:,idx_slice]
      if plot_test == False:
        test_slice = None

      fig, (ax1, ax2) = plt.subplots(1, 2)
      train_slice.plot(ax=ax1, alpha=0.1, linewidth=1, color=train_color, legend=False)
      if plot_test:
        test_slice.plot(ax=ax1, alpha=0.1, linewidth=1, color=test_color, legend=False)

      train_slice.iloc[-M:,:].plot(ax=ax2, alpha=0.1, linewidth=1, color=train_color, legend=False)
      if plot_test:
        mask = test_slice.index >= train_df.shape[0] - M
        test_slice[mask].plot(ax=ax2, alpha=0.1, linewidth=1, color=test_color, legend=False)

      # # Only plot the non outliers
      # def bounds(df):
      #   x = jnp.array(df.to_numpy().ravel())
      #   z_score = (x - jnp.nanmean(x))/jnp.nanstd(x)
      #   mask = jnp.abs(z_score) < 2
      #   x_filter = x[mask]
      #   return jnp.nanmin(x_filter), jnp.nanmax(x_filter)

      def bounds(df):
        # Get the models that are in the top 50%
        latest = df.ewm(alpha=0.1).mean().iloc[-1,:]
        latest = latest[~latest.isna()]
        # K = max(1, latest.shape[0]//2)
        K = max(1, latest.shape[0])
        idx = latest.argsort()[::-1]

        # Select the top K columns
        top_k = df.loc[:,idx[:K]]

        # Find the bounds on the top models
        x = jnp.array(top_k.to_numpy().ravel())
        z_score = (x - jnp.nanmean(x))/jnp.nanstd(x)
        mask = jnp.abs(z_score) < 2
        # x_filter = x[mask]
        x_filter = x
        # return jnp.min(x_filter), jnp.max(x_filter)
        return jnp.nanmin(x_filter), jnp.nanmax(x_filter)

      def set_axis_bounds(ax, train, test):
        train_min, train_max = bounds(train)
        return ax.set_ylim(train_min, train_max)
        if test is None:
          return ax.set_ylim(train_min, train_max)
        test_min, test_max = bounds(test)
        y_min, y_max = min(train_min, test_min), max(train_max, test_max)
        ax.set_ylim(y_min, y_max)

      set_axis_bounds(ax1, train_slice, test_slice)
      set_axis_bounds(ax2, train_slice.iloc[-M:,:], test_slice)

      if hasattr(self, "plot_title"):
        fig.suptitle(self.plot_title)

      plot_save_path = os.path.join(self.training_curves_path, f"{col}.png")
      plt.savefig(plot_save_path)
      plt.close()

    for col in self.train_metrics["aux"].keys():
      make_plot(col)

    del train_df
    del test_df

  def apply_fun(self, x, params, rng_key):

    def _apply_fun(x, params, rng_key):
      z, log_px = self.flow(x, params=params, rng_key=rng_key)
      return z, log_px

    keys = random.split(rng_key, self.n_models)
    return self.map(_apply_fun, in_axes=(None, 0, 0))(x, params, keys)

  @property
  def jitted_apply_fun(self):
    if hasattr(self, "_jitted_apply_fun") == False:
      self._jitted_apply_fun = jax.jit(self.apply_fun)
    return self._jitted_apply_fun

  def sample(self, rng_key, params, n_samples):
    def _sample(params, rng_key):
      z = jnp.zeros((n_samples, *self.latent_shape))
      x, _ = self.flow(z, params=params, rng_key=rng_key, inverse=True)
      return x
    keys = random.split(rng_key, self.n_models)
    return self.map(_sample)(params, keys)

  @property
  def jitted_sample(self):
    if hasattr(self, "_jitted_sample") == False:
      self._jitted_sample = jax.jit(self.sample, static_argnums=(2,))
    return self._jitted_sample


  def reconstruct(self, z, rng_key, params, n_samples):
    def _reconstr(z, params, rng_key):
      x, _ = self.flow(z, params=params, rng_key=rng_key, inverse=True)
      return x
    keys = random.split(rng_key, self.n_models)
    return self.map(_reconstr)(z, params, keys)

  @property
  def jitted_reconstruct(self):
    if hasattr(self, "_jitted_reconstruct") == False:
      self._jitted_reconstruct = jax.jit(self.reconstruct, static_argnums=(3,))
    return self._jitted_reconstruct

  def save_samples(self, train_ds):
    pass
