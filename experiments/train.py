import jax
from jax import random, jit
import jax.numpy as jnp
import nux
from functools import partial
import os
import tqdm
from experiments.datasets import get_dataset
import pathlib
import matplotlib.pyplot as plt

def initialize_trainer(flow,
                       clip=15.0,
                       lr=5e-4,
                       warmup=2000,
                       cosine_decay_steps=None,
                       save_path=None,
                       retrain=False,
                       classification=False,
                       trainer_fun=None):

  if trainer_fun is None:
    if classification == False:
      trainer = nux.MaximumLikelihoodTrainer(flow,
                                             clip=clip,
                                             lr=lr,
                                             warmup=warmup,
                                             cosine_decay_steps=cosine_decay_steps)
    else:
      trainer = nux.JointClassificationTrainer(flow,
                                               clip=clip,
                                               lr=lr,
                                               warmup=warmup,
                                               cosine_decay_steps=cosine_decay_steps)
  else:
    trainer = trainer_fun(flow)

  model_save_path = os.path.join(save_path, "model.pickle")
  if retrain == False and os.path.exists(model_save_path):
    trainer.load(model_save_path)

  return trainer

################################################################################################################

def train_model(create_model,
                args,
                classification=False,
                image=False,
                trainer_fun=None):
  assert args.save_path.endswith(".pickle") == False

  init_key  = random.PRNGKey(args.init_key_seed)
  train_key = random.PRNGKey(args.train_key_seed)
  eval_key  = random.PRNGKey(args.eval_key_seed)

  train_ds, get_test_ds = get_dataset(args.dataset,
                                      args.batch_size,
                                      args.n_batches,
                                      args.test_batch_size,
                                      args.test_n_batches,
                                      quantize_bits=args.quantize_bits,
                                      classification=classification,
                                      label_keep_percent=args.label_keep_percent,
                                      random_label_percent=args.random_label_percent)

  doubly_batched_inputs = next(train_ds)

  inputs = jax.tree_map(lambda x: x[0], doubly_batched_inputs)
  flow = nux.Flow(create_model, init_key, inputs, batch_axes=(0,))

  print("n_params", flow.n_params)

  # Make sure that the save_path folder exists
  pathlib.Path(args.save_path).mkdir(parents=True, exist_ok=True)

  trainer = initialize_trainer(flow,
                               clip=args.clip,
                               lr=args.lr,
                               warmup=args.warmup,
                               cosine_decay_steps=args.cosine_decay_steps,
                               save_path=args.save_path,
                               retrain=args.retrain,
                               classification=classification,
                               trainer_fun=trainer_fun)

  return train(train_key,
               eval_key,
               trainer,
               train_ds,
               get_test_ds,
               max_iters=args.max_iters,
               save_path=args.save_path,
               eval_interval=args.eval_interval,
               bits_per_dim=image,
               classification=classification)

################################################################################################################

def train(train_key,
          eval_key,
          trainer,
          train_ds,
          get_test_ds,
          max_iters=1e7,
          save_path=None,
          eval_interval=None,
          classification=False,
          bits_per_dim=False):

  pbar = tqdm.tqdm(jnp.arange(int(max_iters)))
  for i in pbar:

    train_key, key = random.split(train_key, 2)

    # Perform a bunch of gradient steps
    inputs = next(train_ds)
    res = trainer.grad_step_scan_loop(key, inputs, bits_per_dim=bits_per_dim)
    pbar.set_description(trainer.summarize_losses_and_aux(res))

    # Stop if things have diverged
    if jnp.isnan(trainer.train_losses[-1]):
      assert 0

    # Evaluate the test set
    if eval_interval and \
       (eval_interval == -1 or \
        trainer.n_train_steps//eval_interval >= len(trainer.test_losses)):

      test_ds = get_test_ds()

      res = trainer.evaluate_test(eval_key, test_ds, bits_per_dim=bits_per_dim)
      print("test", trainer.summarize_losses_and_aux(res))

      del test_ds

    # Pull some samples
    fig, axes = plt.subplots(4, 4); axes = axes.ravel()
    samples = trainer.flow.sample(eval_key, n_samples=16, generate_image=True)
    for k, ax in enumerate(axes):
      ax.imshow(samples["image"][k].squeeze())

    plot_save_path = os.path.join(save_path, f"sample_{trainer.n_train_steps}.png")
    plt.savefig(plot_save_path)

    plt.close()

    # Save the model
    if save_path is not None:
      model_save_path = os.path.join(save_path, f"model.pickle")
      trainer.save(model_save_path)