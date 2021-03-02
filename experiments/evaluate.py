import jax
from jax import random, jit
import jax.numpy as jnp
import nux
from functools import partial
from experiments.train import initialize_trainer
from experiments.datasets import get_dataset
import matplotlib.pyplot as plt
import nux.util as util

################################################################################################################

def evaluate_2d_model(create_model,
                      args,
                      classification=False):
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
                                      label_keep_percent=1.0,
                                      random_label_percent=0.0)

  doubly_batched_inputs = next(train_ds)
  inputs = {"x": doubly_batched_inputs["x"][0]}

  if "y" in doubly_batched_inputs:
    inputs["y"] = doubly_batched_inputs["y"][0]

  flow = nux.Flow(create_model, init_key, inputs, batch_axes=(0,))

  outputs = flow.apply(init_key, inputs)

  print("n_params", flow.n_params)

  trainer = initialize_trainer(flow,
                               clip=args.clip,
                               lr=args.lr,
                               warmup=args.warmup,
                               cosine_decay_steps=args.cosine_decay_steps,
                               save_path=args.save_path,
                               retrain=args.retrain,
                               train_args=args.train_args,
                               classification=classification)

  test_losses = sorted(trainer.test_losses.items(), key=lambda x:x[0])
  test_losses = jnp.array(test_losses)

  test_ds = get_test_ds()
  res = trainer.evaluate_test(eval_key, test_ds)
  print("test", trainer.summarize_losses_and_aux(res))

  # Plot samples
  samples = flow.sample(eval_key, n_samples=5000, manifold_sample=True)

  # Find the spread of the data
  data = doubly_batched_inputs["x"].reshape((-1, 2))
  (xmin, ymin), (xmax, ymax) = data.min(axis=0), data.max(axis=0)
  xspread, yspread = xmax - xmin, ymax - ymin
  xmin -= 0.25*xspread
  xmax += 0.25*xspread
  ymin -= 0.25*yspread
  ymax += 0.25*yspread

  # Plot the samples against the true samples and also a dentisy plot
  if "prediction" in samples:
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(28, 7))
    ax1.scatter(*data.T); ax1.set_title("True Samples")
    ax2.scatter(*samples["x"].T, alpha=0.2, s=3, c=samples["prediction"]); ax2.set_title("Learned Samples")
    ax1.set_xlim(xmin, xmax); ax1.set_ylim(ymin, ymax)
    ax2.set_xlim(xmin, xmax); ax2.set_ylim(ymin, ymax)

    n_importance_samples = 100
    x_range, y_range = jnp.linspace(xmin, xmax, 100), jnp.linspace(ymin, ymax, 100)
    X, Y = jnp.meshgrid(x_range, y_range); XY = jnp.dstack([X, Y]).reshape((-1, 2))
    XY = jnp.broadcast_to(XY[None,...], (n_importance_samples,) + XY.shape)
    outputs = flow.scan_apply(eval_key, {"x": XY})
    outputs["log_px"] = jax.scipy.special.logsumexp(outputs["log_px"], axis=0) - jnp.log(n_importance_samples)
    outputs["prediction"] = jnp.mean(outputs["prediction"], axis=0)

    Z = jnp.exp(outputs["log_px"])
    ax3.contourf(X, Y, Z.reshape(X.shape))
    ax4.contourf(X, Y, outputs["prediction"].reshape(X.shape))
    plt.show()
  else:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    ax1.scatter(*data.T); ax1.set_title("True Samples")
    ax2.scatter(*samples["x"].T, alpha=0.2, s=3); ax2.set_title("Learned Samples")
    ax1.set_xlim(xmin, xmax); ax1.set_ylim(ymin, ymax)
    ax2.set_xlim(xmin, xmax); ax2.set_ylim(ymin, ymax)

    n_importance_samples = 100
    x_range, y_range = jnp.linspace(xmin, xmax, 100), jnp.linspace(ymin, ymax, 100)
    X, Y = jnp.meshgrid(x_range, y_range); XY = jnp.dstack([X, Y]).reshape((-1, 2))
    XY = jnp.broadcast_to(XY[None,...], (n_importance_samples,) + XY.shape)
    outputs = flow.scan_apply(eval_key, {"x": XY})
    outputs["log_px"] = jax.scipy.special.logsumexp(outputs["log_px"], axis=0) - jnp.log(n_importance_samples)

    Z = jnp.exp(outputs["log_px"])
    ax3.contourf(X, Y, Z.reshape(X.shape))
    plt.show()

  assert 0

################################################################################################################

def evaluate_image_model(create_model,
                         args,
                         classification=False):
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
                                      label_keep_percent=1.0,
                                      random_label_percent=0.0)

  doubly_batched_inputs = next(train_ds)
  inputs = {"x": doubly_batched_inputs["x"][0]}

  if "y" in doubly_batched_inputs:
    inputs["y"] = doubly_batched_inputs["y"][0]

  flow = nux.Flow(create_model, init_key, inputs, batch_axes=(0,))
  print("n_params", flow.n_params)

  # Evaluate the test set
  trainer = initialize_trainer(flow,
                               clip=args.clip,
                               lr=args.lr,
                               warmup=args.warmup,
                               cosine_decay_steps=args.cosine_decay_steps,
                               save_path=args.save_path,
                               retrain=args.retrain,
                               train_args=args.train_args,
                               classification=classification)

  # Generate reconstructions
  outputs = flow.apply(init_key, inputs, is_training=False)
  outputs["x"] += random.normal(init_key, outputs["x"].shape)
  reconstr = flow.reconstruct(init_key, outputs, generate_image=True)

  # Plot the reconstructions
  fig, axes = plt.subplots(4, 12); axes = axes.ravel()
  for i, ax in enumerate(axes[:8]):
    ax.imshow(reconstr["image"][i].squeeze())

  # Generate some interpolations
  interp = jax.vmap(partial(util.spherical_interpolation, N=4))(outputs["x"][:4], outputs["x"][4:8])
  interp = interp.reshape((-1,) + flow.latent_shape)
  interpolations = flow.reconstruct(init_key, {"x": interp}, generate_image=True)
  for i, ax in enumerate(axes[8:16]):
    ax.imshow(interpolations["image"][i].squeeze())

  # Generate samples
  samples = flow.sample(eval_key, n_samples=axes.size - 16, generate_image=True)
  for i, ax in enumerate(axes[16:]):
    ax.imshow(samples["image"][i].squeeze())

  plt.show()

  import pdb; pdb.set_trace()

  test_losses = sorted(trainer.test_losses.items(), key=lambda x:x[0])
  test_losses = jnp.array(test_losses)

  test_ds = get_test_ds()
  res = trainer.evaluate_test(eval_key, test_ds, bits_per_dim=True)
  print("test", trainer.summarize_losses_and_aux(res))

  import pdb; pdb.set_trace()
