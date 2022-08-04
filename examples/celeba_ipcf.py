import jax
import jax.numpy as jnp
import nux.util as util
from jax import random
from functools import partial
import nux
import einops
from nux.training.trainer import Trainer
import os

class ImageTrainer(Trainer):

  def __init__(self, args, n_samples=8, quantize_bits=8, **kwargs):
    super().__init__(args, **kwargs)
    self.latent_shape = None
    self.n_samples = n_samples
    self.quantize_bits = quantize_bits
    self.sample_key = random.PRNGKey(0)

  def reconstruct(self, x, params, rng_key):
    z, _ = self.flow(x, params=params, rng_key=rng_key, no_llc=True, is_training=False)
    x_reconstr, _ = self.flow(z, params=params, rng_key=rng_key, inverse=True, no_llc=True, is_training=False, reconstruction=True)
    return x_reconstr

  @property
  def jitted_reconstruct(self):
    if hasattr(self, "_jitted_reconstruct") == False:
      self._jitted_reconstruct = self.reconstruct
      self._jitted_reconstruct = jax.jit(self.reconstruct)
    return self._jitted_reconstruct

  def save_samples(self, train_iter):
    rng_key, self.sample_key = random.split(self.sample_key, 2)

    if self.latent_shape is None:
      data = next(train_iter)["x"][0]
      z, _ = self.apply_fun(data, self.params, self.eval_key)
      self.latent_shape = z.shape[1:]


    k1, k2 = random.split(rng_key, 2)

    # Pull samples
    samples = self.jitted_sample(k2, self.params, self.n_samples)
    samples /= 2**self.args.quantize_bits

    # Pull reconstructions
    x = next(train_iter)["x"][0][:self.n_samples]
    x_reconstr = self.jitted_reconstruct(x, self.params, self.eval_key)
    x_reconstr /= 2**self.args.quantize_bits

    size = 5
    fig, axes = plt.subplots(3, self.n_samples, figsize=(size*self.n_samples, size*3))
    for i in range(self.n_samples):
      axes[0,i].imshow(samples[i])

      axes[1,i].imshow(x[i]/256)
      axes[2,i].imshow(x_reconstr[i])

    for ax in axes.ravel():
      ax.set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0)

    save_path = os.path.join(self.samples_path, f"sample_{self.n_train_steps}.png")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

################################################################################################################

class iPCF():
  def __init__(self, preprocess, flow, prior, args):
    self.preprocess = preprocess
    self.flow = flow
    self.prior = prior
    self.args = args

    self.full_flow = nux.Sequential([self.preprocess, self.flow, self.prior])

  def get_params(self):
    return self.full_flow.get_params()

  def training_objective(self, x, params=None, rng_key=None, debug=False, **kwargs):
    k1, k2, k3, k4 = random.split(rng_key, 4)

    # Preprocess
    u, _ = self.preprocess(x, params=params[0], rng_key=k1, **kwargs)

    # Apply the flow
    z, _ = self.flow(u, params=params[1], rng_key=k2, lo_llc=True, **kwargs)
    _, log_pz = self.prior(z, params=params[2], rng_key=k4, **kwargs)
    z_dim = util.list_prod(z.shape[1:])

    def apply_fun_inv(z):
      u, _ = self.flow(z, params=params[1], rng_key=k2, no_llc=True, inverse=True, force_values=True, **kwargs)
      return u

    # Get an estimate of I
    index = random.randint(k3, minval=0, maxval=z_dim, shape=z.shape[:1])
    mask = jnp.arange(z_dim) == index[:,None]
    mask = mask.reshape(z.shape).astype(z.dtype)
    u_reconstr, Jm = jax.jvp(apply_fun_inv, (z,), (mask,))

    # log_det + I
    sum_axes =  util.last_axes(u.shape[1:])
    JTJmm = (Jm**2).sum(axis=sum_axes)
    Ls = 0.5*z_dim*jnp.log(JTJmm)

    # Reconstruction error
    reconstruction_error = jnp.sum((u - u_reconstr)**2, axis=sum_axes)

    objective = -log_pz + Ls + self.args.gamma*reconstruction_error

    aux = dict(log_pz=log_pz,
               Ls=Ls,
               objective=objective,
               reconstruction_error=reconstruction_error)

    if debug:
      import pdb; pdb.set_trace()

    no_nan = lambda x: jnp.where(jnp.isnan(x), 0.0, x)
    objective = no_nan(objective)
    aux = jax.tree_util.tree_map(no_nan, aux)

    return objective, aux

  def __call__(self, x, params=None, inverse=False, rng_key=None, **kwargs):
    return self.full_flow(x, params=params, inverse=inverse, rng_key=rng_key, **kwargs)

  def log_likelihood(self, x, params=None, rng_key=None, **kwargs):
    k1, k2, k3, k4 = random.split(rng_key, 4)

    # Preprocess
    u, _ = self.preprocess(x, params=params[0], rng_key=k1, **kwargs)

    # Apply the flow
    z, _ = self.flow(u, params=params[1], rng_key=k2, lo_llc=True, **kwargs)
    _, log_pz = self.prior(z, params=params[2], rng_key=k4, **kwargs)
    z_dim = util.list_prod(z.shape[1:])

    def apply_fun_inv(z):
      u, _ = self.flow(z, params=params[1], rng_key=k2, no_llc=True, inverse=True, force_values=True, **kwargs)
      return u

    J = jax.vmap(jax.jacobian(lambda z: apply_fun_inv(z[None])[0]))(z)
    JTJ = jnp.einsum("bji,bjk->bik", J, J)
    llc = -0.5*jnp.linalg.slogdet(JTJ)[1]
    return llc + log_pz

################################################################################################################

def make_preprocess(args):
  return nux.Sequential([nux.UniformDequantization(),
                         nux.StaticScale(2**args.quantize_bits),
                         nux.Logit(scale=0.05, force_values=False)])

def make_flow(args):

  def make_block():
    flow = nux.NonlinearCouplingImage(n_layers=4,
                                      working_channel=3,
                                      hidden_channel=512,
                                      nonlinearity=util.square_swish,
                                      dropout_prob=0.2,
                                      n_resnet_layers=4,
                                      K=8,
                                      kind="spline",
                                      with_affine_coupling=True)
    return flow

  return nux.Sequential([make_block(), # 64x64x3
                         nux.Squeeze(),# 32x32x12
                         nux.Slice(3), # 32x32x3
                         make_block(), # 32x32x3
                         nux.Squeeze(),# 16x16x12
                         nux.Slice(3), # 16x16x3
                         make_block(), # 16x16x3
                         nux.Squeeze(),# 8x8x12
                         nux.Slice(3), # 8x8x3
                         ])

def make_prior(args):
  flow = nux.NonlinearCoupling(n_layers=5,
                               working_dim=32,
                               hidden_dim=64,
                               nonlinearity=util.square_swish,
                               dropout_prob=0.2,
                               n_resnet_layers=5,
                               K=8,
                               kind="spline",
                               with_affine_coupling=True)
  return nux.Sequential([nux.Flatten(), flow], prior=nux.UnitGaussianPrior())

################################################################################################################

def create_flow(args):
  args.save_path = f"celeba_outputs/"

  args.dataset = "celeb_a"

  args.lr                   = 1e-4
  args.warmup               = 1000
  args.cosine_decay_steps   = int(1e6)
  args.cosine_decay_amount  = 5e-2

  args.clip = 5.0

  args.train_batch_size   = 8
  args.train_n_batches    = 1000
  args.test_batch_size    = 64
  args.dd_init_batch_size = 64

  args.max_iters = 2000

  args.alpha = 10.0
  args.gamma = 1000

  flow = iPCF(make_preprocess(args),
              make_flow(args),
              make_prior(args),
              args)

  return flow, args

################################################################################################################

def loss(flow, params, inputs, is_training=True, args=None, mean_aux=True, **kwargs):
  x, rng_key = inputs["x"], inputs["rng_key"]

  if params is None:
    flow(x, params=None, rng_key=rng_key, is_training=is_training, **kwargs)
    params = flow.get_params()

  objective, aux = flow.training_objective(x, params=params, rng_key=rng_key, **kwargs)

  if mean_aux:
    aux = jax.tree_util.tree_map(jnp.mean, aux)

  return objective.mean(), aux

################################################################################################################

if __name__ == "__main__":
  from debug import *
  import matplotlib.pyplot as plt
  import tqdm

  def str2bool(v):
    # https://stackoverflow.com/a/43357954
    if isinstance(v, bool):
     return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
      return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
      return False
    else:
      raise argparse.ArgumentTypeError('Boolean value expected.')

  import argparse
  parser = argparse.ArgumentParser(description="Run an experiment")

  # Control args
  parser.add_argument("--retrain", type=str2bool, default=True, help="Retrain model.")
  parser.add_argument("--best_model", type=str2bool, default=False, help="Load the best model.")
  parser.add_argument("--save_path", type=str, default="research_results/test", help="Base path where everything will be saved to.")
  parser.add_argument("--evaluate", type=str2bool, default=False, help="Evaluate all of the results")
  parser.add_argument("--save_samples", type=str2bool, default=True, help="Save samples during training?")

  # Seeds
  parser.add_argument("--init_key_seed", type=int, default=0, help="Initialization random seed")
  parser.add_argument("--train_key_seed", type=int, default=0, help="Training random seed")
  parser.add_argument("--eval_key_seed", type=int, default=0, help="Evaluation random seed")

  # Optimizer args
  parser.add_argument("--clip", type=float, default=15.0, help="Amount to clip gradients.  Set to negative number to turn off.")
  parser.add_argument("--lr", type=float, default=4e-5, help="Default learning rate.")
  parser.add_argument("--warmup", type=int, default=1000, help="Number of steps to linearly increase the learning rate to args.lr.")
  parser.add_argument("--cosine_decay_steps", type=int, default=int(1e5), help="Decay interval if we want to cosine anneal the learning rate.")
  parser.add_argument("--cosine_decay_amount", type=float, default=1e-1, help="Amount to decay learning rate by.")

  # Dataset args
  parser.add_argument("--dataset", type=str, default="celeb_a", help="Dataset to learn.")
  parser.add_argument("--quantize_bits", type=int, default=8, help="Number of color bits to keep from the input images.")
  parser.add_argument("--label_keep_percent", type=float, default=1.0, help="Percent of dataset that should be labeled.")
  parser.add_argument("--random_label_percent", type=float, default=0.0, help="Percent of labels to randomly change.")
  parser.add_argument("--train_split", type=str, default="train", help="Name of trainin split.  Can use to choose subset of data.")
  parser.add_argument("--data_augmentation", type=str2bool, default=False, help="Use data augmentation consistency reg.")

  # Batch args
  parser.add_argument("--dd_init_batch_size", type=int, default=64, help="Batch size for data dependent init.")
  parser.add_argument("--train_batch_size", type=int, default=64, help="Batch size to use.")
  parser.add_argument("--train_n_batches", type=int, default=1000, help="Number training steps to run inside a lax.scan loop.  This can significantly speed up training.")
  parser.add_argument("--test_batch_size", type=int, default=64, help="Batch size of test set.")

  # Training args
  parser.add_argument("--max_iters", type=int, default=2000, help="Number of training steps")
  parser.add_argument("--eval_interval", type=int, default=1000, help="Number of training steps needed before evaluating test set.  This does not take into account batch size.")

  args = parser.parse_args()

  # Create the model
  flow, args = create_flow(args)

  trainer = ImageTrainer(args, use_pmap=False, use_vmap=False, pmap_batch=True, train_for_loop=True)
  trainer.initialize(flow, loss_fun=partial(loss, flow, args=args))

  # Create the dataset iterators.  Also stateless
  train_ds, get_test_ds = trainer.load_data(classification=args.evaluate)

  if trainer.args.retrain == True:
    # Initialize the model parameters from scratch
    def initialize_flow(rng_key, x):
      inputs = dict(x=x, rng_key=rng_key)
      objective, aux = loss(flow, None, inputs, is_training=True, args=args)
      params = flow.get_params()
      return params

    data = next(train_ds)
    x = data["x"]
    x = einops.rearrange(x, "n b ... -> (n b) ...")
    x = x[:args.dd_init_batch_size]
    params = initialize_flow(trainer.init_key, x)

    # Give the trainer the initialized parameters and initialize opt_state
    trainer.initialize_optimizer(params)

  else:
    # Load the parameters from file
    trainer.load_from_file()

  if args.evaluate == False:
    # Train
    pbar = tqdm.tqdm(jnp.arange(args.max_iters))
    for i in pbar:

      # Train for a bit
      avg_train_loss = trainer.train(train_ds)
      pbar.set_description(f"Train loss: {avg_train_loss:5.3f}")

      # Evaluate the test set
      trainer.test(get_test_ds)

      # Save everything
      trainer.checkpoint()

      # Save some visual stuff
      trainer.save_training_plot()

      if args.save_samples:
        # Save samples
        trainer.save_samples(train_ds)

      if jnp.isnan(avg_train_loss):
        assert 0

  else:
    trainer.test(get_test_ds)
    trainer.save_samples(train_ds)
