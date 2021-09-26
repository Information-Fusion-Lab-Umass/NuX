""" Modified from https://github.com/deepmind/dm-haiku/blob/main/examples/vae.py
    Most of the code is the same, but the encoder model is changed to use a
    normalizing flow with NuX """

from typing import Generator, Mapping, Tuple, NamedTuple, Sequence, Any

from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
import nux
import nux.util as util


flags.DEFINE_integer("batch_size", 128, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 5000, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 100, "How often to evaluate the model.")
flags.DEFINE_integer("random_seed", 42, "Random seed.")
FLAGS = flags.FLAGS


PRNGKey = jnp.ndarray
Batch = Mapping[str, np.ndarray]

MNIST_IMAGE_SHAPE: Sequence[int] = (28, 28, 1)


def load_dataset(split: str, batch_size: int) -> Generator[Batch, None, None]:
  ds = tfds.load("binarized_mnist", split=split, shuffle_files=True,
                 read_config=tfds.ReadConfig(shuffle_seed=FLAGS.random_seed))
  ds = ds.shuffle(buffer_size=10 * batch_size, seed=FLAGS.random_seed)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=5)
  ds = ds.repeat()
  return iter(tfds.as_numpy(ds))

# Changes to haiku_vae.py original file
# https://github.com/deepmind/dm-haiku/issues/32
# This is how we can use NuX with Haiku
class Box(NamedTuple):
  value: Any
  shape = property(fget=lambda _: ())
  dtype = jnp.float32

def get_parameter_tree(name, init):
  return hk.get_parameter(name, (), jnp.float32, init=lambda *_: Box(init())).value

class Encoder(hk.Module):
  """Encoder model."""

  def __init__(self, hidden_size: int = 512, latent_size: int = 10):
    super().__init__()
    self._hidden_size = hidden_size
    self._latent_size = latent_size

  def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = hk.Flatten()(x)

    # Changes to haiku_vae.py original file
    # Create the normalizing flow
    flow = nux.NonlinearCoupling(n_layers=4,
                                 working_dim=16,
                                 hidden_dim=16,
                                 nonlinearity=util.square_swish,
                                 dropout_prob=0.2,
                                 n_resnet_layers=4,
                                 K=8,
                                 kind="logistic_mixture_cdf_logit",
                                 with_affine_coupling=False)
    flow = nux.Sequential([flow, nux.UnitGaussianPrior()])

    # Retrieve the parameters
    def init_fun():
      rng_key = hk.next_rng_key()
      flow(jnp.zeros_like(x), aux=x, rng_key=rng_key, inverse=True)
      return flow.get_params()
    params = get_parameter_tree("flow_params", init=init_fun)

    z, log_qzgx = flow(jnp.zeros_like(x), aux=x, params=params, rng_key=hk.next_rng_key(), inverse=True)
    return z, log_qzgx
    return mean, stddev


class Decoder(hk.Module):
  """Decoder model."""

  def __init__(
      self,
      hidden_size: int = 512,
      output_shape: Sequence[int] = MNIST_IMAGE_SHAPE,
  ):
    super().__init__()
    self._hidden_size = hidden_size
    self._output_shape = output_shape

  def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
    z = hk.Linear(self._hidden_size)(z)
    z = jax.nn.relu(z)

    logits = hk.Linear(np.prod(self._output_shape))(z)
    logits = jnp.reshape(logits, (-1, *self._output_shape))

    return logits


class VAEOutput(NamedTuple):
  image: jnp.ndarray
  # mean: jnp.ndarray
  # stddev: jnp.ndarray
  # Changes to haiku_vae.py original file
  log_qzgx: jnp.ndarray
  logits: jnp.ndarray


class VariationalAutoEncoder(hk.Module):
  """Main VAE model class, uses Encoder & Decoder under the hood."""

  def __init__(
      self,
      hidden_size: int = 512,
      latent_size: int = 10,
      output_shape: Sequence[int] = MNIST_IMAGE_SHAPE,
  ):
    super().__init__()
    self._hidden_size = hidden_size
    self._latent_size = latent_size
    self._output_shape = output_shape

  def __call__(self, x: jnp.ndarray) -> VAEOutput:
    x = x.astype(jnp.float32)
    # mean, stddev = Encoder(self._hidden_size, self._latent_size)(x)
    # z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)

    # Changes to haiku_vae.py original file
    z, log_qzgx = Encoder(self._hidden_size, self._latent_size)(x)

    logits = Decoder(self._hidden_size, self._output_shape)(z)

    p = jax.nn.sigmoid(logits)
    image = jax.random.bernoulli(hk.next_rng_key(), p)

    return VAEOutput(image, log_qzgx, logits)


def binary_cross_entropy(x: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
  """Calculate binary (logistic) cross-entropy from distribution logits.
  Args:
    x: input variable tensor, must be of same shape as logits
    logits: log odds of a Bernoulli distribution, i.e. log(p/(1-p))
  Returns:
    A scalar representing binary CE for the given Bernoulli distribution.
  """
  if x.shape != logits.shape:
    raise ValueError("inputs x and logits must be of the same shape")

  x = jnp.reshape(x, (x.shape[0], -1))
  logits = jnp.reshape(logits, (logits.shape[0], -1))

  return -jnp.sum(x * logits - jnp.logaddexp(0.0, logits), axis=-1)


def kl_gaussian(mean: jnp.ndarray, var: jnp.ndarray) -> jnp.ndarray:
  r"""Calculate KL divergence between given and standard gaussian distributions.
  KL(p, q) = H(p, q) - H(p) = -\int p(x)log(q(x))dx - -\int p(x)log(p(x))dx
           = 0.5 * [log(|s2|/|s1|) - 1 + tr(s1/s2) + (m1-m2)^2/s2]
           = 0.5 * [-log(|s1|) - 1 + tr(s1) + m1^2] (if m2 = 0, s2 = 1)
  Args:
    mean: mean vector of the first distribution
    var: diagonal vector of covariance matrix of the first distribution
  Returns:
    A scalar representing KL divergence of the two Gaussian distributions.
  """
  return 0.5 * jnp.sum(-jnp.log(var) - 1.0 + var + jnp.square(mean), axis=-1)


def main(_):
  FLAGS.alsologtostderr = True

  model = hk.transform(lambda x: VariationalAutoEncoder()(x))  # pylint: disable=unnecessary-lambda
  optimizer = optax.adam(FLAGS.learning_rate)

  @jax.jit
  def loss_fn(params: hk.Params, rng_key: PRNGKey, batch: Batch) -> jnp.ndarray:
    """ELBO loss: E_p[log(x)] - KL(d||q), where p ~ Be(0.5) and q ~ N(0,1)."""
    outputs: VAEOutput = model.apply(params, rng_key, batch["image"])

    log_likelihood = -binary_cross_entropy(batch["image"], outputs.logits)
    elbo = log_likelihood - outputs.log_qzgx
    # kl = kl_gaussian(outputs.mean, jnp.square(outputs.stddev))
    # elbo = log_likelihood - kl

    return -jnp.mean(elbo)

  @jax.jit
  def update(
      params: hk.Params,
      rng_key: PRNGKey,
      opt_state: optax.OptState,
      batch: Batch,
  ) -> Tuple[hk.Params, optax.OptState]:
    """Single SGD update step."""
    grads = jax.grad(loss_fn)(params, rng_key, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

  rng_seq = hk.PRNGSequence(FLAGS.random_seed)
  params = model.init(next(rng_seq), np.zeros((1, *MNIST_IMAGE_SHAPE)))
  opt_state = optimizer.init(params)

  train_ds = load_dataset(tfds.Split.TRAIN, FLAGS.batch_size)
  valid_ds = load_dataset(tfds.Split.TEST, FLAGS.batch_size)

  for step in range(FLAGS.training_steps):
    params, opt_state = update(params, next(rng_seq), opt_state, next(train_ds))

    if step % FLAGS.eval_frequency == 0:
      val_loss = loss_fn(params, next(rng_seq), next(valid_ds))
      logging.info("STEP: %5d; Validation ELBO: %.3f", step, -val_loss)

    import pdb; pdb.set_trace()


if __name__ == "__main__":
  app.run(main)