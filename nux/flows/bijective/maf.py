import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap, jit
from functools import partial
import haiku as hk
from typing import Optional, Mapping, Sequence
from nux.internal.layer import InvertibleLayer
import nux.util as util
from jax.scipy.special import logsumexp
import nux.util.weight_initializers as init
import nux.networks as net

__all__ = ["MAF"]

################################################################################################################

class MAF(InvertibleLayer):

  def __init__(self,
               hidden_layer_sizes: Sequence[int],
               method: str="shuffled_sequential",
               nonlinearity="relu",
               name: str="maf"
  ):
    """ Masked autoregressive flow https://arxiv.org/pdf/1705.07057.pdf
        Only works for 1d inputs.
    Args:
      hidden_layer_sizes: How many hidden layer units to use in MADE
      method            : How to generate the masks.  Must be ["random", "sequential", "shuffled_sequential"].
                          "shuffled_sequential" will choose random sequential masks.
      nonlinearity      : Nonlinearity to use in the network.
      name              : Optional name for this module.
    """
    super().__init__(name=name)
    self.hidden_layer_sizes = hidden_layer_sizes
    self.method = method
    self.nonlinearity = nonlinearity

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    def initialize_input_sel(shape, dtype):
      dim = shape[-1]
      if self.method == "random":
        rng = hk.next_rng_key()
        input_sel = random.randint(rng, shape=(dim,), minval=1, maxval=dim+1)
      else:
        input_sel = jnp.arange(1, dim + 1)
      return input_sel

    # Initialize the input selection
    dim = inputs["x"].shape[-1]
    input_sel = hk.get_state("input_sel", (dim,), jnp.int32, init=initialize_input_sel)

    # Create the MADE network that will generate the parameters for the MAF.
    made = net.MADE(input_sel,
                    dim,
                    self.hidden_layer_sizes,
                    self.method,
                    nonlinearity=self.nonlinearity,
                    triangular_jacobian=False)

    if sample == False:
      x = inputs["x"]

      made_outs = made(inputs, rng)
      mu, alpha = made_outs["mu"], made_outs["alpha"]
      z = (x - mu)*jnp.exp(-alpha)
      log_det = -alpha.sum(axis=-1)*jnp.ones(self.batch_shape)
      outputs = {"x": z, "log_det": log_det}
    else:
      z = inputs["x"]

      def inverse(z):
        x = jnp.zeros_like(z)

        # We need to build output a dimension at a time
        def carry_body(carry, inputs):
          x, idx = carry, inputs
          made_outs = made(inputs, rng)
          mu, alpha = made_outs["mu"], made_outs["alpha"]
          w = mu + z*jnp.exp(alpha)
          x = jax.ops.index_update(x, idx, w[idx])
          return x, alpha[idx]

        indices = jnp.nonzero(input_sel == (1 + jnp.arange(x.shape[0])[:,None]))[1]
        x, alpha_diag = jax.lax.scan(carry_body, x, indices)
        log_det = -alpha_diag.sum(axis=-1)
        return x, log_det

      x, log_det = self.auto_batch(inverse)(z)
      outputs = {"x": x, "log_det": log_det}

    return outputs
