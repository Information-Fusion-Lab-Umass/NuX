import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from functools import partial
import nux.util as util
from typing import Optional, Mapping, Callable, Sequence
from nux.internal.layer import Layer
import haiku as hk
from haiku._src.typing import PRNGKey
from jax.scipy.special import gammaln, logsumexp
import nux
import nux.networks as net
import nux.util.weight_initializers as init

__all__ = ["ZeroPadding"]

class ZeroPadding(Layer):
  """ Augmented Normalizing Flows https://arxiv.org/pdf/2002.07101.pdf
  """

  def __init__(self,
               output_dim: int,
               generative_only: bool=False,
               flow: Optional[Callable]=None,
               network_kwargs: Optional=None,
               name: str="zero_padding",
               **kwargs):
    if generative_only == False:
      self._output_dim = output_dim
    else:
      self._input_dim = output_dim

    self.flow           = flow
    self.network_kwargs = network_kwargs
    super().__init__(name=name, **kwargs)

  @property
  def input_shape(self):
    return self.unbatched_input_shapes["x"]

  @property
  def output_shape(self):
    return self.unbatched_output_shapes["x"]

  @property
  def input_dim(self):
    if hasattr(self, "_input_dim"):
      return self._input_dim
    return util.list_prod(self.input_shape)

  @property
  def output_dim(self):
    if hasattr(self, "_output_dim"):
      return self._output_dim
    return util.list_prod(self.output_shape)

  @property
  def kind(self):
    return "tall" if self.input_dim > self.output_dim else "wide"

  @property
  def small_dim(self):
    return self.output_dim if self.input_dim > self.output_dim else self.input_dim

  @property
  def big_dim(self):
    return self.input_dim if self.input_dim > self.output_dim else self.output_dim

  def default_flow(self):

    def create_network(out_shape):
      return net.MLP(out_dim=out_shape[-1],
                     layer_sizes=[64]*4,
                     nonlinearity="relu",
                     parameter_norm="weight_norm",
                     zero_init=True,
                     dropout_rate=None)

    kwargs = dict(n_components=8,
                  create_network=create_network,
                  use_condition=True)
    main_flow = nux.sequential(nux.AffineLDU(),
                               nux.CouplingLogitsticMixtureLogit(**kwargs),
                               nux.AffineLDU(),
                               nux.CouplingLogitsticMixtureLogit(**kwargs),
                               nux.AffineLDU(),
                               nux.CouplingLogitsticMixtureLogit(**kwargs),
                               nux.AffineLDU())
    return nux.sequential(nux.reverse_flow(main_flow),
                          nux.UnitGaussianPrior())

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           no_noise: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    # Figure out which direction we should go
    if sample == False:
      big_to_small = True if self.kind == "tall" else False
    else:
      big_to_small = False if self.kind == "tall" else True

    x = inputs["x"]
    x_shape = self.get_unbatched_shapes(sample)["x"]
    assert len(x_shape) == 1, "Only supporting 1d inputs"

    log_det = jnp.zeros(self.batch_shape)
    flow = self.flow if self.flow is not None else self.default_flow()
    noise_dim = self.big_dim - self.small_dim

    if big_to_small:
      z, noise = x[...,:self.small_dim], x[...,self.small_dim:]

      flow_inputs = {"x": noise, "condition": z}
      outputs = flow(flow_inputs, rng, sample=False)
      log_pepsgs = outputs["log_det"] + outputs["log_pz"]
      log_det += log_pepsgs

    else:
      flow_inputs = {"x": jnp.zeros(self.batch_shape + (noise_dim,)), "condition": x}
      outputs = flow(flow_inputs, rng, sample=True)
      noise = outputs["x"]
      z = jnp.concatenate([x, noise], axis=-1)
      log_qepsgs = outputs["log_det"] + outputs["log_pz"]
      log_det -= log_qepsgs

    return {"x": z, "log_det": log_det}
