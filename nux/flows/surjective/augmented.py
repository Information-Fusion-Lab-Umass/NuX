import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from functools import partial
import nux.util as util
from typing import Optional, Mapping, Callable, Sequence
from nux.internal.layer import InvertibleLayer
import haiku as hk
from haiku._src.typing import PRNGKey
from jax.scipy.special import gammaln, logsumexp
import nux
import nux.networks as net
import nux.util.weight_initializers as init

__all__ = ["Padding",
           "PaddingChannel",
           "PaddingMultiscaleAndChannel"]

class Padding(InvertibleLayer):
  """ Augmented Normalizing Flows https://arxiv.org/pdf/2002.07101.pdf
  """

  def __init__(self,
               output_dim: int,
               generative_only: bool=False,
               flow: Optional[Callable]=None,
               create_network: Optional[Callable]=None,
               create_feature_network: Optional[Callable]=None,
               network_kwargs: Optional=None,
               name: str="zero_padding",
               **kwargs):
    if generative_only == False:
      self._output_dim = output_dim
    else:
      self._input_dim = output_dim

    self.flow           = flow
    self.network_kwargs = network_kwargs
    self.create_network = create_network
    self.create_feature_network = create_feature_network
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

    def block():
      return nux.sequential(nux.RationalQuadraticSpline(K=8,
                                             network_kwargs=self.network_kwargs,
                                             create_network=self.create_network,
                                             use_condition=True,
                                             coupling=True,
                                             condition_method="concat"),
                            nux.AffineLDU(safe_diag=True))

    f = nux.repeat(block, n_repeats=3)

    return nux.sequential(nux.reverse_flow(f),
                          nux.ParametrizedGaussianPrior(network_kwargs=self.network_kwargs,
                                                        create_network=self.create_network))

  def make_features(self, x, rng):
    # return x
    network = self.create_feature_network(self.input_shape)
    return network({"x": x}, rng)["x"]

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           no_noise: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    k1, k2 = random.split(rng, 2)

    assert self.big_dim - self.small_dim > 0

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

      # Extract features to condition on
      f = self.make_features(z, k1) if self.create_feature_network is not None else z

      flow_inputs = {"x": noise, "condition": f}
      outputs = flow(flow_inputs, k2, sample=False)
      log_pepsgs = outputs.get("log_det", 0.0) + outputs["log_pz"]
      log_det += log_pepsgs

    else:
      # Extract features to condition on
      f = self.make_features(x, k1) if self.create_feature_network is not None else x

      flow_inputs = {"x": jnp.zeros(self.batch_shape + (noise_dim,)), "condition": f}
      outputs = flow(flow_inputs, k2, sample=True)

      noise = outputs["x"]

      z = jnp.concatenate([x, noise], axis=-1)
      log_qepsgs = outputs.get("log_det", 0.0) + outputs["log_pz"]
      log_det -= log_qepsgs

    return {"x": z, "log_det": log_det}

################################################################################################################

class PaddingChannel(InvertibleLayer):
  """ Augmented Normalizing Flows https://arxiv.org/pdf/2002.07101.pdf
  """

  def __init__(self,
               output_channel: int,
               generative_only: bool=False,
               flow: Optional[Callable]=None,
               create_network: Optional[Callable]=None,
               create_feature_network: Optional[Callable]=None,
               network_kwargs: Optional=None,
               name: str="zero_padding",
               **kwargs):
    if generative_only == False:
      self._output_channel = output_channel
    else:
      self._input_channel = output_channel

    self.flow           = flow
    self.network_kwargs = network_kwargs
    self.create_network = create_network
    self.create_feature_network = create_feature_network
    super().__init__(name=name, **kwargs)

  @property
  def input_shape(self):
    return self.unbatched_input_shapes["x"]

  @property
  def output_shape(self):
    return self.unbatched_output_shapes["x"]

  @property
  def input_channel(self):
    if hasattr(self, "_input_channel"):
      return self._input_channel
    return self.input_shape[-1]

  @property
  def output_channel(self):
    if hasattr(self, "_output_channel"):
      return self._output_channel
    return self.output_shape[-1]

  @property
  def kind(self):
    return "tall" if self.input_channel > self.output_channel else "wide"

  @property
  def small_dim(self):
    return self.output_channel if self.input_channel > self.output_channel else self.input_channel

  @property
  def big_dim(self):
    return self.input_channel if self.input_channel > self.output_channel else self.output_channel

  def default_flow(self):

    def block():
      return nux.sequential(nux.RationalQuadraticSpline(K=8,
                                             network_kwargs=self.network_kwargs,
                                             create_network=self.create_network,
                                             use_condition=True,
                                             coupling=True,
                                             condition_method="nin"),
                            nux.OneByOneConv())

    f = nux.repeat(block, n_repeats=3)
    return nux.sequential(nux.reverse_flow(f),
                          nux.ParametrizedGaussianPrior(network_kwargs=self.network_kwargs,
                                                        create_network=self.create_network))

  def make_features(self, x, rng):
    # return x
    network = self.create_feature_network(self.input_shape)
    return network({"x": x}, rng)["x"]

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    k1, k2 = random.split(rng, 2)

    # Figure out which direction we should go
    if sample == False:
      big_to_small = True if self.kind == "tall" else False
    else:
      big_to_small = False if self.kind == "tall" else True

    x = inputs["x"]
    x_shape = self.get_unbatched_shapes(sample)["x"]
    assert len(x_shape) == 3, "Only supporting 3d inputs"

    log_det = jnp.zeros(self.batch_shape)
    flow = self.flow if self.flow is not None else self.default_flow()
    noise_shape = x_shape[:-1] + (self.big_dim - self.small_dim,)

    if big_to_small:
      z, noise = x[...,:self.small_dim], x[...,self.small_dim:]

      # Extract features to condition on
      f = self.make_features(z, k1) if self.create_feature_network is not None else z

      flow_inputs = {"x": noise, "condition": f}
      outputs = flow(flow_inputs, k2, sample=False)
      log_pepsgs = outputs["log_det"] + outputs["log_pz"]
      log_det += log_pepsgs

    else:
      # Extract features to condition on
      f = self.make_features(x, k1) if self.create_feature_network is not None else x

      flow_inputs = {"x": jnp.zeros(self.batch_shape + noise_shape), "condition": f}
      outputs = flow(flow_inputs, k2, sample=True)

      noise = outputs["x"]

      z = jnp.concatenate([x, noise], axis=-1)
      log_qepsgs = outputs["log_det"] + outputs["log_pz"]
      log_det -= log_qepsgs

    return {"x": z, "log_det": log_det}

################################################################################################################

class PaddingMultiscaleAndChannel(InvertibleLayer):
  """ Augmented Normalizing Flows https://arxiv.org/pdf/2002.07101.pdf
  """

  def __init__(self,
               n_squeeze: int,
               output_channel: int,
               generative_only: bool=False,
               flow: Optional[Callable]=None,
               create_network: Optional[Callable]=None,
               create_feature_network: Optional[Callable]=None,
               network_kwargs: Optional=None,
               name: str="zero_padding",
               **kwargs):
    if generative_only == False:
      self._output_channel = output_channel
    else:
      self._input_channel = output_channel

    self.n_squeeze      = n_squeeze
    self.flow           = flow
    self.network_kwargs = network_kwargs
    self.create_network = create_network
    self.create_feature_network = create_feature_network
    super().__init__(name=name, **kwargs)

  @property
  def input_shape(self):
    return self.unbatched_input_shapes["x"]

  @property
  def output_shape(self):
    return self.unbatched_output_shapes["x"]

  @property
  def input_channel(self):
    if hasattr(self, "_input_channel"):
      return self._input_channel
    return self.input_shape[-1]

  @property
  def output_channel(self):
    if hasattr(self, "_output_channel"):
      return self._output_channel
    return self.output_shape[-1]

  @property
  def kind(self):
    return "tall"
    # return "tall" if self.input_channel > self.output_channel else "wide"

  @property
  def small_channel(self):
    return self.output_channel if self.input_channel > self.output_channel else self.input_channel

  @property
  def big_channel(self):
    return self.input_channel if self.input_channel > self.output_channel else self.output_channel

  def default_flow(self):

    def block():
      return nux.sequential(nux.RationalQuadraticSpline(K=8,
                                             network_kwargs=self.network_kwargs,
                                             create_network=self.create_network,
                                             use_condition=True,
                                             coupling=True,
                                             condition_method="nin"),
                            nux.OneByOneConv())

    f = nux.repeat(block, n_repeats=3)
    return nux.sequential(f,
                          nux.ParametrizedGaussianPrior(network_kwargs=self.network_kwargs,
                                                        create_network=self.create_network))

  def make_features(self, x, rng):
    # return x
    network = self.create_feature_network(self.input_shape)
    return network({"x": x}, rng)["x"]

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: PRNGKey,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:

    k1, k2 = random.split(rng, 2)

    # Figure out which direction we should go
    if sample == False:
      big_to_small = True if self.kind == "tall" else False
    else:
      big_to_small = False if self.kind == "tall" else True

    x = inputs["x"]
    x_shape = self.get_unbatched_shapes(sample)["x"]
    assert len(x_shape) == 3, "Only supporting 3d inputs"

    log_det = jnp.zeros(self.batch_shape)
    flow = self.flow if self.flow is not None else self.default_flow()

    if big_to_small:

      for _ in range(self.n_squeeze):
        x = self.auto_batch(util.dilated_squeeze)(x)

      z, noise = x[...,:self.small_channel], x[...,self.small_channel:]

      # Extract features to condition on
      f = self.make_features(z, k1) if self.create_feature_network is not None else z

      flow_inputs = {"x": noise, "condition": f}
      outputs = flow(flow_inputs, k2, sample=False)
      log_pepsgs = outputs["log_det"] + outputs["log_pz"]
      log_det += log_pepsgs

    else:
      # Extract features to condition on
      f = self.make_features(x, k1) if self.create_feature_network is not None else x
      noise_shape = x_shape[:-1] + (self.big_channel - self.small_channel,)

      flow_inputs = {"x": jnp.zeros(self.batch_shape + noise_shape), "condition": f}
      outputs = flow(flow_inputs, k2, sample=True)

      noise = outputs["x"]

      z = jnp.concatenate([x, noise], axis=-1)

      for _ in range(self.n_squeeze):
        z = self.auto_batch(util.dilated_unsqueeze)(z)

      log_qepsgs = outputs["log_det"] + outputs["log_pz"]
      log_det -= log_qepsgs

    return {"x": z, "log_det": log_det}

################################################################################################################

if __name__ == "__main__":
  from debug import *

  def create_fun():

    # def create_network(out_shape):
    #   return net.MLP(out_dim=out_shape[-1],
    #                  layer_sizes=[16, 16],
    #                  nonlinearity="relu",
    #                  parameter_norm="weight_norm",
    #                  # parameter_norm="spectral_norm",
    #                  dropout_rate=None)

    def create_network(out_shape):
      return net.ResNet(out_channel=out_shape[-1],
                        n_blocks=3,
                        hidden_channel=3,
                        nonlinearity="relu",
                        normalization="batch_norm",
                        parameter_norm="weight_norm",
                        block_type="reverse_bottleneck",
                        squeeze_excite=False)

    # def create_network(out_shape):
    #   return net.CNN(out_channel=out_shape[-1],
    #                  n_blocks=1,
    #                  hidden_channel=3,
    #                  nonlinearity="relu",
    #                  normalization=None,
    #                  parameter_norm=None,
    #                  block_type="reverse_bottleneck",
    #                  squeeze_excite=False,
    #                  zero_init=False)

    flat_flow = nux.sequential(PaddingMultiscaleAndChannel(n_squeeze=2,
                                                           output_channel=1,
                                                           create_network=create_network),
                               nux.UnitGaussianPrior())
    return flat_flow

  rng = random.PRNGKey(1)
  # x = random.normal(rng, (10, 8))
  x = random.normal(rng, (10, 8, 8, 3))

  inputs = {"x": x}
  flow = nux.Flow(create_fun, rng, inputs, batch_axes=(0,))
  print(f"flow.n_params: {flow.n_params}")

  def loss(params, state, key, inputs):
    outputs, _ = flow._apply_fun(params, state, key, inputs)
    log_px = outputs.get("log_pz", 0.0) + outputs.get("log_det", 0.0)
    return -log_px.mean()

  outputs = flow.scan_apply(rng, inputs)
  samples = flow.sample(rng, n_samples=4)
  trainer = nux.MaximumLikelihoodTrainer(flow)

  trainer.grad_step(rng, inputs)
  trainer.grad_step_for_loop(rng, inputs)
  trainer.grad_step_scan_loop(rng, inputs)

  gradfun = jax.grad(loss)
  gradfun = jax.jit(gradfun)
  gradfun(flow.params, flow.state, rng, inputs)

  assert 0