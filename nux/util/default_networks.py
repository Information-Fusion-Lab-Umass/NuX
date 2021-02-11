import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence, Any

def get_default_network(out_shape, network_kwargs=None, resnet=True, lipschitz=False):
  import nux.networks as net

  out_dim = out_shape[-1]

  if network_kwargs is not None:
    if lipschitz:
      assert network_kwargs["parameter_norm"] == "spectral_norm"

  # Otherwise, use default networks
  if len(out_shape) == 1:
    if network_kwargs is None:

      network_kwargs = dict(layer_sizes=[64]*4,
                            nonlinearity="relu",
                            parameter_norm="weight_norm",
                            zero_init=True)
      if lipschitz:
        network_kwargs["parameter_norm"] = "spectral_norm"
        network_kwargs["nonlinearity"] = "lipswish"
        network_kwargs["max_singular_value"] = 0.9
        network_kwargs["max_power_iters"] = 5

    network_kwargs["out_dim"] = out_dim

    return net.MLP(**network_kwargs)

  else:
    if network_kwargs is None:

      network_kwargs = dict(n_blocks=2,
                            hidden_channel=32,
                            nonlinearity="relu",
                            normalization="instance_norm",
                            parameter_norm="weight_norm",
                            block_type="reverse_bottleneck",
                            squeeze_excite=False,
                            zero_init=True)
      if lipschitz:
        network_kwargs["parameter_norm"] = "spectral_norm"
        network_kwargs["normalization"] = None
        network_kwargs["max_singular_value"] = 0.9
        network_kwargs["max_power_iters"] = 5

    network_kwargs["out_channel"] = out_dim

    if resnet:
        return net.ResNet(**network_kwargs)
    else:
        return net.CNN(**network_kwargs)
