import jax
import jax.numpy as jnp
import nux.util as util
from jax import random, vmap
from functools import partial
import haiku as hk
from typing import Optional, Mapping
from nux.internal.layer import InvertibleLayer
import nux.util as util

__all__ = ["ActNorm"]

################################################################################################################

class ActNorm(InvertibleLayer):

  def __init__(self,
               axis=-1,
               safe_diag: bool=True,
               name: str="act_norm"
  ):
    """ Act norm.  Used in GLOW https://arxiv.org/pdf/1807.03039.pdf
    Args:
      axis: Axes to normalize over.  -1 will normalize over the channel dim like in GLOW,
            (-3, -2, -1) will normalize over every dimension like in Flow++
      name: Optional name for this module.
    """
    super().__init__(name=name)
    self.axes = (axis,) if isinstance(axis, int) else axis
    for ax in self.axes:
      assert ax < 0, "For convenience, pass in negative indexed axes"

    self.safe_diag = safe_diag

  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
  ) -> Mapping[str, jnp.ndarray]:
    outputs = {}
    x = inputs["x"]
    x_shape = self.get_unbatched_shapes(sample)["x"]

    def b_init(shape, dtype):
      keep_axes = [ax%len(x.shape) for ax in self.axes]
      reduce_axes = tuple([ax for ax in range(len(x.shape)) if ax not in keep_axes])
      if len(reduce_axes) == 1:
        reduce_axes = reduce_axes[0]
      return jnp.mean(x, axis=reduce_axes)

    def log_s_init(shape, dtype):
      keep_axes = [ax%len(x.shape) for ax in self.axes]
      reduce_axes = tuple([ax for ax in range(len(x.shape)) if ax not in keep_axes])
      if len(reduce_axes) == 1:
        reduce_axes = reduce_axes[0]
      return jnp.log(jnp.std(x, axis=reduce_axes) + 1e-5)

    param_shape = tuple([x_shape[ax] for ax in self.axes])
    b     = hk.get_parameter("b", shape=param_shape, dtype=x.dtype, init=b_init)
    log_s = hk.get_parameter("log_s", shape=param_shape, dtype=x.dtype, init=log_s_init)

    if self.safe_diag:
      s = util.proximal_relu(log_s) + 1e-5
      log_s = jnp.log(s)

    if sample == False:
      outputs["x"] = (x - b)*jnp.exp(-log_s)
    else:
      outputs["x"] = jnp.exp(log_s)*x + b

    log_det = jnp.broadcast_to(-log_s, x_shape)
    outputs["log_det"] = log_det.sum()

    return outputs
