from functools import partial, wraps
import jax.numpy as jnp
import jax
from jax import random, jit, vmap
import haiku as hk
from abc import ABC, abstractmethod
from typing import Optional, Mapping, Type, Callable, Iterable, Any, Sequence, Union, Tuple, MutableMapping, NamedTuple, Set, TypeVar
from haiku._src.typing import PRNGKey, Params, State
from nux.internal.base import get_tree_shapes

__all__ = ["Layer"]

################################################################################################################

class Layer(hk.Module, ABC):

  batch_axes = ()
  _is_initializing = False

  def __init__(self,
               name=None,
               monitor_stats=False):
    """ This base class will keep track of the input and output shapes of each function call
        so that we can know the batch size of inputs and automatically use vmap to make unbatched
        code work with batched code.
    """
    super().__init__(name=name)
    self.monitor_stats = monitor_stats

  def monitor_layer_stats(self, inputs, outputs):
    assert 0, "Not implemented"

  def get_unbatched_shapes(self):
    return self.unbatched_input_shapes

  def __call__(self,
               inputs: Mapping[str, jnp.ndarray],
               rng: jnp.ndarray=None,
               **kwargs
    ) -> Mapping[str, jnp.ndarray]:

    batch_axes = Layer.batch_axes

    self.unbatched_input_shapes = get_tree_shapes("unbatched_input_shapes", inputs, batch_axes=batch_axes)
    self.unbatched_output_shapes = get_tree_shapes("unbatched_output_shapes", None, batch_axes=batch_axes, do_not_set=True)

    # For convenience, also get the batch axes
    try:
      self.batch_shape = inputs["x"].shape[:-len(self.unbatched_input_shapes["x"])]
    except:
      self.batch_shape = jax.tree_leaves(inputs["x"])[0].shape[:-len(self.unbatched_input_shapes["x"][0])]

    # Run the actual function
    outputs = self.call(inputs, rng, **kwargs)

    # Keep track of the initial output shapes
    get_tree_shapes("unbatched_output_shapes", outputs, batch_axes=batch_axes)

    if self.monitor_stats:
      self.monitor_layer_stats(inputs, outputs)

    return outputs

  def make_singly_batched(self, x):
    x_shape = self.unbatched_input_shapes["x"]

    if len(self.batch_shape) > 1:
      x = jax.tree_map(lambda x: x.reshape((-1,) + x_shape), x)
      def reshape(x):
        return jax.tree_map(lambda x: x.reshape(self.batch_shape + x_shape), x)

    elif len(self.batch_shape) == 0:
      x = jax.tree_map(lambda x: x[None], x)
      def reshape(x):
        return jax.tree_map(lambda x: x[0], x)
    else:
      reshape = lambda x: x

    return x, reshape

  def auto_batch(self,
                 fun,
                 in_axes=None,
                 out_axes=None,
                 expected_depth=None):

    vmap_kwargs = {}
    if in_axes is not None:
      vmap_kwargs["in_axes"] = in_axes
    # if out_axes is not None:
    #   vmap_kwargs["out_axes"] = out_axes

    # We might have functions that expect batched code
    batch_depth = len(self.batch_shape)
    if expected_depth is None:
      n_newaxis = 0
      n_vmaps = batch_depth
    elif expected_depth > batch_depth:
      n_newaxis = expected_depth - batch_depth
      n_vmaps = 0
    elif expected_depth == batch_depth:
      n_newaxis = 0
      n_vmaps = 0
    elif expected_depth < batch_depth:
      n_newaxis = 0
      n_vmaps = batch_depth - expected_depth

    def vmapped_fun(*args, **kwargs):
      # Add in dummy axes to the args
      args = list(args)
      for _ in range(n_newaxis):
        if in_axes is not None:
          for i, ax in enumerate(in_axes):
            if ax is not None:
              args[i] = jax.tree_map(lambda x: jnp.expand_dims(x, ax), args[i])
        else:
          args = jax.tree_map(lambda x: x[None], args)
      args = tuple(args)

      # Apply vmap
      vf = partial(fun, **kwargs)
      for i in range(n_vmaps):
        vf = vmap(vf, **vmap_kwargs)

      # Run the function
      out = vf(*args)

      # Remove the dummy axes
      for _ in range(n_newaxis):
        if out_axes is None:
          out = jax.tree_map(lambda x: x[0], out)
        else:
          for i, ax in enumerate(out_axes):
            if ax is not None:
              out[i] = jax.tree_map(lambda x: x[0], out[i])

      return out

    return vmapped_fun

  @abstractmethod
  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           **kwargs
    ) -> Mapping[str, jnp.ndarray]:
    """ The expectation is that inputs will be a dicionary with
        "x" holding data and "y" holding possible labels.  Other inputs
        can be passed in too """
    pass

################################################################################################################

class InvertibleLayer(Layer):

  def __init__(self,
               name=None,
               invertible_ad=False,
               monitor_stats=False):
    """ This base class will keep track of the input and output shapes of each function call
        so that we can know the batch size of inputs and automatically use vmap to make unbatched
        code work with batched code.
    """
    super().__init__(monitor_stats=monitor_stats,
                     name=name)
    self.invertible_ad = invertible_ad

  def get_unbatched_shapes(self, sample):
    if sample == False:
      return self.unbatched_input_shapes
    else:
      return self.unbatched_output_shapes

  def __call__(self,
               inputs: Mapping[str, jnp.ndarray],
               rng: jnp.ndarray=None,
               sample: Optional[bool]=False,
               **kwargs
    ) -> Mapping[str, jnp.ndarray]:

    batch_axes = Layer.batch_axes

    if sample == False:
      self.unbatched_input_shapes = get_tree_shapes("unbatched_input_shapes", inputs, batch_axes=batch_axes)
      self.unbatched_output_shapes = get_tree_shapes("unbatched_output_shapes", None, batch_axes=batch_axes, do_not_set=True)
    else:
      self.unbatched_input_shapes = get_tree_shapes("unbatched_input_shapes", None, batch_axes=batch_axes, do_not_set=True)
      self.unbatched_output_shapes = get_tree_shapes("unbatched_output_shapes", inputs, batch_axes=batch_axes)

    # For convenience, also get the batch axes
    try:
      if sample == False:
        self.batch_shape = inputs["x"].shape[:-len(self.unbatched_input_shapes["x"])]
      else:
        self.batch_shape = inputs["x"].shape[:-len(self.unbatched_output_shapes["x"])]
    except:
      if sample == False:
        self.batch_shape = jax.tree_leaves(inputs["x"])[0].shape[:-len(self.unbatched_input_shapes["x"][0])]
      else:
        self.batch_shape = jax.tree_leaves(inputs["x"])[0].shape[:-len(self.unbatched_output_shapes["x"][0])]

    # Run the actual function
    if self.invertible_ad == False:
      outputs = self.call(inputs, rng, sample=sample, **kwargs)
    else:
      assert 0, "Not implemented"

    if sample == False:
      # Keep track of the initial output shapes
      get_tree_shapes("unbatched_output_shapes", outputs, batch_axes=batch_axes)
    else:
      get_tree_shapes("unbatched_input_shapes", outputs, batch_axes=batch_axes)

    if self.monitor_stats:
      self.monitor_layer_stats(inputs, outputs)

    return outputs

  @abstractmethod
  def call(self,
           inputs: Mapping[str, jnp.ndarray],
           rng: jnp.ndarray=None,
           sample: Optional[bool]=False,
           **kwargs
    ) -> Mapping[str, jnp.ndarray]:
    """ The expectation is that inputs will be a dicionary with
        "x" holding data and "y" holding possible labels.  Other inputs
        can be passed in too """
    pass
