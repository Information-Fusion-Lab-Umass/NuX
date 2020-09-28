import jax.numpy as jnp
from jax import jit, random
from functools import partial
import jax
import haiku as hk
from typing import Optional, Mapping, Callable, Sequence, Any
import nux.networks as net

__all__ = ["ImageTransformer"]

################################################################################################################

class MultiheadSelfAttention(hk.Module):
  """ Not able to import this from haiku/examples/transformer/model.py
      This class is an adaptation of the Attention class there. """
  def __init__(self,
               num_heads: int,
               init_scale: float,
               name: Optional[str]=None):
    super().__init__(name=name)
    self.num_heads = num_heads
    self.init_scale = init_scale
    self.linear_init = hk.initializers.VarianceScaling(self.init_scale)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    T, D = x.shape
    assert D%self.num_heads == 0
    head_dim = D//self.num_heads

    w_q = hk.get_parameter("w_q", (D, D), x.dtype, init=self.linear_init)
    b_q = hk.get_parameter("b_q", (D,), x.dtype, init=jnp.zeros)
    q = jnp.einsum("ij,tj->ti", w_q, x) + b_q
    q = q.reshape((T, head_dim, self.num_heads))

    w_k = hk.get_parameter("w_k", (D, D), x.dtype, init=self.linear_init)
    b_k = hk.get_parameter("b_k", (D,), x.dtype, init=jnp.zeros)
    k = jnp.einsum("ij,tj->ti", w_k, x) + b_k
    k = k.reshape((T, head_dim, self.num_heads))

    w_v = hk.get_parameter("w_v", (D, D), x.dtype, init=self.linear_init)
    b_v = hk.get_parameter("b_v", (D,), x.dtype, init=jnp.zeros)
    v = jnp.einsum("ij,tj->ti", w_v, x) + b_v
    v = v.reshape((T, head_dim, self.num_heads))

    # Compute attention matrix.  This takes a ton of memory!!
    attention = jnp.einsum('thd,Thd->htT', q, k)/jnp.sqrt(head_dim)
    attention = jax.nn.softmax(attention)

    # Attend over values, flatten, and return linear result.
    attended_v = jnp.einsum('htT,Thd->thd', attention, v).reshape((T, D))
    return hk.Linear(D, w_init=self.linear_init)(attended_v)

################################################################################################################

class ImageTransformer(hk.Module):
  """ TODO: Create this architecture with purpose. """
  def __init__(self,
               out_channel: int,
               working_channel: int=32,
               conv_block_channel: int=64,
               num_heads: int=4,
               num_layers: int=4,
               name: str="image_transformer"):
    super().__init__(name=name)
    self.working_channel = working_channel
    self.conv_block_channel = conv_block_channel
    self.out_channel = out_channel
    self.num_heads  = num_heads
    self.num_layers = num_layers
    self.init_scale = 2/jnp.sqrt(self.num_layers)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

    # The transformer will run in C order over the height/width of x.
    # Also assume that we're handling batching with vmap when calling this.
    H, W, C = x.shape

    # First, project up the channels
    conv = net.Conv(self.working_channel,
                    kernel_shape=(1, 1),
                    parameter_norm=None,
                    stride=(1, 1),
                    padding="SAME",
                    use_bias=False)
    x = conv(x)

    # Generate the position embeddings for the attention
    pos_emb = hk.get_parameter("pos_emb", x.shape, x.dtype, init=hk.initializers.RandomNormal(stddev=0.01))

    for i in range(self.num_layers):

      # Conv Block
      conv = net.ConvBlock(out_channel=self.working_channel,
                           hidden_channel=self.conv_block_channel,
                           parameter_norm=None,
                           norm="instance_norm",
                           nonlinearity="swish",
                           use_bias=True,
                           zero_init=False)
      x_conv = conv(x)

      # Instead of gating, use SE
      x += net.SqueezeExcitation(reduce_ratio=8)(x_conv)

      # Instance norm
      norm = hk.InstanceNorm(create_scale=True, create_offset=True)
      x = norm(x)

      # Attention with no mask (we don't care about the autoregressive property here!)
      attn = MultiheadSelfAttention(self.num_heads, self.init_scale)
      x_attn = (x + pos_emb).reshape((H*W, -1))
      x_attn = attn(x_attn).reshape((H, W, -1))

      # SE
      x += net.SqueezeExcitation(reduce_ratio=8)(x_attn)

      # Instance norm
      norm = hk.InstanceNorm(create_scale=True, create_offset=True)
      x = norm(x)

    # Project to the final number of channels
    conv_out = net.Conv(self.out_channel,
                        kernel_shape=(1, 1),
                        parameter_norm=None,
                        stride=(1, 1),
                        padding="SAME",
                        use_bias=False,
                        w_init=hk.initializers.RandomNormal(stddev=0.1))
    x = conv_out(x)
    return x