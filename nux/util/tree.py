import jax
import jax.numpy as jnp
from jax import random
from nux.internal.base import get_tree_shapes
import haiku as hk

__all__ = ["key_tree_like",
           "tree_multimap_multiout",
           "tree_shapes",
           "tree_ndims",
           "tree_equal",
           "get_state_tree",
           "set_state_tree",
           "tree_concat",
           "tree_hstack"]

def key_tree_like(key, pytree):
  # Figure out what the tree structure is
  flat_tree, treedef = jax.tree_util.tree_flatten(pytree)

  # Generate a tree of keys with the same structure as pytree
  n_keys = len(flat_tree)
  keys = random.split(key, n_keys)
  key_tree = jax.tree_util.tree_unflatten(treedef, keys)
  return key_tree

def tree_multimap_multiout(f, tree, *rest):
  # Like tree_multimap but expects f(leaves) to return a tuple.
  # This function will return trees for each tuple element.
  leaves, treedef = jax.tree_util.tree_flatten(tree)
  all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in rest]
  new_leaves = [f(*xs) for xs in zip(*all_leaves)]
  return [treedef.unflatten(leaf) for leaf in zip(*new_leaves)]

def tree_shapes(pytree):
  return jax.tree_util.tree_map(lambda x:x.shape, pytree)

def tree_ndims(pytree):
  return jax.tree_util.tree_map(lambda x:x.ndim, pytree)

def tree_equal(x, y):
  return jax.tree_util.tree_all(jax.tree_multimap(lambda x, y: x == y, x, y))

##########################################################################

# https://github.com/deepmind/dm-haiku/issues#issuecomment-611203193
from typing import Any, NamedTuple

class Box(NamedTuple):
  value: Any
  shape = property(fget=lambda _: ())
  dtype = jnp.float32

def get_state_tree(name, init):
  return hk.get_state(name, (), jnp.float32, init=lambda *_: Box(init())).value

def set_state_tree(name, val):
  return hk.set_state(name, Box(val))

##########################################################################

def tree_concat(x, y, axis=0):
  return jax.tree_multimap(lambda a, b: jnp.concatenate([a, b], axis=axis), x, y)

def tree_hstack(x, y):
  return jax.tree_multimap(lambda a, b: jnp.hstack([a, b]), x, y)