import jax
from jax import random

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
