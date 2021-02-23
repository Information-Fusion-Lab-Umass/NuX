import jax
from jax import random
from nux.internal.base import get_tree_shapes

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

def get_state_tree(name, dtype, init_fun):

  # Check to see if the tree exists.  We will know if we've stored
  # the shapes of the leaves
  if constant_exists(name):
    # Retrieve the saved value
    leaf_shapes = get_tree_shapes(name)

  else:
    pytree = init_fun()
    tree_leaves, treedef = jax.tree_flatten(template)


def get_state_tree(template, name_prefix=""):
  tree_leaves, treedef = jax.tree_flatten(template)

  leaves = []
  for i, val in enumerate(tree_leaves):
    state_leaf = hk.get_state(self.make_name(i, name_prefix), shape=val.shape, dtype=val.dtype, init=lambda s, d: val)
    leaves.append(state_leaf)

  return jax.tree_unflatten(treedef, leaves)

def set_state_tree(tree, name_prefix=""):
  leaves, _ = jax.tree_flatten(tree)

  for i, val in enumerate(leaves):
    hk.get_state(self.make_name(i, name_prefix), val)
