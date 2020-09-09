import optax
import jax
from haiku._src.data_structures import frozendict
from collections import OrderedDict
from optax._src import transform
from typing import NamedTuple
OptState = NamedTuple

__all__ = ['apply_freeze']

def get_tree_keys(pytree, root_key=None):
    """ Get the key paths for a dictionary based pytree """
    if((isinstance(pytree, dict) or
       isinstance(pytree, OrderedDict) or
       isinstance(pytree, frozendict)) and
       len(pytree.keys()) > 0):

        return_list = []
        items = pytree.items()
        for key, val in items:
            joined_key = key if root_key is None else root_key+'/'+key
            ret_list = get_tree_keys(val, joined_key)
            return_list.extend(ret_list)

        return return_list
    else:
        return [root_key]

def verify_freeze_path(params, freeze_paths):

    param_list = get_tree_keys(params)

    # Double check that all of the parameters we want to freeze are valid
    invalid = []
    for name in param_list:
        valid = False
        for path in freeze_paths:
            if(path in name):
                valid = True

        if(valid is False):
            invalid_names.append(name)

    if(len(invalid) > 0):
        assert 0, f'Got invalid parameter paths {str(invalid)}'

################################################################################################################

def get_freeze_tree_keys(freeze_paths, pytree, root_key=None, freeze=False):
    """ Create a pytree whose leaves are True or False depending if the branch is frozen """

    if(root_key in freeze_paths):
        freeze = True

    if((isinstance(pytree, dict) or
       isinstance(pytree, OrderedDict) or
       isinstance(pytree, frozendict)) and
       len(pytree.keys()) > 0):

        return_dict = {}
        items = pytree.items()
        for key, val in items:
            joined_key = key if root_key is None else root_key+'/'+key
            ret_dict = get_freeze_tree_keys(freeze_paths, val, joined_key, freeze=freeze)
            return_dict[key] = ret_dict

        return type(pytree)(return_dict)
    else:
        return 0.0 if freeze else 1.0

################################################################################################################

class FreezeState(OptState):
    freeze_tree: Updates

def apply_freeze(freeze_paths):

    def init_fn(params):
        verify_freeze_path(params, freeze_paths)

        freeze_tree = get_freeze_tree_keys(freeze_paths, params)
        return FreezeState(freeze_tree=freeze_tree)

    def update_fn(updates, state, params=None):
        del params
        updates = jax.tree_multimap(lambda g, f: g*f, updates, state.freeze_tree)
        return updates, state

    return transform.GradientTransformation(init_fn, update_fn)
