import jax
import jax.numpy as jnp
from jax import vmap, random, jit
from functools import partial
import src.util as util
import tqdm

def multistep_data_init(data_iter,
                        target_param_names,
                        names,
                        params,
                        state,
                        forward,
                        inverse,
                        flag_names,
                        n_seed_examples=10000,
                        batch_size=32,
                        notebook=False):
    # language=rst
    """
    Data dependent initialization for a normalizing flow that is split up into multiple steps

    :param x: The data seed
    :param target_param_names: A list of the names of parameters to seed
    :param name_tree: A pytree (nested structure) of names.  This is the first output of an init_fun call
    :param params: The parameter pytree
    :param state: The state pytree
    :param forward: Forward function
    :param flag_names: The names of the flag that will turn on seeding.
    """
    seed_steps = int(jnp.ceil(n_seed_examples/batch_size))

    # JIT the forward function.  Need to fill the kwargs before jitting otherwise this will fail.
    if(isinstance(flag_names, list) == False and isinstance(flag_names, tuple) == False):
        flag_names = (flag_names,)
    flag_names = dict([(name, True) for name in flag_names])
    jitted_forward = jit(partial(forward, **flag_names))

    @partial(jit, static_argnums=(0,))
    def running_average(i, x, y):
        return i/(i + 1)*x + y/(i + 1)

    # Run the data dependent initialization
    pbar = tqdm.notebook.tqdm(jnp.arange(seed_steps)) if notebook else tqdm(jnp.arange(seed_steps))
    for i in pbar:
        # Get the next batch of data
        x_batch = data_iter()

        # Run the network
        _, _, states_with_seed = jitted_forward(params, state, x_batch)

        # Replace the parameters with the seeded parameters
        new_params = params
        for name in target_param_names:
            seeded_param = util.get_param(name, names, states_with_seed)
            new_params = util.modify_param(name, names, new_params, seeded_param)

        assert jax.tree_util.tree_flatten(params)[1] == jax.tree_util.tree_flatten(new_params)[1]

        # Compute a running mean of the parameters
        params = running_average(i, params, new_params)
        # params = jax.tree_multimap(lambda x, y: i/(i + 1)*x + y/(i + 1), params, new_params)

    return params