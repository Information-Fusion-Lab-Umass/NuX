import numpy as np
import jax
from jax import random, jit, vmap, jacobian, grad, value_and_grad, pmap
import jax.nn
import jax.numpy as jnp
from functools import partial, reduce
from jax.experimental import stax
from jax.nn.initializers import glorot_normal, normal, ones, zeros
from jax.ops import index, index_add, index_update
import staxplusplus as spp
from jax.scipy.special import logsumexp
from util import is_testing, TRAIN, TEST, householder_prod, householder_prod_transpose
import util
from non_dim_preserving import *
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_multimap, tree_flatten
ravel_pytree = jit(ravel_pytree)
from tqdm.notebook import tnrange
from tqdm import tqdm

################################################################################################################

def UnitGaussianPrior(axis=(-1,), name='unnamed'):
    # language=rst
    """
    Prior for the normalizing flow.

    :param axis - Axes to reduce over
    """
    def init_fun(key, input_shape):
        params, state = (), ()
        return name, input_shape, params, state

    def forward(params, state, x, **kwargs):
        dim = jnp.prod([x.shape[ax] for ax in axis])
        log_det = -0.5*jnp.sum(x**2, axis=axis) + -0.5*dim*jnp.log(2*jnp.pi)
        return log_det, x, state

    def inverse(params, state, z, **kwargs):
        # Usually we're sampling z from a Gaussian, so if we want to do Monte Carlo
        # estimation, ignore the value of N(z|0,I).
        return 0.0, z, state

    return init_fun, forward, inverse

################################################################################################################

def flow_data_dependent_init(x, target_param_names, name_tree, params, state, forward, flag_names, **kwargs):
    # language=rst
    """
    Data dependent initialization for a normalizing flow.

    :param x: The data seed
    :param target_param_names: A list of the names of parameters to seed
    :param name_tree: A pytree (nested structure) of names.  This is the first output of an init_fun call
    :param params: The parameter pytree
    :param state: The state pytree
    :param forward: Forward function
    :param flag_names: The names of the flag that will turn on seeding.

    **Example**

    .. code-block:: python
        from jax import random
        import jax.numpy as jnp
        from normalizing_flows import ActNorm, flow_data_dependent_init
        from util import TRAIN, TEST

        # Create the model
        flow = ActNorm(name='an')

        # Initialize it
        init_fun, forward, inverse = flow
        key = random.PRNGKey(0)
        names, output_shape, params, state = init_fun(key, input_shape=(5, 5, 3), condition_shape=())

        # Seed weight norm and retrieve the new parameters
        data_seed = jnp.ones((10, 5, 5, 3))
        actnorm_names = ['an']
        params = flow_data_dependent_init(data_seed, actnorm_names, names, params, state, forward, (), 'actnorm_seed')
    """
    def filled_forward_function(params, state, x, **kwargs):
        _, ans, updated_states = forward(params, state, jnp.zeros(x.shape[0]), x, **kwargs)
        return ans, updated_states

    return spp.data_dependent_init(x, target_param_names, name_tree, params, state, filled_forward_function, flag_names, **kwargs)

def multistep_flow_data_dependent_init(x,
                                       target_param_names,
                                       flow_model,
                                       condition,
                                       flag_names,
                                       key,
                                       data_loader=None,
                                       n_seed_examples=1000,
                                       batch_size=4,
                                       notebook=True,
                                       **kwargs):
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
    (names, output_shape, params, state), forward, inverse = flow_model

    seed_steps = int(jnp.ceil(n_seed_examples/batch_size))

    # Get the inital parameters
    flat_params, unflatten = ravel_pytree(params)
    unflatten = jit(unflatten)

    # JIT the forward function.  Need to fill the kwargs before jitting otherwise this will fail.
    if(isinstance(flag_names, list) == False and isinstance(flag_names, tuple) == False):
        flag_names = (flag_names,)
    flag_names = dict([(name, True) for name in flag_names])
    jitted_forward = jit(partial(forward, **flag_names))

    # Define a single gpu slice of the dependent init
    @jit
    def single_gpu_init(params, key, x_batch):
        new_params = flow_data_dependent_init(x_batch, target_param_names, names, params, state, jitted_forward, (), None, key=key)
        new_flat_params, _ = ravel_pytree(new_params)
        return new_flat_params

    # Run the data dependent initialization
    pbar = tnrange(seed_steps) if notebook else tqdm(range(seed_steps))
    for i in pbar:
        key, *keys = random.split(key, 3)

        # Get the next batch of data for each gpu
        if(data_loader is not None):
            x_batch = data_loader((batch_size,), key=key)
        else:
            batch_idx = random.randint(keys[0], (batch_size,), minval=0, maxval=x.shape[0])
            x_batch = x[batch_idx,:]

        # Compute the seeded parameters
        new_params = flow_data_dependent_init(x_batch, target_param_names, names, params, state, jitted_forward, (), None, key=key)
        new_flat_params, _ = ravel_pytree(new_params)

        # Compute a running mean of the parameters
        flat_params = i/(i + 1)*flat_params + new_flat_params/(i + 1)
        params = unflatten(flat_params)

    return params

################################################################################################################

def GLOWBlock(transform_fun,
              conditioned_actnorm=False,
              masked=True,
              mask_type='checkerboard',
              additive_coupling=True,
              top_left_zero=False,
              use_ldu=False,
              name='unnamed'):
    # language=rst
    """
    One step of GLOW https://arxiv.org/pdf/1807.03039.pdf

    :param transform: A transformation that will act on half of the input vector. Must return 2 vectors!!!
    :param mask_type: What kind of masking to use.  For images, can use checkerboard
    """
    if(name == 'unnamed'):
        an_name = 'unnamed'
        conv_name = 'unnamed'
        coupling_name = 'unnamed'
    else:
        an_name = '%s_act_norm'%name
        conv_name = '%s_one_by_one_conv'%name
        coupling_name = '%s_affine_coupling'%name

    if(conditioned_actnorm):
        actnorm = ConditionedActNorm(name=an_name)
    else:
        actnorm = ActNorm(name=an_name)

    if(use_ldu):
        conv = OnebyOneConvLDU(name=conv_name)
    else:
        conv = OnebyOneConvLAX(name=conv_name)

    if(masked):
        coupling = MaskedAffineCoupling(transform_fun, mask_type=mask_type, top_left_zero=top_left_zero, name=coupling_name)
    else:
        if(additive_coupling):
            coupling = AdditiveCoupling(transform_fun, name=coupling_name)
        else:
            coupling = AffineCoupling(transform_fun, name=coupling_name)

    return sequential_flow(actnorm, conv, coupling)

def ResidualGLOW(filter_shape=(1, 1), dilation=(1, 1), n_channels=512, ratio=2, name='unnamed'):
    def Coupling2D(out_shape, n_channels=n_channels):
        return spp.DoubledLowDimInputConvBlock(n_channels=n_channels)
        # return spp.sequential(spp.LowDimInputConvBlock(n_channels=n_channels), spp.SqueezeExcitation(ratio=ratio))

    if(name == 'unnamed'):
        an_name = 'unnamed'
        conv_name = 'unnamed'
        coupling_name = 'unnamed'
    else:
        an_name = '%s_act_norm'%name
        conv_name = '%s_one_by_one_conv'%name
        coupling_name = '%s_affine_coupling'%name

    return sequential_flow(LocalDense(filter_shape=filter_shape, dilation=dilation),
                           ActNorm(name=an_name),
                           Squeeze(),
                           AffineCoupling(Coupling2D, name=coupling_name),
                           UnSqueeze())

################################################################################################################

def ConditionedResidualGLOW(filter_shape=(1, 1), dilation=(1, 1), n_channels=512, ratio=2, name='unnamed'):
    def Coupling2D(out_shape, n_channels=n_channels):
        return spp.sequential(spp.LowDimInputConvBlock(n_channels=n_channels), spp.SqueezeExcitation(ratio=ratio))

    if(name == 'unnamed'):
        an_name = 'unnamed'
        conv_name = 'unnamed'
        coupling_name = 'unnamed'
    else:
        an_name = '%s_act_norm'%name
        conv_name = '%s_one_by_one_conv'%name
        coupling_name = '%s_affine_coupling'%name

    return sequential_flow(ConditionedLocalDense(filter_shape=filter_shape, dilation=dilation),
                           ConditionedActNorm(name=an_name),
                           Squeeze(),
                           AdditiveCoupling(Coupling2D, name=coupling_name),
                           UnSqueeze())

################################################################################################################

def ConditionedNIFCoupling(n_channels=512, ratio=4, name='unnamed'):
    conv_init, conv_apply = spp.LowDimInputConvBlock(n_channels=n_channels, init_zeros=False)
    se_init, se_apply = spp.SqueezeExcitation(ratio=ratio)
    # se_init, se_apply = spp.ConditionedSqueezeExcitation(ratio=ratio)

    def init_fun(key, input_shape):
        H, W, C = input_shape
        (K,), = condition_shape
        k1, k2 = random.split(key, 2)

        # Find the shape of the dilated_squeeze output
        H_sq, W_sq, C_sq = (H//2, W//2, C*2*2)

        # Initialize the parameters
        conv_name, conv_output_shape, conv_params, conv_state = conv_init(k1, (H_sq, W_sq, C_sq//2))
        se_name, se_output_shape, se_params, se_state = se_init(k2, conv_output_shape)
        # se_name, se_output_shape, se_params, se_state = se_init(k2, (conv_output_shape, (K,)))

        names = (conv_name, se_name)
        params = (conv_params, se_params)
        state = (conv_state, se_state)
        return names, input_shape, params, state

    def forward(params, state, x, **kwargs):
        cond, = condition
        conv_params, se_params = params
        conv_state, se_state = state

        if(x.ndim == 3):
            dil_sq = dilated_squeeze
            dil_unsq = dilated_unsqueeze
        else:
            dil_sq = vmap(dilated_squeeze, in_axes=(0, None, None))
            dil_unsq = vmap(dilated_unsqueeze, in_axes=(0, None, None))

        x = dil_sq(x, (2, 2), (1, 1))
        x1, x2 = jnp.split(x, 2, axis=-1)

        residual, updated_conv_state = conv_apply(conv_params, conv_state, x2)
        residual, updated_se_state = se_apply(se_params, se_state, residual)
        # residual, updated_se_state = se_apply(se_params, se_state, (residual, cond))

        z1 = x1 + residual
        z = jnp.concatenate([z1, x2], axis=-1)

        z = dil_unsq(z, (2, 2), (1, 1))
        log_det = 0.0
        updated_states = (updated_conv_state, updated_se_state)
        return log_px + log_det, z, updated_states

    def inverse(params, state, z, **kwargs):
        cond, = condition
        conv_params, se_params = params
        conv_state, se_state = state

        if(z.ndim == 3):
            dil_sq = dilated_squeeze
            dil_unsq = dilated_unsqueeze
        else:
            dil_sq = vmap(dilated_squeeze, in_axes=(0, None, None))
            dil_unsq = vmap(dilated_unsqueeze, in_axes=(0, None, None))

        z = dil_sq(z, (2, 2), (1, 1))
        z1, z2 = jnp.split(z, 2, axis=-1)

        residual, updated_conv_state = conv_apply(conv_params, conv_state, z2)
        residual, updated_se_state = se_apply(se_params, se_state, residual)
        # residual, updated_se_state = se_apply(se_params, se_state, (residual, cond))

        x1 = z1 - residual
        x = jnp.concatenate([x1, z2], axis=-1)

        x = dil_unsq(x, (2, 2), (1, 1))
        log_det = 0.0
        updated_states = (updated_conv_state, updated_se_state)
        return log_pz + log_det, x, updated_states

    return init_fun, forward, inverse

################################################################################################################

def GaussianMixtureCDF(n_components=4, weight_logits_init=normal(), mean_init=normal(), variance_init=ones, name='unnamed'):
    # language=rst
    """
    Inverse transform sampling of a Gaussian Mixture Model.  CDF(x|pi,mus,sigmas) = sum[pi_i*erf(x|mu, sigma)]

    :param n_components: The number of components to use in the GMM
    """
    def init_fun(key, input_shape):
        k1, k2, k3 = random.split(key, 3)
        weight_logits = weight_logits_init(k1, (n_components,))
        means = mean_init(k2, (n_components,))
        variances = variance_init(k3, (n_components,))
        params = (weight_logits, means, variances)
        state = ()
        return name, input_shape, params, state

    def forward(params, state, x, **kwargs):
        weights, means, variances = params

        # z is the CDF of x
        dxs = x[...,None] - means[...,:]
        cdfs = 0.5*(1 + jax.scipy.special.erf(dxs/jnp.sqrt(2*variances[...,:])))
        z = jnp.sum(jnp.exp(weights)*cdfs, axis=-1)

        # log_det is log_pdf(x)
        log_pdfs = -0.5*(dxs**2)/variances[...,:] - 0.5*jnp.log(variances[...,:]) - 0.5*jnp.log(2*jnp.pi)
        log_det = logsumexp(weight_logits + log_pdfs, axis=-1)

        # We computed the element-wise log_dets, so sum over the dimension axis
        log_det = log_det.sum(axis=-1)

        return log_px + log_det, z, state

    def inverse(params, state, x, **kwargs):
        # TODO: Implement iterative method to do this
        assert 0, 'Not implemented'

    return init_fun, forward, inverse

################################################################################################################
