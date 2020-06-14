
def every_other(x):
    assert x.ndim == 1
    dim_x = x.shape[0]
    y = jnp.pad(x, (0, 1)) if dim_x%2 == 1 else x

    dim_y = y.shape[0]
    y = y.reshape((-1, 2)).T.reshape(dim_y)

    return y[:-1] if dim_x%2 == 1 else y

def CoupledDimChange(transform_fun,
                     prior_flow,
                     out_dim,
                     kind='every_other',
                     A_init=glorot_normal(),
                     name='unnamed'):
    ### p(x1, x2) = \int \int p(z1, z2)N(x1|A1@z1+b(x2),\Sigma(x2))N(x2|A2@z2+b(z1),\Sigma(z1))dz1 dz2
    """ General change of dimension.

        Args:
    """
    apply_fun = None
    prior_init_fun, prior_forward, prior_inverse = prior_flow
    x1_dim, x2_dim, z1_dim, z2_dim = None, None, None, None
    x_every_other_idx, x_regular_idx, z_every_other_idx, z_regular_idx = None, None, None, None

    def init_fun(key, input_shape):
        x_shape = input_shape
        assert len(x_shape) == 1, 'Only working with vectors for the moment!!!'
        assert out_dim > 1, 'Can\'t end up with single dimension!  Need at least 2.'

        output_shape = (out_dim,)
        keys = random.split(key, 5)

        x_dim = x_shape[-1]
        z_dim = out_dim
        assert x_dim >= z_dim

        # Figure out how to split x and how that impacts the shapes of z_1 and z_2
        nonlocal x1_dim, x2_dim, z1_dim, z2_dim
        x1_dim = x_dim//2
        x2_dim = x_dim - x1_dim

        z1_dim = out_dim//2
        z2_dim = out_dim - z1_dim

        # If we're splitting using every other index, generate the indexers needed
        if(kind == 'every_other'):
            nonlocal x_every_other_idx, x_regular_idx, z_every_other_idx, z_regular_idx
            x_every_other_idx = every_other(jnp.arange(x_dim))
            x_regular_idx = jnp.array([list(x_every_other_idx).index(i) for i in range(x_dim)])

            z_every_other_idx = every_other(jnp.arange(z_dim))
            z_regular_idx = jnp.array([list(z_every_other_idx).index(i) for i in range(z_dim)])

        # We're not going to learn A for the moment
        A1 = A_init(keys[0], (x1_dim, z1_dim))
        A2 = A_init(keys[1], (x2_dim, z2_dim))

        # Initialize the flow
        prior_name, prior_output_shape, prior_params, prior_state = prior_init_fun(keys[2], output_shape, condition_shape)

        # Initialize each of the flows.  apply_fun can be shared
        nonlocal apply_fun
        init_fun1, apply_fun = transform_fun(out_shape=(x1_dim,))
        init_fun2, _ = transform_fun(out_shape=(x2_dim,))

        # Initialize the transform function.
        # Should output bias and log diagonal covariance
        t_name1, (log_diag_cov_shape1, b_shape1), t_params1, t_state1 = init_fun1(keys[3], (x2_dim,))
        t_name2, (log_diag_cov_shape2, b_shape2), t_params2, t_state2 = init_fun2(keys[4], (z1_dim,))

        names = (name, prior_name, t_name1, t_name2)
        params = ((A1, A2), prior_params, t_params1, t_params2)
        state = ((), prior_state, t_state1, t_state2)
        return names, prior_output_shape, params, state

    def forward(params, state, x, **kwargs):
        (A1, A2), prior_params, t_params1, t_params2 = params
        _, prior_state, t_state1, t_state2 = state

        # Get multiple keys if we're sampling
        key = kwargs.pop('key', None)
        if(key is not None):
            # Re-fill key for the rest of the flow
            k1, k2, k3, k4, k5 = random.split(key, 5)
        else:
            k1, k2, k3, k4, k5 = (None,)*5

        # Determine if we are batching or not
        is_batched = x.ndim == 2
        sigma = kwargs.get('sigma', 1.0)
        posterior_fun = vmap(tall_affine_posterior_diag_cov, in_axes=(0, 0, None, 0, None)) if is_batched else tall_affine_posterior_diag_cov

        # Split x
        if(kind == 'every_other'):
            x1, x2 = jnp.split(x[...,x_every_other_idx], jnp.array([x1_dim]), axis=-1)
        else:
            x1, x2 = jnp.split(x, jnp.array([x1_dim]), axis=-1)

        # Compute the bias and covariance conditioned on x2
        (log_diag_cov1, b1), updated_t_state1 = apply_fun(t_params1, t_state1, x2, key=k1, **kwargs)

        # Get the terms to compute and sample from the posterior
        z1, log_hx1, sigma_ATA_chol1 = posterior_fun(x1, b1, A1, log_diag_cov1, sigma)

        # Sample z1
        if(key is not None):
            noise = random.normal(k2, z1.shape)
            if(is_batched):
                z1 += jnp.einsum('bij,bj->bi', sigma_ATA_chol1, noise)
            else:
                z1 += jnp.einsum('ij,j->i', sigma_ATA_chol1, noise)

        # Compute the bias and covariance conditioned on z1
        (log_diag_cov2, b2), updated_t_state2 = apply_fun(t_params2, t_state2, z1, key=k3, **kwargs)

        # Get the terms to compute and sample from the posterior
        z2, log_hx2, sigma_ATA_chol2 = posterior_fun(x2, b2, A2, log_diag_cov2, sigma)

        # Sample z2
        if(key is not None):
            noise = random.normal(k4, z2.shape)
            if(is_batched):
                z2 += jnp.einsum('bij,bj->bi', sigma_ATA_chol2, noise)
            else:
                z2 += jnp.einsum('ij,j->i', sigma_ATA_chol2, noise)

        # Combine z
        if(kind == 'every_other'):
            z = jnp.concatenate([z1, z2], axis=-1)[...,z_regular_idx]
        else:
            z = jnp.concatenate([z1, z2], axis=-1)

        # Compute the prior
        log_pz, z, updated_prior_state = prior_forward(prior_params, prior_state, z, key=k5, **kwargs)

        # Return the full estimate of the integral and the updated
        updated_states = ((), updated_prior_state, updated_t_state1, updated_t_state2)
        return log_pz + log_hx1 + log_hx2, z, updated_states

    def inverse(params, state, log_pz, z, **kwargs):
        (A1, A2), prior_params, t_params1, t_params2 = params
        _, prior_state, t_state1, t_state2 = state

        # Get multiple keys if we're sampling
        key = kwargs.pop('key', None)
        if(key is not None):
            # Re-fill key for the rest of the flow
            k1, k2, k3 = random.split(key, 3)
        else:
            k1, k2, k3 = (None,)*3

        # Run the input through the prior
        log_pz, z, updated_prior_state = prior_inverse(prior_params, prior_state, log_pz, z, key=k1, **kwargs)

        # Split z
        if(kind == 'every_other'):
            z1, z2 = jnp.split(z[...,z_every_other_idx], jnp.array([z1_dim]), axis=-1)
        else:
            z1, z2 = jnp.split(z, jnp.array([z1_dim]), axis=-1)

        # Compute the bias and covariance conditioned on z1
        (log_diag_cov2, b2), updated_t_state2 = apply_fun(t_params2, t_state2, z1, key=k2, **kwargs)

        # Compute x2
        x2 = jnp.dot(z2, A2.T) + b2

        # Compute the bias and covariance conditioned on x2
        (log_diag_cov1, b1), updated_t_state1 = apply_fun(t_params1, t_state1, x2, key=k3, **kwargs)

        # Compute x1
        x1 = jnp.dot(z1, A1.T) + b1

        # Combine x
        if(kind == 'every_other'):
            x = jnp.concatenate([x1, x2], axis=-1)[...,x_regular_idx]
        else:
            x = jnp.concatenate([x1, x2], axis=-1)

        # Add noise
        assert 0, 'This is incomplete for some reason'

        # Compute N(x|Az + b, \Sigma).  This is just the log partition function.
        log_px1 = - 0.5*jnp.sum(log_diag_cov1) - 0.5*x.shape[-1]*jnp.log(2*jnp.pi)
        log_px2 = - 0.5*jnp.sum(log_diag_cov2) - 0.5*x.shape[-1]*jnp.log(2*jnp.pi)

        updated_states = ((), updated_prior_state, updated_t_state1, updated_t_state2)
        return log_pz + log_px1 + log_px2, x, updated_states

    return init_fun, forward, inverse

################################################################################################################


def split_x(x, idx):
    H, W, C = x.shape
    # W will be cut in half
    return x[idx > 0].reshape((H, W//2, C))

def recombine(z, index):
    # language=rst
    """
    Use a structured set of indices to create a matrix from a vector

    :param z: Flat input that contains the elements of the output matrix
    :param indices: An array of indices that correspond to values in z
    """
    return jnp.pad(z.ravel(), (1, 0))[index]

# Applies the upsampled z indices
recombine_vmapped = vmap(recombine, in_axes=(2, None), out_axes=2)


def CoupledUpSample(transform_fun, repeats, name='unnamed'):
    # language=rst
    """
    Up sample by just repeating consecutive values over specified axes

    :param repeats - The number of times to repeat.  Pass in (2, 1, 2), for example, to repeat twice over
                     the 0th axis, no repeats over the 1st axis, and twice over the 2nd axis
    """
    full_repeats = None
    apply_fun = None
    z_masks, z_shapes = None, None
    z_indices, upsampled_z_indices = None, None

    def init_fun(key, input_shape):
        keys = random.split(key, 3)
        x_shape = input_shape
        nonlocal full_repeats
        full_repeats = [repeats[i] if i < len(repeats) else 1 for i in range(len(x_shape))]
        z_shape = []
        for s, r in zip(x_shape, full_repeats):
            assert s%r == 0
            z_shape.append(s//r)
        z_shape = tuple(z_shape)
        Hz, Wz, Cz = z_shape
        Hx, Wx, Cx = x_shape

        # Going to be squeezing the splitting on channel axis
        z1_shape = (Hz//2, Wz//2, Cz*2)
        x2_shape = (Hx//2, Wx//2, Cx*2)

        # Initialize each of the flows.  apply_fun can be shared
        nonlocal apply_fun
        init_fun1, apply_fun = transform_fun(out_shape=x2_shape)
        init_fun2, _ = transform_fun(out_shape=x2_shape)

        # Initialize the transform function.
        # Should output bias and log diagonal covariance
        t_name1, (log_diag_cov_shape1, b_shape1), t_params1, t_state1 = init_fun1(keys[1], x2_shape)
        t_name2, (log_diag_cov_shape2, b_shape2), t_params2, t_state2 = init_fun2(keys[2], x2_shape) # For simplicity, will be passing in an upsampled z1 to this

        names = (name, t_name1, t_name2)
        params = ((), t_params1, t_params2)
        state = ((), t_state1, t_state2)
        return names, z_shape, params, state

    def forward(params, state, x, **kwargs):
        if(x.ndim == 4):
            return vmap(partial(forward, params, state, **kwargs), in_axes=(0, 0, None))(log_px, x, condition)

        _, t_params1, t_params2 = params
        _, t_state1, t_state2 = state

        # Get multiple keys if we're sampling
        key = kwargs.pop('key', None)
        if(key is not None):
            # Re-fill key for the rest of the flow
            k1, k2, k3, k4, k5 = random.split(key, 5)
        else:
            k1, k2, k3, k4, k5 = (None,)*5

        # Determine if we are batching or not
        is_batched = x.ndim == 4
        posterior_fun = vmap(upsample_posterior, in_axes=(0, 0, 0, None)) if is_batched else upsample_posterior

        # Split x
        x_squeezed = dilated_squeeze(x, (2, 2), (1, 1))
        x1, x2 = jnp.split(x_squeezed, 2, axis=-1)

        """ Posterior of N(x_1|Az_1 + b(x_2), Sigma(x_2)) """

        # Compute the bias and covariance conditioned on x2.  \sigma(x2), b(x2)
        (log_diag_cov1, b1), updated_t_state1 = apply_fun(t_params1, t_state1, x2, key=k1, **kwargs)
        log_diag_cov1 = -jax.nn.softplus(log_diag_cov1) - 3.0

        # Compute the posterior and the manifold penalty
        z1_mean, log_hx1, rm1_diag = posterior_fun(x1, b1, log_diag_cov1, full_repeats)

        # Sample z1
        if(key is not None):
            noise = random.normal(k2, z1_mean.shape)
            z1 = z1_mean + noise/jnp.sqrt(rm1_diag)
        else:
            z1 = z1_mean

        """ Posterior of N(x_2|Az_2 + b(z_1), Sigma(z_1)) """

        # Compute the bias and covariance conditioned on z1.  \sigma(z1), b(z1)
        (log_diag_cov2, b2), updated_t_state2 = apply_fun(t_params2, t_state2, upsample(full_repeats, z1), key=k3, **kwargs)
        log_diag_cov2 = -jax.nn.softplus(log_diag_cov2) - 3.0

        # Compute the posterior and the manifold penalty
        z2_mean, log_hx2, rm2_diag = posterior_fun(x2, b2, log_diag_cov2, full_repeats)

        # Sample z2
        if(key is not None):
            noise = random.normal(k4, z2_mean.shape)
            z2 = z2_mean + noise/jnp.sqrt(rm2_diag)
        else:
            z2 = z2_mean

        """ Combine z """
        z_squeezed = jnp.concatenate([z1, z2], axis=-1)
        z = dilated_unsqueeze(z_squeezed, (2, 2), (1, 1))

        # Return the full estimate of the integral and the updated
        updated_states = ((), updated_t_state1, updated_t_state2)
        return log_px + log_hx1 + log_hx2, z, updated_states

    def inverse(params, state, log_pz, z, **kwargs):
        if(z.ndim == 4):
            return vmap(partial(inverse, params, state, **kwargs), in_axes=(0, 0, None))(log_pz, z, condition)

        _, t_params1, t_params2 = params
        _, t_state1, t_state2 = state

        # Get multiple keys if we're sampling
        key = kwargs.pop('key', None)
        if(key is not None):
            # Re-fill key for the rest of the flow
            k1, k2, k3, k4, k5 = random.split(key, 5)
        else:
            k1, k2, k3, k4, k5 = (None,)*5

        # Split z
        z_squeezed = dilated_squeeze(z, (2, 2), (1, 1))
        z1, z2 = jnp.split(z_squeezed, 2, axis=-1)

        """ N(x_2|Az_2 + b(z_1), Sigma(z_1)) """

        # Compute the bias and covariance conditioned on z1.  \sigma(z1), b(z1)
        (log_diag_cov2, b2), updated_t_state2 = apply_fun(t_params2, t_state2, upsample(full_repeats, z1), key=k2, **kwargs)
        log_diag_cov2 = -jax.nn.softplus(log_diag_cov2) - 3.0

        # Compute the mean of x2
        x2_mean = upsample(full_repeats, z2) + b2

        # Sample x2
        if(key is not None):
            noise2 = jnp.exp(0.5*log_diag_cov2)*random.normal(k3, x2_mean.shape)
            x2 = x2_mean + noise2
        else:
            noise2 = jnp.zeros_like(x2_mean)
            x2 = x2_mean

        """ N(x_1|Az_1 + b(x_2), Sigma(x_2)) """

        # Compute the bias and covariance conditioned on x2.  \sigma(x2), b(x2)
        (log_diag_cov1, b1), updated_t_state1 = apply_fun(t_params1, t_state1, x2, key=k4, **kwargs)
        log_diag_cov1 = -jax.nn.softplus(log_diag_cov1) - 3.0

        # Compute the mean of x2
        x1_mean = upsample(full_repeats, z1) + b1

        # Sample x2
        if(key is not None):
            noise1 = jnp.exp(0.5*log_diag_cov1)*random.normal(k5, x1_mean.shape)
            x1 = x1_mean + noise1
        else:
            noise1 = jnp.zeros_like(x1_mean)
            x1 = x1_mean

        # Combine x
        x_squeezed = jnp.concatenate([x1, x2], axis=-1)
        x = dilated_unsqueeze(x_squeezed, (2, 2), (1, 1))

        # Compute N(x1|Az1 + b(x2), Sigma(x2))N(x2|Az2 + b(z1), Sigma(z1))
        log_px1 = -0.5*jnp.sum(noise1*jnp.exp(-0.5*log_diag_cov1)*noise1) - 0.5*jnp.sum(log_diag_cov1) - 0.5*x1.shape[-1]*jnp.log(2*jnp.pi)
        log_px2 = -0.5*jnp.sum(noise2*jnp.exp(-0.5*log_diag_cov2)*noise2) - 0.5*jnp.sum(log_diag_cov2) - 0.5*x2.shape[-1]*jnp.log(2*jnp.pi)

        updated_states = ((), updated_t_state1, updated_t_state2)
        return log_pz + log_px1 + log_px2, x, updated_states

    return init_fun, forward, inverse

################################################################################################################
