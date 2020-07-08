
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
