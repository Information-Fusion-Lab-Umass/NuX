
################################################################################################################

def key_wrap(flow, key):
    # language=rst
    """
    Add the ability to specify a key to initialize
    """
    _init_fun, forward, inverse = flow

    def init_fun(unused_key, input_shape):
        name, output_shape, params, state = _init_fun(key, input_shape)
        return name, output_shape, params, state

    return init_fun, forward, inverse

def named_wrap(flow, name='unnamed'):
    _init_fun, _forward, _inverse = flow

    def init_fun(key, input_shape):
        _name, output_shape, params, state = _init_fun(key, input_shape)
        return name, output_shape, params, state

    def forward(params, state, x, **kwargs):
        log_det, z, updated_state = _forward(params, state, x, **kwargs)
        return log_det, z, updated_state

    def inverse(params, state, z, **kwargs):
        log_det, x, updated_state = _inverse(params, state, z, **kwargs)
        return log_det, x, updated_state

    return init_fun, forward, inverse

################################################################################################################

def Augment(flow, sampler, name='unnamed'):
    # language=rst
    """
    Run a normalizing flow in an augmented space https://arxiv.org/pdf/2002.07101.pdf

    :param flow: The normalizing flow
    :param sampler: Function to sample from the convolving distribution
    """
    _init_fun, _forward, _inverse = flow

    def init_fun(key, input_shape):
        augmented_input_shape = input_shape[:-1] + (2*input_shape[-1],)
        return _init_fun(key, augmented_input_shape)

    def forward(params, state, x, **kwargs):
        key = kwargs.pop('key', None)
        if(key is None):
            assert 0, 'Need a key for this'
        k1, k2 = random.split(key, 2)

        # Sample e and concatenate it to x
        e = random.normal(k1, x.shape)
        xe = jnp.concatenate([x, e], axis=-1)

        return _forward(params, state, xe, key=k2, **kwargs)

    def inverse(params, state, z, **kwargs):
        key = kwargs.pop('key', None)
        if(key is None):
            assert 0, 'Need a key for this'
        k1, k2 = random.split(key, 2)

        x, e = jnp.split(z, axis=-1)

        return _inverse(params, state, x, key=k2, **kwargs)

    return init_fun, forward, inverse

def Debug(message,
          print_init_shape=True,
          print_forward_shape=False,
          print_inverse_shape=False,
          compare_vals=False,
          name='unnamed'):
    # language=rst
    """
    Help debug shapes

    :param print_init_shape: Print the shapes
    :param print_forward_shape: Print the shapes
    :param print_inverse_shape: Print the shapes
    :param compare_vals: Print the difference between the value of the forward pass and the reconstructed
    """

    saved_val = None

    def init_fun(key, input_shape):
        if(print_init_shape):
            print(message, 'input_shape', input_shape)
        return name, input_shape, (), ()

    def forward(params, state, x, **kwargs):
        if(print_forward_shape):
            if(isinstance(x, tuple) or isinstance(x, list)):
                print(message, 'x shapes', [_x.shape for _x in x], 'log_px shapes', [_x.shape for _x in log_px])
            else:
                print(message, 'x.shape', x.shape, 'log_px.shape', log_px.shape)

        if(compare_vals):
            nonlocal saved_val
            saved_val = x

        return log_px, x, state

    def inverse(params, state, z, **kwargs):
        if(print_inverse_shape):
            if(isinstance(z, tuple) or isinstance(z, list)):
                print(message, 'z shapes', [_z.shape for _z in z], 'log_pz shapes', [_z.shape for _z in log_pz])
            else:
                print(message, 'z.shape', z.shape, 'log_pz.shape', log_pz.shape)
        if(compare_vals):
            if(isinstance(z, tuple) or isinstance(z, list)):
                print(message, 'jnp.linalg.norm(z - saved_val)', [jnp.linalg.norm(_z - _x) for _x, _z in zip(saved_val, z)])
            else:
                print(message, 'jnp.linalg.norm(z - saved_val)', jnp.linalg.norm(z - saved_val))

        return log_pz, z, state

    return init_fun, forward, inverse

