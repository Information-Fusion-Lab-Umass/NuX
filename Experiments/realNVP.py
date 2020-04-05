from jax import random, vmap, jit, value_and_grad
from jax.experimental import optimizers, stax
import jax.numpy as np
import staxplusplus as spp
from ..normalizing_flows import *
from ..util import *
import matplotlib.pyplot as plt
from ..datasets import get_cifar10_data, get_mnist_data
import jax.nn.initializers
z_dim = 100

def Transform(out_shape, n_hidden_layers=5, layer_size=1024):
    out_dim = out_shape[-1]
    # Build the s and t networks that xb will be fed into
    layer_sizes = [layer_size for _ in range(n_hidden_layers)]
    log_s_out = spp.sequential(spp.Dense(out_dim), spp.Tanh())
    t_out = spp.sequential(spp.Dense(out_dim))
    dense_layers = [spp.Dense(layer_size), spp.Relu()]*n_hidden_layers
    coupling_param_architecture = spp.sequential(*dense_layers, spp.FanOut(2), spp.parallel(log_s_out, t_out))

    # Split x into xa, xb and feed xb into its path
    return coupling_param_architecture

def ResidualBlock(n_channels, name_prefix=''):
    network = spp.sequential(spp.WeightNormConv(n_channels, filter_shape=(1, 1), padding=((0, 0), (0, 0)), name='%s_wn0'%name_prefix),
                             spp.Relu(name='%s_relu0'%name_prefix), 
                             spp.WeightNormConv(n_channels, filter_shape=(3, 3), padding=((1, 1), (1, 1)), name='%s_wn1'%name_prefix),
                             spp.Relu(name='%s_relu1'%name_prefix),
                             spp.WeightNormConv(n_channels, filter_shape=(1, 1), padding=((0, 0), (0, 0)), name='%s_wn2'%name_prefix))
    return spp.Residual(network)

def ResNet(out_shape, n_filters=64, n_blocks=8, name_prefix=''):
    height, width, channel = out_shape

    res_blocks = [ResidualBlock(n_filters, name_prefix='%s_res_%d'%(name_prefix, i)) for i in range(n_blocks)]

    network = spp.sequential(spp.WeightNormConv(n_filters, filter_shape=(3, 3), padding=((1, 1), (1, 1)), name='%s_wn0'%name_prefix),
                             spp.Relu(name='%s_relu0'%(name_prefix)),
                             *res_blocks,
                             spp.Relu(name='%s_relu1'%(name_prefix)),
                             spp.WeightNormConv(2*channel, filter_shape=(3, 3), padding=((1, 1), (1, 1)), name='%s_wn1'%name_prefix, b_init=jax.nn.initializers.zeros))

    return spp.sequential(network, spp.Split(2, axis=-1), spp.parallel(spp.Tanh(), spp.Identity()))

def RealNVP():
    
    checker_transforms1 = sequential_flow(MaskedAffineCoupling(partial(ResNet, 
                                                                       name_prefix='AC_0'), 
                                                                       mask_type='checkerboard', 
                                                                       top_left_zero=False),
                                          ActNorm(name='an_0'),
                                          MaskedAffineCoupling(partial(ResNet, 
                                                                       name_prefix='AC_1'), 
                                                                       mask_type='checkerboard', 
                                                                       top_left_zero=True),
                                          ActNorm(name='an_1'),
                                          MaskedAffineCoupling(partial(ResNet, 
                                                                       name_prefix='AC_2'), 
                                                                       mask_type='checkerboard', 
                                                                       top_left_zero=False),
                                          ActNorm(name='an_2'),
                                          MaskedAffineCoupling(partial(ResNet, 
                                                                       name_prefix='AC_3'), 
                                                                       mask_type='checkerboard', 
                                                                       top_left_zero=True),
                                          ActNorm(name='an_3'))
    
    channel_transforms = sequential_flow(MaskedAffineCoupling(partial(ResNet, 
                                                                      name_prefix='AC_4'), 
                                                                      mask_type='channel_wise'),
                                         Reverse(),
                                         ActNorm(name='an_4'),
                                         MaskedAffineCoupling(partial(ResNet, 
                                                                      name_prefix='AC_5'), 
                                                                      mask_type='channel_wise'),
                                         Reverse(),
                                         ActNorm(name='an_5'),
                                         MaskedAffineCoupling(partial(ResNet, 
                                                                      name_prefix='AC_6'), 
                                                                      mask_type='channel_wise'))
    
    checker_transforms2 = sequential_flow(MaskedAffineCoupling(partial(ResNet, 
                                                                       name_prefix='AC_7'), 
                                                                       mask_type='checkerboard', 
                                                                       top_left_zero=False),
                                          ActNorm(name='an_6'),
                                          MaskedAffineCoupling(partial(ResNet, 
                                                                       name_prefix='AC_8'), 
                                                                       mask_type='checkerboard', 
                                                                       top_left_zero=True),
                                          ActNorm(name='an_7'),
                                          MaskedAffineCoupling(partial(ResNet, 
                                                                       name_prefix='AC_9'), 
                                                                       mask_type='checkerboard', 
                                                                       top_left_zero=False))


    real_nvp = sequential_flow(Dequantization(scale=2**quantize_level_bits), 
                               Logit(), 
                               checker_transforms1, 
                               Reverse(),
                               CheckerboardSqueeze(), 
                               Reverse(),
                               channel_transforms, 
                               Reverse(),
                               CheckerboardUnSqueeze(),
                               checker_transforms2,
                               UnitGaussianPrior(axis=(-3, -2, -1)))
    return real_nvp




flow = RealNVP()
init_fun, forward, inverse = flow

key = random.PRNGKey(0)
names, output_shape, params, state = init_fun(key, x_train.shape[1:], ())
output_shape




actnorm_names = [name for name in tree_flatten(names)[0] if 'an' in name]

batch_size = 2
seed_steps = 100
flat_params, unflatten = ravel_pytree(params)
for i in range(seed_steps):
    key, *keys = random.split(key, 3)
    
    # Get the next batch of data
    batch_idx = random.randint(keys[0], (batch_size,), minval=0, maxval=x_train.shape[0])
    x_batch = x_train[batch_idx,:]
    
    # Compute the seeded parameters
    new_params = flow_data_dependent_init(x_batch, actnorm_names, names, params, state, forward, (), 'actnorm_seed', key=key)

    # Compute a running mean of the parameters
    new_flat_params, _ = ravel_pytree(new_params)
    flat_params = i/(i + 1)*flat_params + new_flat_params/(i + 1)
    params = unflatten(flat_params)



@jit
def nll(params, state, x, **kwargs):
    cond = ()
    log_px, z, updated_state = forward(params, state, np.zeros(x.shape[0]), x, cond, **kwargs)
    return -np.mean(log_px), updated_state
#     flat_params, _ = ravel_pytree(params)
#     return -np.mean(log_px) + 0.005*np.linalg.norm(flat_params), updated_state

# Create the gradient function
valgrad = value_and_grad(nll, has_aux=True)
valgrad = jit(valgrad)




opt_init, opt_update, get_params = optimizers.adam(0.0001)
opt_update = jit(opt_update)
opt_state = opt_init(params)

losses = []




batch_size = 2


for i in range(100):
    key, *keys = random.split(key, 3)
    
    batch_idx = random.randint(keys[0], (batch_size,), minval=0, maxval=x_train.shape[0])
    x_batch = x_train[batch_idx,:]
    
    params = get_params(opt_state)
    (val, state), grad = valgrad(params, state, x_batch, key=keys[1], test=TRAIN)
#     if(np.isnan(val) or np.any(np.isnan(ravel_pytree(grad)[0]))):
#         assert 0, 'NaN loss'
    val = val/np.prod(x_train.shape[1:])
    opt_state = opt_update(i, grad, opt_state)
    
    losses.append(val)
    if(i % 10 == 0):
    	print('Negative Log Likelihood: %5.3f'%(val))
    
    if(i%10 == 0):
        n_samples = 8
        z = random.normal(key, (n_samples, z_dim))
        _, fz, _ = inverse(params, state, np.zeros(n_samples), z, (), test=TEST)
        fz /= (2.0**quantize_level_bits) # Put the image (mostly) between 0 and 1

        n_cols = 8
        n_rows = int(np.ceil(n_samples/n_cols))

        fig, axes = plt.subplots(n_rows, n_cols); axes = axes.ravel()
        fig.set_size_inches(2*n_cols, 2*n_rows)

        for i, ax in enumerate(axes):
            ax.imshow(fz[i])
            
        plt.savefig('realnvp_out/image_at_sample' + str(i) + '.png')
        plt.close()


plt.plot(losses)
plt.savefig('realnvp_out/losses' + str(i) + '.png')
plt.close()























