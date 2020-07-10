import nux.flows as flows
from nux.tests.nf_test import standard_layer_tests, image_layer_test, unit_test, flow_test
from nux.tests.nif_test import nif_test
import jax.numpy as jnp
from debug import *

import nux.flows as nux
import jax
from jax import jit, vmap, random
from functools import partial
import nux.util as util

if(__name__ == '__main__'):

    standard_layer_tests()
    image_layer_test()
    nif_test()