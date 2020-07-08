import src.flows as flows
from src.tests.nf_test import standard_layer_tests, image_layer_test, unit_test, flow_test
from src.tests.nif_test import nif_test
import jax.numpy as jnp
from debug import *

import src.flows as nux
import jax
from jax import jit, vmap, random
from functools import partial
import src.util as util

if(__name__ == '__main__'):

    standard_layer_tests()
    image_layer_test()
    unit_test()


    nif_test()