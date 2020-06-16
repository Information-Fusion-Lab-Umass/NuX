import src.flows as flows
from src.tests.nf_test import standard_layer_tests, image_layer_test, unit_test
import jax.numpy as jnp
import jax
from jax import random
from debug import *

if(__name__ == '__main__'):

    # standard_layer_tests()
    # image_layer_test()
    unit_test()