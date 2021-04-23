from jax.config import config

config.update("jax_enable_x64", True)
import numpy as np
from jax import jacfwd, jacrev
import jax.numpy as jnp
import jax
from jax import grad, jit, vmap
from jax import random
from functools import partial
from threePN_simple import Amp_35PN, Psi_35PN
from math import pi
from core import metric
from core import utils


m_star = 1 - 0.95
p_ranges = jnp.array([[1, 3], [1, 3]])
N_test = 1000
eta = 0.9999

f = np.linspace(10, 500, 10000)
print("Calculating the number of templates")
N_templates = utils.N_templates(
    p_ranges, N_test, eta, m_star, Amp_35PN, Psi_35PN, f, utils.Sn_func
)
print("Number of Templates: %.0f" % N_templates)
print("Starting to generate a bank")
print(
    utils.rejection_sampling(
        p_ranges, Amp_35PN, Psi_35PN, f, utils.Sn_func, Nmax=int(N_templates)
    )
)
print("Finished generating bank")
