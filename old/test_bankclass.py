from jax.config import config

config.update("jax_enable_x64", True)
import numpy as np
from jax import jacfwd, jacrev
import jax.numpy as jnp
import jax
from jax import grad, jit, vmap
from jax import random
from functools import partial

# from kappa_waveform import Amp_35PN, Psi_35PN
from threePN_simple import Amp_35PN, Psi_35PN
from math import pi
from core import metric
from core import utils
from core import bank
import matplotlib.pyplot as plt

m_star = 1 - 0.97
p_ranges = np.array([[1, 3], [1, 3]])
N_test = 1000
eta = 0.9999

f = np.linspace(10, 500, 10000)

print("Testing the bank generation classmethod")

kappa_bank = bank.Bank.genbank_rejection_sampling(
    p_ranges,
    Amp_35PN,
    Psi_35PN,
    f,
    utils.Sn_func,
    utils.gen_samples_from_ranges,
    m_star,
    eta,
)

print(
    kappa_bank.prange,
    kappa_bank.dimension,
    kappa_bank.Ntemplates,
    # kappa_bank.bank_points,
)

plt.scatter(kappa_bank.bank_points[:, 0], kappa_bank.bank_points[:, 1], s=1)
plt.xlabel("m1")
plt.ylabel("m2")
plt.savefig("figures/test_bankclass.pdf", bbox_inches="tight")


# Now lets test the effectualness
kappa_bank.test_bank_effectualness(500)

print(
    kappa_bank.effectualness,
    kappa_bank.median_effectualness,
    kappa_bank.mean_effectualness,
    kappa_bank.effectualness_97,
)
plt.figure(figsize=(6, 5))

plt.hist(kappa_bank.effectualness, bins=20, histtype="step", color="C0")
plt.hist(kappa_bank.effectualness, bins=20, alpha=0.4, color="C0")
plt.xlim(0.05, 1.001)
plt.xlabel("Effectualness")
plt.ylabel("PDF")

plt.savefig("figures/test_PDF.pdf", bbox_inches="tight")
