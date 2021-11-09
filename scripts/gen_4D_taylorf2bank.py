import click
from typing import Callable
import numpy as np
from diffbank.bank import Bank
from jax import random
import jax.numpy as jnp
from diffbank.waveforms import taylorF2

# import jax

# from diffbank.noise import Sn_aLIGO

import matplotlib.pyplot as plt

##### Frequency settings
f_u = 512.0  # Hz
f_l = 24.0  # Hz
N_fbins = 4880
#####

m1_range = (1.001, 3.0)
m2_range = (1.0, 3.0)

# Mt_range = (2, 8)
# eta_range = (0.1875, 0.2499)
chi1_range = (-0.99, 0.99)
chi2_range = (-0.99, 0.99)


def sampler(key, n):
    ms = random.uniform(
        key,
        minval=jnp.array([m1_range[0], m2_range[0]]),
        maxval=jnp.array([m1_range[1], m2_range[1]]),
        shape=(n, 2),
    )
    key, subkey = random.split(key)
    chi1s = random.uniform(
        subkey,
        minval=jnp.array(chi1_range[0]),
        maxval=jnp.array(chi1_range[1]),
        shape=(n, 1),
    )
    key, subkey = random.split(key)
    chi2s = random.uniform(
        subkey,
        minval=jnp.array(chi2_range[0]),
        maxval=jnp.array(chi2_range[1]),
        shape=(n, 1),
    )
    ms_correct = jnp.stack(
        [
            ms.max(axis=1),
            ms.min(axis=1),
        ]
    ).T
    return jnp.hstack((ms_correct, chi1s, chi2s))


# def get_Sn_O3a() -> Callable[[jnp.ndarray], jnp.ndarray]:
#     """
#     Get interpolator for noise curve.
#     """
#     xp, yp = np.loadtxt("O3a_Livingston_ASD.txt", unpack=True)
#     PSD = yp ** 2
#     return lambda f: jnp.interp(f, xp, PSD, left=jnp.inf, right=jnp.inf)


def get_Sn_aLIGO() -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Get interpolator for noise curve.
    """
    xp, yp = np.loadtxt("LIGO-P1200087-v18-aLIGO_MID_LOW.txt", unpack=True)
    PSD = yp ** 2
    return lambda f: jnp.interp(f, xp, PSD, left=jnp.inf, right=jnp.inf)


# def sampler(key, n):
#     theta = random.uniform(
#         key,
#         minval=jnp.array([Mt_range[0], eta_range[0], chi1_range[0], chi2_range[0]]),
#         maxval=jnp.array([Mt_range[1], eta_range[1], chi1_range[1], chi2_range[1]]),
#         shape=(n, 4),
#     )
#     return theta


# def accept(vars: jnp.ndarray) -> jnp.ndarray:
#     """
#     Returns 1 if vars is in bounds, 0 otherwise.
#     """
#     var1, var2, var3, var4 = (
#         vars[..., 0],
#         vars[..., 1],
#         vars[..., 2],
#         vars[..., 3],
#     )
#     var1_clipped = jnp.clip(var1, Mt_range[0], Mt_range[1])
#     var2_clipped = jnp.clip(var2, eta_range[0], eta_range[1])
#     var3_clipped = jnp.clip(var3, chi1_range[0], chi1_range[1])
#     var4_clipped = jnp.clip(var4, chi2_range[0], chi2_range[1])
#     # Check if clipped and unclipped point are equal
#     return jnp.logical_and(
#         jnp.logical_and(
#             jnp.logical_and(var1 == var1_clipped, var2 == var2_clipped),
#             var3 == var3_clipped,
#         ),
#         var4 == var4_clipped,
#     ).astype(jnp.float64)


def test_sampler(key):
    n = 10
    samples = sampler(key, n)
    print(samples)


def gen_4D_taylorf2bank(key, seed, kind):
    minimum_match = 0.95
    eta = 0.9
    fs = jnp.linspace(f_l, f_u, N_fbins)
    Sn = get_Sn_aLIGO()

    bank = Bank(
        taylorF2.Amp,
        taylorF2.Psi,
        fs,
        Sn,
        1 - minimum_match,
        eta,
        sampler,
        name="banks/TaylorF2-%s-%i"
        % (
            kind,
            seed,
        ),
    )

    ############################
    # For testing
    # key, subkey = random.split(key)
    # N = 10000
    # thetas = sampler(subkey, N)

    # plt.scatter(thetas[:, 2], thetas[:, 3], s=1)
    # plt.xlabel("chi")
    # plt.ylabel("K")
    # plt.savefig("figures/test_4D_sampler.pdf")
    # densities = jax.lax.map(bank.get_density, thetas)
    # print(densities.min(), densities.max())
    ############################

    theta_max = jnp.array([m1_range[0], m2_range[0], chi1_range[0], chi2_range[1]])
    bank.ratio_max = bank.density_fun(theta_max)

    ############################
    # Making a fake bank for Cody
    # key, subkey = random.split(key)
    # N_template_test = 1000
    # test_templates = bank.gen_templates_rejection(subkey, N_template_test)
    # np.savetxt("Cody_testbank.txt", test_templates)

    # plt.figure()
    # plt.scatter(test_templates[:, 0], test_templates[:, 1], s=1)
    # plt.xlabel("m1")
    # plt.ylabel("m2")
    # plt.savefig("figures/test_4D_sampler_m1m2.pdf")

    # plt.figure()
    # plt.scatter(test_templates[:, 2], test_templates[:, 3], s=1)
    # plt.xlabel("chi")
    # plt.ylabel("K")
    # plt.savefig("figures/test_4D_sampler_chiK.pdf")
    ############################

    key, subkey = random.split(key)
    bank.fill_bank(subkey, method=kind)
    bank.save()

    key, subkey = random.split(key)
    N = 1000
    bank.calc_bank_effectualness(subkey, N)

    bank.save()


# def calc_number_templates(key):
#     minimum_match = 0.9
#     eta = 0.99
#     fs = jnp.linspace(f_l, f_u, N_fbins)
#     # Sn = get_Sn_O3a()

#     bank = Bank(
#         taylorF2.Amp,
#         taylorF2.Psi,
#         fs,
#         Sn_aLIGO,
#         sampler,
#         m_star=1 - minimum_match,
#         eta=eta,
#         name="banks/4D_taylorf2_O3a",
#     )

#     key, subkey = random.split(key)
#     proposal_vol = (
#         (Mt_range[1] - Mt_range[0])
#         * (eta_range[1] - eta_range[0])
#         * (chi1_range[1] - chi1_range[0])
#         * (chi2_range[1] - chi2_range[0])
#     )

#     theta_max = jnp.array([Mt_range[0], eta_range[0], chi1_range[0], chi2_range[1]])
#     bank.density_max = bank.get_density(theta_max)
#     print(bank.density_max)

#     N = 10
#     n_per_point = 10
#     # proposals = naive_sampler(subkey, N)
#     # in_bounds = accept(proposals)
#     # naive_vol = in_bounds.mean() * proposal_vol
#     # naive_vol_err = in_bounds.std() * proposal_vol / jnp.sqrt(len(proposals))

#     key, subkey = random.split(key)
#     N_templates, N_templates_err = bank.estimate_n_templates(
#         subkey,
#         N,
#         n_per_point,
#         accept,
#         proposal_vol,
#         0.0,
#     )
#     print(N_templates, N_templates_err)


def check_saved_bank():

    # Sn = get_Sn_O3a()
    Sn = get_Sn_aLIGO()
    loaded_bank = Bank.load(
        "banks/taylorF2_O2_IAScomp.npz", taylorF2.Amp, taylorF2.Psi, Sn, sampler
    )
    print(loaded_bank.eta)
    print(loaded_bank.m_star)
    print(loaded_bank.n_templates)
    print(loaded_bank.templates[:, 0].min(), loaded_bank.templates[:, 0].max())
    print(loaded_bank.templates[:, 1].min(), loaded_bank.templates[:, 1].max())
    print(loaded_bank.templates[:, 2].min(), loaded_bank.templates[:, 2].max())
    print(loaded_bank.templates[:, 3].min(), loaded_bank.templates[:, 3].max())


@click.command()
@click.option("--seed", type=int, help="PRNG seed")
@click.option("--kind", type=str, help="Type of bank")
def main(seed, kind):
    key = random.PRNGKey(seed)
    gen_4D_taylorf2bank(key, seed, kind)
    # test_sampler(key)
    # calc_number_templates(key)
    # check_saved_bank()


if __name__ == "__main__":
    main()
