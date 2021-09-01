import click
from typing import Callable
import numpy as np
from diffbank.bank import Bank
from jax import random
import jax.numpy as jnp
from diffbank.waveforms import kappa4D
import jax

import matplotlib.pyplot as plt

##### Frequency settings
f_u = 512.0  # Hz
f_l = 32.0  # Hz
N_fbins = 2000
#####

m1_range = (1.001, 5.0)
m2_range = (1.0, 5.0)
chi_range = (-0.6, 0.6)
kappa2PN_range = (0.0, 250.0)

x_low = np.array([-0.6, 0.0, 0.6])
y_low = np.array([0.0, 0.0, 0.0])

x_up = np.array([-0.6, 0.0, 0.6])
y_up = np.array([250.0, 150.0, 250.0])


def sampler(key, n):
    ms = random.uniform(
        key,
        minval=jnp.array([m1_range[0], m2_range[0]]),
        maxval=jnp.array([m1_range[1], m2_range[1]]),
        shape=(n, 2),
    )
    key, subkey = random.split(key)
    chis = random.uniform(
        subkey,
        minval=jnp.array(chi_range[0]),
        maxval=jnp.array(chi_range[1]),
        shape=(n, 1),
    )
    key, subkey = random.split(key)

    K_up = jnp.interp(chis, x_up, y_up)
    K_low = jnp.interp(chis, x_low, y_low)
    Ks = random.uniform(
        subkey,
        minval=K_low,
        maxval=K_up,
        shape=(n, 1),
    )
    ms_correct = jnp.stack(
        [
            ms.max(axis=1),
            ms.min(axis=1),
        ]
    ).T
    return jnp.hstack((ms_correct, chis, Ks))


def get_Sn_O3a() -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Get interpolator for noise curve.
    """
    xp, yp = np.loadtxt("O3a_Livingston_ASD.txt", unpack=True)
    PSD = yp ** 2
    return lambda f: jnp.interp(f, xp, PSD, left=jnp.inf, right=jnp.inf)


# def get_Sn_aLIGO() -> Callable[[jnp.ndarray], jnp.ndarray]:
#     """
#     Get interpolator for noise curve.
#     """
#     xp, yp = np.loadtxt("LIGO-P1200087-v18-aLIGO_MID_LOW.txt", unpack=True)
#     return lambda f: jnp.interp(f, xp, yp, left=jnp.inf, right=jnp.inf)


def naive_sampler(key, n):
    theta = random.uniform(
        key,
        minval=jnp.array([m1_range[0], m2_range[0], chi_range[0], kappa2PN_range[0]]),
        maxval=jnp.array([m1_range[1], m2_range[1], chi_range[1], kappa2PN_range[1]]),
        shape=(n, 4),
    )
    return theta


def accept(vars: jnp.ndarray) -> jnp.ndarray:
    """
    Returns 1 if vars is in bounds, 0 otherwise.
    """
    var1, var2, var3, var4 = (
        vars[..., 0],
        vars[..., 1],
        vars[..., 2],
        vars[..., 3],
    )
    var1_clipped = jnp.clip(var1, m1_range[0], m1_range[1])
    var2_clipped = jnp.clip(var2, m2_range[0], var1)
    var3_clipped = jnp.clip(var3, chi_range[0], chi_range[1])
    var4_clipped = jnp.clip(
        var4, jnp.interp(var3, x_low, y_low), jnp.interp(var3, x_up, y_up)
    )
    # Check if clipped and unclipped point are equal
    return jnp.logical_and(
        jnp.logical_and(
            jnp.logical_and(var1 == var1_clipped, var2 == var2_clipped),
            var3 == var3_clipped,
        ),
        var4 == var4_clipped,
    ).astype(jnp.float64)


def test_sampler(key):
    n = 10
    samples = sampler(key, n)
    print(samples)


def gen_4D_kappabank(key):
    minimum_match = 0.8
    eta = 0.9
    fs = jnp.linspace(f_l, f_u, N_fbins)
    Sn = get_Sn_O3a()

    bank = Bank(
        kappa4D.Amp,
        kappa4D.Psi,
        fs,
        Sn,
        sampler,
        m_star=1 - minimum_match,
        eta=eta,
        name="banks/kappa4D_O3a_newconvcriteria2",
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

    theta_max = jnp.array(
        [m1_range[1], m2_range[0], chi_range[0], jnp.interp(chi_range[0], x_up, y_up)]
    )
    print(theta_max, bank.density_fun(theta_max))
    bank.density_max = bank.density_fun(theta_max)

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
    bank.fill_bank(subkey)

    # key, subkey = random.split(key)
    # N = 100
    # bank.calculate_bank_effectualness(subkey, N)

    bank.save()


def check_saved_bank(key):

    Sn = get_Sn_O3a()
    loaded_bank = Bank.load(
        "banks/kappa4D_O3a_newconvcriteria2.npz", kappa4D.Amp, kappa4D.Psi, Sn, sampler
    )
    print(loaded_bank.eta)
    print(loaded_bank.m_star)
    print(loaded_bank.n_templates)
    print(loaded_bank.templates[:, 0].min(), loaded_bank.templates[:, 0].max())
    print(loaded_bank.templates[:, 1].min(), loaded_bank.templates[:, 1].max())
    print(loaded_bank.templates[:, 2].min(), loaded_bank.templates[:, 2].max())
    print(loaded_bank.templates[:, 3].min(), loaded_bank.templates[:, 3].max())

    np.savetxt("banks/kappabank4D_eta09_mm08_search.txt", loaded_bank.templates)

    plt.figure()
    plt.scatter(loaded_bank.templates[:, 0], loaded_bank.templates[:, 1], s=1)
    plt.xlabel("M1")
    plt.ylabel("M2")
    plt.savefig("figures/kappabank4D_eta09_mm08_search_m1m2.pdf")

    plt.figure()
    plt.scatter(loaded_bank.templates[:, 2], loaded_bank.templates[:, 3], s=1)
    plt.xlabel("Chi")
    plt.ylabel("Effective Kappa")
    plt.savefig("figures/kappabank4D_eta09_mm08_search_chikappa.pdf")

    key, subkey = random.split(key)
    N = 1000
    loaded_bank.calc_bank_effectualness(subkey, N)
    loaded_bank.save()

    print(
        "Mean and median",
        loaded_bank.effectualnesses.mean(),
        np.median(loaded_bank.effectualnesses),
    )

    plt.figure()
    plt.hist(loaded_bank.effectualnesses, 50, density=True, histtype='step', cumulative=True)
    plt.yscale("log")
    plt.xlabel("Effectualness")
    plt.ylabel("CDF")
    plt.savefig("figures/kappabank4D_eta09_mm08_search_CDF.pdf")

    plt.figure()
    sc = plt.scatter(loaded_bank.effectualness_points[:, 0], loaded_bank.effectualness_points[:, 1], c=loaded_bank.effectualnesses, s=3)
    plt.colorbar(sc)
    plt.xlabel("M1")
    plt.ylabel("M2")
    plt.savefig("figures/kappabank4D_eta09_mm08_search_m1m2_effectualness.pdf")

    plt.figure()
    sc = plt.scatter(loaded_bank.effectualness_points[:, 2], loaded_bank.effectualness_points[:, 3], c=loaded_bank.effectualnesses, s=3)
    plt.colorbar(sc)
    plt.xlabel("Chi")
    plt.ylabel("Effective Kappa")
    plt.savefig("figures/kappabank4D_eta09_mm08_search_chikappa_effectualness.pdf")


@click.command()
@click.option("--seed", type=int, help="PRNG seed")
def main(seed):
    key = random.PRNGKey(seed)
    # gen_4D_kappabank(key)
    # test_sampler(key)
    # calc_number_templates(key)
    check_saved_bank(key)
    # test_n_eff_pts()


if __name__ == "__main__":
    main()
