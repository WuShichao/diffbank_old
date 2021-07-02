import click
from functools import partial
from typing import Callable
import numpy as np

# import matplotlib.pyplot as plt

from diffbank.bank import Bank

# from diffbank.constants import C, G, MSUN
# from diffbank.utils import ms_to_Mc_eta
from diffbank.utils import gen_templates_rejection
from jax import random

import jax
import jax.numpy as jnp
from jax import jit
from diffbank.waveforms import kappa6D
from diffbank.metric import get_g

"""
Tests consistency of random template bank
"""
f_u = 512.0  # Hz
f_l = 32.0  # Hz
# S_0 = 1e-46  # arbitrary

Mt_range = (2, 6)
eta_range = (0.1875, 0.2499)
# th0_range = (0.3 * 1e4, 1.0 * 1e4)
# th3_range = (3.0 * 1e2, 3.5 * 1e2)
chi1_range = (0.0, 0.2)
chi2_range = (0.0, 0.2)
k1_range = (1.0, 4.0)
k2_range = (1.0, 4.0)


def get_Sn_aLIGO() -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Get interpolator for noise curve.
    """
    xp, yp = np.loadtxt("LIGO-P1200087-v18-aLIGO_MID_LOW.txt", unpack=True)
    return lambda f: jnp.interp(f, xp, yp, left=jnp.inf, right=jnp.inf)


######################## 4D versions


# def propose(key, n):
#     """
#     Proposal distribution for var rejection sampling.
#     """
#     return random.uniform(
#         key,
#         shape=(
#             n,
#             4,
#         ),
#         minval=jnp.stack((th0_range[0], th3_range[0], chi1_range[0], chi2_range[0])),
#         maxval=jnp.stack((th0_range[1], th3_range[1], chi1_range[1], chi2_range[1])),
#     )


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
#     var1_clipped = jnp.clip(var1, th0_range[0], th0_range[1])
#     var2_clipped = jnp.clip(var2, th3_range[0], th3_range[1])
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


def propose(key, n):
    """
    Proposal distribution for var rejection sampling.
    """
    return random.uniform(
        key,
        shape=(
            n,
            6,
        ),
        minval=jnp.stack(
            (
                Mt_range[0],
                eta_range[0],
                chi1_range[0],
                chi2_range[0],
                k1_range[0],
                k2_range[0],
            )
        ),
        maxval=jnp.stack(
            (
                Mt_range[1],
                eta_range[1],
                chi1_range[1],
                chi2_range[1],
                k1_range[1],
                k2_range[1],
            )
        ),
    )


def accept(vars: jnp.ndarray) -> jnp.ndarray:
    """
    Returns 1 if vars is in bounds, 0 otherwise.
    """
    var1, var2, var3, var4, var5, var6 = (
        vars[..., 0],
        vars[..., 1],
        vars[..., 2],
        vars[..., 3],
        vars[..., 4],
        vars[..., 5],
    )
    var1_clipped = jnp.clip(var1, Mt_range[0], Mt_range[1])
    var2_clipped = jnp.clip(var2, eta_range[0], eta_range[1])
    var3_clipped = jnp.clip(var3, chi1_range[0], chi1_range[1])
    var4_clipped = jnp.clip(var4, chi2_range[0], chi2_range[1])
    var5_clipped = jnp.clip(var5, k1_range[0], k1_range[1])
    var6_clipped = jnp.clip(var6, k2_range[0], k2_range[1])
    # Check if clipped and unclipped point are equal
    return jnp.logical_and(
        jnp.logical_and(
            jnp.logical_and(
                jnp.logical_and(
                    jnp.logical_and(var1 == var1_clipped, var2 == var2_clipped),
                    var3 == var3_clipped,
                ),
                var4 == var4_clipped,
            ),
            var5 == var5_clipped,
        ),
        var6 == var6_clipped,
    ).astype(jnp.float64)


def test_6D_metric(key):
    """
    Returns a filled bank.
    """
    naive_vol_key, frac_ib_key, n_templates_key, fill_key = random.split(key, 4)

    naive_vol = (
        (Mt_range[1] - Mt_range[0])
        * (eta_range[1] - eta_range[0])
        * (chi1_range[1] - chi1_range[0])
        * (chi2_range[1] - chi2_range[0])
        * (k1_range[1] - k1_range[0])
        * (k2_range[1] - k2_range[0])
    )
    naive_vol_err = 0.0

    # testing
    fs = jnp.linspace(f_l, f_u, 10000)
    Sn_aLIGO = get_Sn_aLIGO()
    mm = 0.95
    m_star = 1 - mm
    eta = 0.95
    bank = Bank(
        kappa6D.Amp,
        kappa6D.Psi,
        fs,
        Sn_aLIGO,
        propose,
        naive_vol,
        m_star,
        eta,
        accept,
        "6D",
    )

    # def my_get_density(thetas):
        # return jnp.sqrt(jnp.linalg.det(get_g(thetas, kappa6D.Amp, kappa6D.Psi, fs, Sn_aLIGO)))
    
    def my_get_density(g):
        return jnp.sqrt(jnp.linalg.det(g))
    
    N = 40
    thetas = propose(naive_vol_key, N)

    def map_test(f, xs):
        return jnp.stack([f(x) for x in xs])

    # g_loop = map_test(bank.get_g, thetas)
    # g_map = jax.lax.map(bank.get_g, thetas)
    # print(g_map, g_loop)
    # print(jnp.allclose(g_map, g_loop, atol=0.0, rtol=1e-6))

    # print(jax.make_jaxpr(my_get_density)(thetas[0]))
    gs = map_test(bank.get_g, thetas)
    densities_loop = map_test(my_get_density, gs)
    densities_map = jax.lax.map(my_get_density, gs)
    print(densities_loop)
    print(densities_map)
    print(densities_loop)
    print(densities_map)
    print(jnp.allclose(densities_map, densities_loop, atol=0.0, rtol=1e-6))
    # print(densities, densities_loop)
    # print(densities)
    # print(thetas)
    # # mask = jnp.argwhere(jnp.isnan(densities))
    # mask = jnp.nanargmin(densities)
    # # print(mask)
    # print(thetas[mask])
    # # # print(thetas)
    # print(densities[mask])
    # g = bank.get_g(thetas[mask])
    # print(g)
    # print(jnp.linalg.eig(g))
    # print(jnp.sqrt(jnp.linalg.det(g)))
    # print(densities[mask])
    # print(thetas[mask])
    # print(thetas)

    return False


@click.command()
@click.option("--seed", type=int, help="PRNG seed")
def main(seed):
    key = random.PRNGKey(seed)
    bank_key, eff_key = random.split(key)
    test_6D_metric(key)
    # get_bank(bank_key)


if __name__ == "__main__":
    main()


# def get_bank(key):
#     """
#     Returns a filled bank.
#     """
#     naive_vol_key, frac_ib_key, n_templates_key, fill_key = random.split(key, 4)

#     # Estimate naive var volume to within 10%
#     naive_vol = (
#         (th0_range[1] - th0_range[0])
#         * (th3_range[1] - th3_range[0])
#         * (chi1_range[1] - chi1_range[0])
#         * (chi2_range[1] - chi2_range[0])
#         * (k1_range[1] - k1_range[0])
#         * (k2_range[1] - k2_range[0])
#     )
#     naive_vol_err = 0.0
#     print(
#         f"Naive parameter space volume (just using a box): {naive_vol:.6f} +/- {naive_vol_err:.6f}"
#     )
#     assert naive_vol_err / naive_vol < 0.1

#     # Configure bank
#     Sn_aLIGO = get_Sn_aLIGO()
#     fs = jnp.linspace(f_l, f_u, 5000)
#     mm = 0.97
#     m_star = 1 - mm
#     eta = 0.95
#     bank = Bank(
#         kappa6D.Amp,
#         kappa6D.Psi,
#         fs,
#         Sn_aLIGO,
#         var_sampler,
#         naive_vol,
#         m_star,
#         eta,
#         accept,
#         "6D",
#     )

#     max_density_theta = jnp.array(
#         [
#             th0_range[0],
#             th3_range[0],
#             chi1_range[1],
#             chi2_range[0],
#             k1_range[1],
#             k2_range[1],
#         ]
#     )  # I guessed this point, need to verify
#     bank.density_max = bank.get_density(max_density_theta)

#     bank.compute_template_frac_in_bounds(frac_ib_key, 1000, 20)
#     print(
#         f"{bank.frac_in_bounds * 100:.3f} +/- {bank.frac_in_bounds_err * 100:.3f} % "
#         "of the average template ellipse's volume is in bounds"
#     )
#     bank.compute_n_templates(n_templates_key, 10000)
#     print(f"{bank.n_templates} +/- {bank.n_templates_err:.2f} templates required")

#     return bank
