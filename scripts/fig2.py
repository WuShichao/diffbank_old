from itertools import product
from typing import Callable

import click
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from diffbank.constants import MSUN
from diffbank.metric import get_density, get_g
from diffbank.utils import ms_to_Mc_eta
from diffbank.waveforms.twoPN_chirptimes import (
    Psi,
    Amp,
    get_th_boundary_interps,
    phys_to_th,
    analytic_metric,
)
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from diffjeom import get_ricci_scalar

import matplotlib.pyplot as plt
import matplotlib as mpl

# Set some plotting parameters
fs = 16  # font size

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams.update(
    {
        "legend.fontsize": fs,
        "axes.labelsize": fs,
        "axes.titlesize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
    }
)
plt.rcParams["figure.figsize"] = (7, 5)  # figure size

mpl.rcParams["xtick.top"] = True
mpl.rcParams["ytick.right"] = True
# from pycbc import psd

##### Frequency settings
f_u = 512.0  # Hz
f_l = 10.0  # Hz
# f0 = 20.0
N_fbins = 1000
#####

# def get_aLIGOZeroDetHighPower_interp(
#     f_0,
#     f_max=1024,
# ) -> Callable[[jnp.ndarray], jnp.ndarray]:
#     """
#     Get interpolator for noise curve.
#     """
#     delta_f = 1.0 / 4
#     flen = int(f_max / delta_f)

#     # One can either call the psd generator by name
#     p = psd.aLIGOZeroDetHighPower(flen, delta_f, f_0)
#     xp = p.sample_frequencies
#     yp = p
#     yp[yp == 0.0] = jnp.inf
#     xp, yp = jnp.array(xp), jnp.array(yp)
#     return lambda f: jnp.interp(f, xp, yp, left=jnp.inf, right=jnp.inf)


def get_Sn_O3a() -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Get interpolator for noise curve.
    """
    xp, yp = np.loadtxt("O3a_Livingston_ASD.txt", unpack=True)
    PSD = yp ** 2
    return lambda f: jnp.interp(f, xp, PSD, left=jnp.inf, right=jnp.inf)


def get_Sn_O3a_np():
    """
    Get interpolator for noise curve.
    """
    xp, yp = np.loadtxt("O3a_Livingston_ASD.txt", unpack=True)
    PSD = yp ** 2
    return lambda f: np.interp(f, xp, PSD, left=jnp.inf, right=jnp.inf)


@click.command()
@click.argument(
    "fig_path",
    type=str,
    required=True,
    default="figures/Fig2_density_comparison.pdf",
)
@click.option("--n_m1s", type=int, default=100)
@click.option("--n_m2s", type=int, default=90)
# @click.option("--n_th0s", type=int, default=20)
# @click.option("--n_th3s", type=int, default=19)
def make_fig2(fig_path, n_m1s, n_m2s):
    # Table 1: global analysis configuration
    # Low-frequency cutoff
    fs = jnp.linspace(f_l, f_u, N_fbins)
    # Component BH masses
    m_min = jnp.array(1.0) * MSUN
    m_max = jnp.array(20.0) * MSUN
    # Total mass
    M_max = m_min + m_max

    # (th0, th3) boundaries
    (th0_min, th0_max), bound_itp_low, bound_itp_high = get_th_boundary_interps(
        m_min, m_max, f_l
    )

    # Sn = get_aLIGOZeroDetHighPower_interp(f_0)
    Sn = get_Sn_O3a()
    Sn_np = get_Sn_O3a_np()
    density_fun = lambda theta: get_density(theta, Amp, Psi, fs, Sn)

    # Set parameter grid
    m1s = jnp.geomspace(m_min, m_max, n_m1s)
    m2s = jnp.geomspace(m_min, m_max, n_m2s)
    m1s, m2s = jnp.array(list(product(m1s, m2s))).T
    m1s, m2s = m1s[m1s >= m2s], m2s[m1s >= m2s]  # remove redundant systems
    m1s, m2s = m1s[m1s + m2s <= M_max], m2s[m1s + m2s <= M_max]
    M_chirps, etas = ms_to_Mc_eta(jnp.stack([m1s, m2s]))

    # Plotting configuration
    vmin = -15.0
    vmax = -12.5
    # levels = jnp.linspace(vmin, vmax, 60)
    cbar_ticks = jnp.arange(vmin, vmax + 0.05, 0.05)

    plt.figure(figsize=(7, 5))
    thetas = phys_to_th(jnp.stack([M_chirps, etas]), f_l).T
    densities = jax.lax.map(density_fun, thetas)
    densities_analytic = np.zeros(thetas.shape[0])

    for i in tqdm(range(thetas.shape[0])):
        densities_analytic[i] = np.sqrt(
            np.linalg.det(analytic_metric(fs, thetas[i], Sn_np))
        )
    print(np.log10(np.abs((densities - densities_analytic) / densities_analytic)))
    # quit()

    # Plot!
    # th0s = jnp.linspace(thetas[:, 0].min(), thetas[:, 0].max(), n_th0s)
    # th3s = jnp.linspace(thetas[:, 1].min(), thetas[:, 1].max(), n_th3s)

    # cs = plt.contourf(
    #     th0s,
    #     th3s,
    #     jnp.clip(
    #         griddata(
    #             thetas[:, :2],
    #             jnp.log10(densities),
    #             jnp.stack(jnp.meshgrid(th0s, th3s)).reshape([2, -1]).T,
    #         ).reshape([len(th3s), len(th0s)]),
    #         vmin,
    #         vmax,
    #     ),
    #     levels=levels,
    #     cmap="PuOr",
    # )
    # plt.colorbar(cs, label=r"$\log_{10} \sqrt{|g|}$", ticks=cbar_ticks)
    col = np.log10(np.abs((densities - densities_analytic) / densities_analytic))
    cs = plt.scatter(thetas[:, 0], thetas[:, 1], c=col, cmap="PuOr")
    plt.colorbar(
        cs,
        label=r"$\log_{10} \left(\frac{\sqrt{|g^ {ad}|}-\sqrt{|g|}}{\sqrt{|g|}}\right)$",
    ).ax.tick_params(axis="y", direction="in")
    # plt.colorbar().ax.tick_params(axis='y', direction='in')
    # , ticks=cbar_ticks

    # Mask outside boundaries
    th0_grid = jnp.linspace(th0_min, th0_max, 200)
    plt.fill_between(
        th0_grid,
        bound_itp_low(th0_grid),
        jnp.full_like(th0_grid, -1e3),
        where=jnp.full_like(th0_grid, True),
        color="w",
    )
    plt.fill_between(
        th0_grid,
        bound_itp_high(th0_grid),
        jnp.full_like(th0_grid, 1e3),
        where=jnp.full_like(th0_grid, True),
        color="w",
    )

    plt.tick_params(axis="y", which="major", direction="in", length=6, pad=6)
    plt.tick_params(axis="y", which="minor", direction="in", length=3)
    plt.tick_params(axis="x", which="major", direction="in", length=6, pad=6)
    plt.tick_params(axis="x", which="minor", direction="in", length=3)

    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_3$")
    plt.xlim(0.0 * 1e4, 10.0 * 1e4)
    plt.ylim(0.8 * 1e2, 8 * 1e2)
    plt.savefig(fig_path, bbox_inches="tight")


@click.command()
@click.argument(
    "fig_path",
    type=str,
    required=True,
    default="figures/Fig3_ricci_scalar.pdf",
)
@click.option("--n_m1s", type=int, default=200)
@click.option("--n_m2s", type=int, default=100)
@click.option("--n_th0s", type=int, default=200)
@click.option("--n_th3s", type=int, default=100)
def make_fig3(fig_path, n_m1s, n_m2s, n_th0s, n_th3s):
    # Table 1: global analysis configuration
    # Low-frequency cutoff
    fs = jnp.linspace(f_l, f_u, N_fbins)
    # Component BH masses
    m_min = jnp.array(1.0) * MSUN
    m_max = jnp.array(20.0) * MSUN
    # Total mass
    M_max = m_min + m_max

    # (th0, th3) boundaries
    (th0_min, th0_max), bound_itp_low, bound_itp_high = get_th_boundary_interps(
        m_min, m_max, f_l
    )

    # Sn = get_aLIGOZeroDetHighPower_interp(f_0)
    Sn = get_Sn_O3a()
    # density_fun = lambda theta: get_density(theta, Amp, Psi, fs, Sn)
    g_fun = lambda theta: get_g(theta, Amp, Psi, fs, Sn)

    # Set parameter grid
    m1s = jnp.geomspace(m_min, m_max, n_m1s)
    m2s = jnp.geomspace(m_min, m_max, n_m2s)
    m1s, m2s = jnp.array(list(product(m1s, m2s))).T
    m1s, m2s = m1s[m1s >= m2s], m2s[m1s >= m2s]  # remove redundant systems
    m1s, m2s = m1s[m1s + m2s <= M_max], m2s[m1s + m2s <= M_max]
    M_chirps, etas = ms_to_Mc_eta(jnp.stack([m1s, m2s]))

    # Plotting configuration
    vmin = -10.0
    vmax = -7.5
    levels = jnp.linspace(vmin, vmax, 60)
    cbar_ticks = jnp.arange(vmin, vmax + 0.05, 0.5)

    plt.figure(figsize=(7, 5))
    thetas = phys_to_th(jnp.stack([M_chirps, etas]), f_l).T
    Rss = jax.vmap(lambda x: get_ricci_scalar(x, g_fun))(thetas)
    # print(Rss)
    # densities = jax.lax.map(density_fun, thetas)

    # Plot!
    th0s = jnp.linspace(thetas[:, 0].min(), thetas[:, 0].max(), n_th0s)
    th3s = jnp.linspace(thetas[:, 1].min(), thetas[:, 1].max(), n_th3s)

    cs = plt.contourf(
        th0s,
        th3s,
        jnp.clip(
            griddata(
                thetas[:, :2],
                jnp.log10(jnp.abs(Rss)),
                jnp.stack(jnp.meshgrid(th0s, th3s)).reshape([2, -1]).T,
            ).reshape([len(th3s), len(th0s)]),
            vmin,
            vmax,
        ),
        levels=levels,
        cmap="viridis",
    )
    plt.colorbar(cs, label=r"$\log_{10}(|R|)$", ticks=cbar_ticks)
    # col = np.log10(np.abs(Rss))
    # cs = plt.scatter(thetas[:, 0], thetas[:, 1], c=col, cmap="PuOr")
    # plt.colorbar(
    #     cs,
    #     label=r"Ricci Scalar",
    # ).ax.tick_params(axis="y", direction="in")

    # Mask outside boundaries
    th0_grid = jnp.linspace(th0_min, th0_max, 200)
    plt.fill_between(
        th0_grid,
        bound_itp_low(th0_grid),
        jnp.full_like(th0_grid, -1e3),
        where=jnp.full_like(th0_grid, True),
        color="w",
    )
    plt.fill_between(
        th0_grid,
        bound_itp_high(th0_grid),
        jnp.full_like(th0_grid, 1e3),
        where=jnp.full_like(th0_grid, True),
        color="w",
    )

    plt.tick_params(axis="y", which="major", direction="in", length=6, pad=6)
    plt.tick_params(axis="y", which="minor", direction="in", length=3)
    plt.tick_params(axis="x", which="major", direction="in", length=6, pad=6)
    plt.tick_params(axis="x", which="minor", direction="in", length=3)

    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_3$")
    plt.xlim(0.0 * 1e4, 10.0 * 1e4)
    plt.ylim(0.8 * 1e2, 8 * 1e2)
    plt.savefig(fig_path, bbox_inches="tight")


def metric_speed():
    fs = jnp.linspace(f_l, f_u, N_fbins)
    # Component BH masses
    m_min = jnp.array(1.0) * MSUN
    m_max = jnp.array(20.0) * MSUN
    # Total mass
    M_max = m_min + m_max

    # (th0, th3) boundaries
    (th0_min, th0_max), bound_itp_low, bound_itp_high = get_th_boundary_interps(
        m_min, m_max, f_l
    )

    # Sn = get_aLIGOZeroDetHighPower_interp(f_0)
    Sn = get_Sn_O3a()
    Sn_np = get_Sn_O3a_np()
    density_fun = lambda theta: get_density(theta, Amp, Psi, fs, Sn)

    # Set parameter grid
    n_m1s = 100
    n_m2s = 90
    m1s = jnp.geomspace(m_min, m_max, n_m1s)
    m2s = jnp.geomspace(m_min, m_max, n_m2s)
    m1s, m2s = jnp.array(list(product(m1s, m2s))).T
    m1s, m2s = m1s[m1s >= m2s], m2s[m1s >= m2s]  # remove redundant systems
    m1s, m2s = m1s[m1s + m2s <= M_max], m2s[m1s + m2s <= M_max]
    M_chirps, etas = ms_to_Mc_eta(jnp.stack([m1s, m2s]))

    plt.figure(figsize=(7, 5))
    import time

    thetas = phys_to_th(jnp.stack([M_chirps, etas]), f_l).T
    size_theta = thetas.shape[0]
    t0 = time.time()
    densities = jax.lax.map(density_fun, thetas)
    t1 = time.time()
    print(size_theta, t1 - t0)
    print(
        "Time taken to run %.0f metric calculations: %.3f s"
        % (
            size_theta,
            t1 - t0,
        )
    )
    return None


def analytic_g():
    fs = jnp.linspace(f_l, f_u, N_fbins)
    # Component BH masses
    m_min = jnp.array(1.0) * MSUN
    m_max = jnp.array(20.0) * MSUN
    # Total mass
    M_max = m_min + m_max

    Sn = get_Sn_O3a()

    # Set parameter grid
    m1s = jnp.geomspace(m_min, m_max, 10)
    m2s = jnp.geomspace(m_min, m_max, 10)
    m1s, m2s = jnp.array(list(product(m1s, m2s))).T
    m1s, m2s = m1s[m1s >= m2s], m2s[m1s >= m2s]  # remove redundant systems
    m1s, m2s = m1s[m1s + m2s <= M_max], m2s[m1s + m2s <= M_max]
    M_chirps, etas = ms_to_Mc_eta(jnp.stack([m1s, m2s]))

    thetas = phys_to_th(jnp.stack([M_chirps, etas]), f_l).T
    print(np.sqrt(np.linalg.det(analytic_metric(fs, thetas[0], Sn))))
    density_fun = lambda theta: get_density(theta, Amp, Psi, fs, Sn)
    print(density_fun(thetas[0]))


# @click.command()
# @click.option("--seed", type=int, help="PRNG seed")
def main():
    # key = random.PRNGKey(seed)
    # make_fig2()
    # make_fig3()
    # analytic_g()
    metric_speed()


if __name__ == "__main__":
    main()
