from itertools import product
from typing import Callable, Union

import click
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from diffbank.constants import MSUN
from diffbank.metric import get_density
from diffbank.utils import ms_to_Mc_eta
from diffbank.waveforms.taylorf2reducedspin import (
    Psi,
    amp,
    get_th_boundary_interps,
    phys_to_th,
)
import jax
import jax.numpy as jnp
from pycbc import psd

"""
Reproduces metric calculation (fig. 2) from https://arxiv.org/abs/1210.6666.

Run with:

    >>> python taylorf2reducedspin_density_plot.py

"""


def get_aLIGOZeroDetHighPower_interp(
    f_0: Union[float, jnp.ndarray],
    f_max: Union[float, jnp.ndarray] = 1024,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Get interpolator for noise curve.
    """
    delta_f = 1.0 / 4
    flen = int(f_max / delta_f)

    # One can either call the psd generator by name
    p = psd.aLIGOZeroDetHighPower(flen, delta_f, f_0)
    xp = p.sample_frequencies
    yp = p
    yp[yp == 0.0] = jnp.inf
    xp, yp = jnp.array(xp), jnp.array(yp)
    return lambda f: jnp.interp(f, xp, yp, left=jnp.inf, right=jnp.inf)


@click.command()
@click.argument(
    "fig_path",
    type=str,
    required=True,
    default="figures/taylorf2reducedspin_density.pdf",
)
@click.option("--n_m1s", type=int, default=100)
@click.option("--n_m2s", type=int, default=90)
@click.option("--n_th0s", type=int, default=200)
@click.option("--n_th3s", type=int, default=190)
def make_density_plot(fig_path: str, n_m1s: int, n_m2s: int, n_th0s: int, n_th3s: int):
    # Table 1: global analysis configuration
    # Low-frequency cutoff
    f_0 = jnp.array(20.0)
    # Component BH masses
    m_min = jnp.array(1.0) * MSUN
    m_max = jnp.array(20.0) * MSUN
    # Total mass
    M_max = m_min + m_max

    # (th0, th3) boundaries
    (th0_min, th0_max), bound_itp_low, bound_itp_high = get_th_boundary_interps(
        m_min, m_max, f_0
    )

    f_max = 2200.0  # encompasses highest ISCO frequency
    fs = jnp.geomspace(f_0, f_max, 300)
    Sn = get_aLIGOZeroDetHighPower_interp(f_0)
    density_fun = lambda theta: get_density(theta, amp, Psi, fs, Sn)

    # Set parameter grid
    m1s = jnp.geomspace(m_min, m_max, n_m1s)
    m2s = jnp.geomspace(m_min, m_max, n_m2s)
    m1s, m2s = jnp.array(list(product(m1s, m2s))).T
    m1s, m2s = m1s[m1s >= m2s], m2s[m1s >= m2s]  # remove redundant systems
    m1s, m2s = m1s[m1s + m2s <= M_max], m2s[m1s + m2s <= M_max]
    M_chirps, etas = ms_to_Mc_eta(jnp.stack([m1s, m2s]))
    chis = [-0.3, 0.0, 0.3]

    # Plotting configuration
    vmin = -7
    vmax = -5.5
    levels = jnp.linspace(vmin, vmax, 60)
    cbar_ticks = jnp.arange(vmin, vmax + 0.5, 0.5)
    th0_scale = 1e4
    th3_scale = 1e2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i, chi in enumerate(chis):
        print(f"[{i + 1} / {len(axes)}] chi = {chi}")
        ax = axes[i]

        chis = jnp.full_like(etas, chi)
        thetas = phys_to_th(jnp.stack([M_chirps, etas, chis])).T
        densities = jax.lax.map(density_fun, thetas)

        # Plot!
        th0s = jnp.linspace(thetas[:, 0].min(), thetas[:, 0].max(), n_th0s)
        th3s = jnp.linspace(thetas[:, 1].min(), thetas[:, 1].max(), n_th3s)

        cs = ax.contourf(
            th0s / th0_scale,
            th3s / th3_scale,
            jnp.clip(
                griddata(
                    thetas[:, :2],
                    jnp.log10(densities),
                    jnp.stack(jnp.meshgrid(th0s, th3s)).reshape([2, -1]).T,
                ).reshape([len(th3s), len(th0s)]),
                vmin,
                vmax,
            ),
            levels=levels,
            cmap="jet",
        )
        if i == len(axes) - 1:
            fig.colorbar(cs, ax=ax, label=r"$\log_{10} \sqrt{|g|}$", ticks=cbar_ticks)

        # Mask outside boundaries
        th0_grid = jnp.linspace(th0_min, th0_max, 200)
        ax.fill_between(
            th0_grid / th0_scale,
            bound_itp_low(th0_grid) / th3_scale,
            jnp.full_like(th0_grid, -1e3),
            where=jnp.full_like(th0_grid, True),
            color="w",
        )
        ax.fill_between(
            th0_grid / th0_scale,
            bound_itp_high(th0_grid) / th3_scale,
            jnp.full_like(th0_grid, 1e3),
            where=jnp.full_like(th0_grid, True),
            color="w",
        )

        ax.set_title(r"$\chi = %g$" % chi)
        ax.set_xlabel(r"$\theta_0 / 10^{%i}$" % jnp.log10(th0_scale))
        ax.set_ylabel(r"$\theta_3 / 10^{%i}$" % jnp.log10(th3_scale))
        ax.set_xlim(0.0, 3.2)
        ax.set_ylim(0.8, 5)

    fig.tight_layout()
    fig.savefig(fig_path)


if __name__ == "__main__":
    make_density_plot()
