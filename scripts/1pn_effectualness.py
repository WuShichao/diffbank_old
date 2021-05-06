from math import pi

import click
import matplotlib.pyplot as plt

from diffbank.bank import Bank
from diffbank.constants import C, G, MSUN
from diffbank.utils import ms_to_Mc_eta
from diffbank.utils import gen_templates_rejection
from jax import random
import jax
import jax.numpy as jnp


"""
Tests consistency of random template bank for the 1PN chirp time waveform and
noise PSD model from Owen 1995 (https://arxiv.org/abs/gr-qc/9511032).
"""

f_0 = 200.0  # Hz
f_s = 40.0  # Hz
S_0 = 1e-46  # arbitrary


def phys_to_th(phys: jnp.ndarray) -> jnp.ndarray:
    """
    Convert (M_chirp [kg], eta) -> (tau1 [s], tau2 [s]).
    """
    _M_chirp, eta = phys
    M_chirp = _M_chirp / (C ** 3 / G)  # kg -> s
    M = M_chirp / eta ** (3 / 5)  # total mass
    tau1 = 5 / 256 / (eta * M ** (5 / 3) * (pi * f_0) ** (8 / 3))
    tau2 = 5 / 192 / (eta * M) * (743 / 336 + 11 / 4 * eta) / (pi * f_0) ** 2
    return jnp.stack([tau1, tau2])


def get_th_boundary_interps(m_min, m_max, n=200):
    """
    Given a range of BH masses, returns corresponding range of tau1 and
    interpolators for the minimum and maximum corresponding values of tau2.
    """
    # Lower boundary
    ms = jnp.linspace(m_min, m_max, n)
    M_chirps = ms_to_Mc_eta(jnp.stack([ms, ms]))[0]
    etas = jnp.full_like(M_chirps, 0.25)

    tau1_lows, tau2_lows = phys_to_th(jnp.stack([M_chirps, etas]))[:2, ::-1]
    tau2_interp_low = lambda tau1: jnp.interp(
        tau1, tau1_lows, tau2_lows, left=jnp.nan, right=jnp.nan
    )

    # Upper boundary
    M_chirps, etas = ms_to_Mc_eta(jnp.stack([ms, jnp.full_like(ms, m_max)]))
    tau1_highs_1, tau2_highs_1 = phys_to_th(jnp.stack([M_chirps, etas]))[:2, ::-1]

    M_chirps, etas = ms_to_Mc_eta(jnp.stack([ms, jnp.full_like(ms, m_min)]))
    tau1_highs_2, tau2_highs_2 = phys_to_th(jnp.stack([M_chirps, etas]))[:2, ::-1]

    tau1_highs = jnp.concatenate([tau1_highs_1, tau1_highs_2])
    tau2_highs = jnp.concatenate([tau2_highs_1, tau2_highs_2])
    tau2_interp_high = lambda tau1: jnp.interp(
        tau1, tau1_highs, tau2_highs, left=jnp.nan, right=jnp.nan
    )

    tau1_min = min(tau1_lows.min(), tau1_highs_1.min())
    tau1_max = max(tau1_lows.max(), tau1_highs_2.max())

    return (tau1_min, tau1_max), tau2_interp_low, tau2_interp_high


m_range = (1 * MSUN, 100 * MSUN)
tau1_range, tau2_interp_low, tau2_interp_high = get_th_boundary_interps(*m_range)
tau2_range = (tau2_interp_low(tau1_range[0]), tau2_interp_high(tau1_range[1]))


def propose(key, n):
    """
    Proposal distribution for tau rejection sampling.
    """
    return random.uniform(
        key,
        shape=(
            n,
            2,
        ),
        minval=jnp.stack((tau1_range[0], tau2_range[0])),
        maxval=jnp.stack((tau1_range[1], tau2_range[1])),
    )


def accept(taus: jnp.ndarray) -> jnp.ndarray:
    """
    Returns 1 if taus is in bounds, 0 otherwise.
    """
    tau1, tau2 = taus[..., 0], taus[..., 1]
    tau1_clipped = jnp.clip(tau1, tau1_range[0], tau1_range[1])
    tau2_clipped = jnp.clip(tau2, tau2_interp_low(tau1), tau2_interp_high(tau1))
    # Check if clipped and unclipped point are equal
    return jnp.logical_and(tau1 == tau1_clipped, tau2 == tau2_clipped).astype(
        jnp.float64
    )


def tau_sampler(key, n):
    """
    Rejection sampler for tau.
    """
    return gen_templates_rejection(key, jnp.array(1.0), n, accept, propose)


def Sn_LIGO(f):
    """
    LIGO noise curve.
    """
    return jnp.where(
        f > f_s, 1 / 5 * S_0 * ((f / f_0) ** (-4) + 2 * (1 + (f / f_0) ** 2)), jnp.inf
    )


def amp(f, _):
    return f ** (-7 / 6)


def Psi(f, theta):
    """
    1PN phase.
    """
    tau1, tau2 = theta
    return (
        6 / 5 * pi * f_0 * (f / f_0) ** (-5 / 3) * tau1
        + 2 * pi * f_0 * (f / f_0) ** (-1) * tau2
    )


def get_bank(key):
    """
    Returns a filled bank.
    """
    naive_vol_key, frac_ib_key, n_templates_key, fill_key = random.split(key, 4)

    # Estimate naive tau volume to within 10%
    proposals = propose(naive_vol_key, 1000000)
    proposal_vol = (tau1_range[1] - tau1_range[0]) * (tau2_range[1] - tau2_range[0])
    in_bounds = accept(proposals)
    naive_vol = in_bounds.mean() * proposal_vol
    naive_vol_err = in_bounds.std() * proposal_vol / jnp.sqrt(len(proposals))
    print(f"Naive parameter space volume: {naive_vol:.5f} +/- {naive_vol_err:.5f}")
    assert naive_vol_err / naive_vol < 0.1

    # Configure bank
    fs = jnp.linspace(f_s, 10000, 3000)
    mm = 0.95
    m_star = 1 - mm
    eta = 0.95
    bank = Bank(
        amp, Psi, fs, Sn_LIGO, tau_sampler, naive_vol, m_star, eta, accept, "owen"
    )

    # Metric is constant, so can just compute density at any point
    bank.density_max = bank.get_density(tau_sampler(naive_vol_key, 1)[0])
    bank.compute_template_frac_in_bounds(frac_ib_key, 20000, 20)
    print(
        f"{bank.frac_in_bounds * 100:.2f} +/- {bank.frac_in_bounds_err * 100:.2f} % "
        "of the average template ellipse's volume is in bounds"
    )
    bank.compute_n_templates(n_templates_key, 50)
    print(f"{bank.n_templates} +/- {bank.n_templates_err} templates required")
    print("Filling the bank")
    bank.fill_bank(fill_key)
    print(f"Done: {bank}")

    return bank


def plot_bank(bank, seed):
    """
    Plot template positions and tau space boundaries.
    """
    plt.figure()

    # Boundaries
    tau1s = jnp.linspace(tau1_range[0], tau1_range[1], 300)
    plt.plot(tau1s, tau2_interp_low(tau1s), "C0", linewidth=0.5)
    plt.plot(tau1s, tau2_interp_high(tau1s), "C0", linewidth=0.5)

    # Proposal distribution boundaries
    plt.plot(
        [tau1_range[0], tau1_range[1], tau1_range[1], tau1_range[0], tau1_range[0]],
        [tau2_range[0], tau2_range[0], tau2_range[1], tau2_range[1], tau2_range[0]],
        "C1",
        linewidth=0.5,
    )

    # Template centers
    plt.scatter(*bank.templates.T, c="C3", s=0.5, linewidth=0)

    plt.xlabel(r"$\tau_1$ [s]")
    plt.ylabel(r"$\tau_2$ [s]")
    plt.title(
        r"$m \in [%i, %i] \, \mathrm{M}_\odot$" % (m_range[0] / MSUN, m_range[1] / MSUN)
    )

    plt.savefig(f"figures/owens-bank_seed={seed}.pdf")
    plt.close()


def plot_effectualnesses(key, bank, seed, n=250):
    """
    Histogram effectualnesses for n signal injections sampled using parameter
    space metric density.
    """
    # Sample signal injections from template bank
    bank.compute_effectualnesses(key, n)  # slow

    plt.figure(figsize=(8, 3.5))

    plt.subplot(1, 2, 1)
    plt.hist(bank.effectualnesses)  # , bins=50)
    plt.axvline(bank.minimum_match, color="r")
    plt.xlabel("Effectualness")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    idxs = jnp.argsort(bank.effectualnesses)[::-1]
    plt.scatter(
        *bank.effectualness_points[idxs].T,
        c=bank.effectualnesses[idxs],
        s=5,
        linewidth=0,
        vmin=0.9,
        vmax=1.0,
        cmap="bwr",
    )
    plt.colorbar(label="Effectualness")
    plt.xlabel(r"$\tau_1$ [s]")
    plt.ylabel(r"$\tau_2$ [s]")

    plt.suptitle(
        r"$^{%.2f}\mathcal{R}_{%i}(%.2f)$, $\hat{\eta} \in [%.3f, %.3f]$"
        % (
            bank.eta,
            bank.dim,
            bank.m_star,
            bank.eta_est - 1 * bank.eta_est_err,
            bank.eta_est + 1 * bank.eta_est_err,
        )
    )
    plt.tight_layout()

    plt.savefig(f"figures/owens-effs_seed={seed}.pdf")
    plt.close()


@click.command()
@click.option("--seed", type=int, help="PRNG seed")
def main(seed):
    key = random.PRNGKey(seed)
    bank_key, eff_key = random.split(key)

    bank = get_bank(bank_key)

    plot_bank(bank, seed)
    plot_effectualnesses(eff_key, bank, seed)


if __name__ == "__main__":
    main()
