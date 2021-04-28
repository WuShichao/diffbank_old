from math import pi

from diffbank.bank import Bank
from diffbank.constants import C, G, MSUN
from diffbank.utils import ms_to_Mc_eta
from diffbank.utils import gen_templates_rejection
from jax import random
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


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


@jax.jit
def accept(pt):
    """
    Acceptance probability for tau rejection sampling.
    """
    tau1, tau2 = pt
    # I don't know of a better way to handle boolean operators in jax...
    return jnp.where(
        tau1 < tau1_range[0],
        jnp.array(0.0),
        jnp.where(
            tau1 > tau1_range[1],
            jnp.array(0.0),
            jnp.where(
                tau2 < tau2_interp_low(tau1),
                jnp.array(0.0),
                jnp.where(
                    tau2 > tau2_interp_high(tau1),
                    jnp.array(0.0),
                    jnp.array(1.0),
                ),
            ),
        ),
    )


def tau_sampler(key, n):
    """
    Rejection sampler for tau.
    """
    return gen_templates_rejection(key, jnp.array(1.0), n, accept, propose)


def Sn_LIGO(f):
    return jnp.where(
        f > f_s, 1 / 5 * S_0 * ((f / f_0) ** (-4) + 2 * (1 + (f / f_0) ** 2)), jnp.inf
    )


def amp(f, _):
    return f ** (-7 / 6)


def Psi(f, theta):
    tau1, tau2 = theta
    return (
        6 / 5 * pi * f_0 * (f / f_0) ** (-5 / 3) * tau1
        + 2 * pi * f_0 * (f / f_0) ** (-1) * tau2
    )


def get_bank(key):
    """
    Returns a filled bank.
    """
    vol_key, n_templates_key, fill_key = random.split(key, 3)
    # Estimate tau volume to within 10%
    proposals = propose(vol_key, 10000)
    proposal_vol = (tau1_range[1] - tau1_range[0]) * (tau2_range[1] - tau2_range[0])
    naive_vol = accept(proposals.T).mean() * proposal_vol
    naive_vol_err = accept(proposals.T).std() * proposal_vol / jnp.sqrt(len(proposals))
    print(f"Bank volume: {naive_vol} +/- {naive_vol_err}")
    assert naive_vol_err / naive_vol < 0.1

    # Configure bank
    fs = jnp.linspace(f_s, 10000, 3000)
    mm = 0.95
    m_star = 1 - mm
    eta = 0.95
    bank = Bank(amp, Psi, fs, Sn_LIGO, tau_sampler, naive_vol, m_star, eta, "owen")

    # Metric is constant, so can just compute density at any point
    bank.density_max = bank.get_density(tau_sampler(vol_key, 1)[0])

    bank.compute_n_templates(n_templates_key, 50)
    print(f"Filling bank with {bank.n_templates} templates")
    bank.fill_bank(fill_key)
    print(f"Done: {bank}")

    return bank


def plot_bank(bank):
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

    plt.savefig("figures/owens-bank.pdf")
    plt.close()


def plot_effectualnesses(key, bank, n=1000):
    """
    Histogram effectualnesses for n signal injections sampled using parameter
    space metric density.
    """
    # Sample signal injections from template bank
    points = bank.gen_templates_rejection(key, n)
    bank.compute_effectualnesses(points)  # slow

    eff_frac = jnp.mean(bank.effectualnesses > bank.minimum_match)
    eff_frac_err = jnp.std(bank.effectualnesses > bank.minimum_match) / jnp.sqrt(
        len(bank.effectualnesses)
    )

    plt.figure()

    plt.hist(bank.effectualnesses, bins=50)
    plt.axvline(bank.minimum_match, color="r")

    plt.title(
        r"$^{%.2f}\mathcal{R}_{%i}(%.2f)$, effectualness $\in [%.3f, %.3f]$"
        % (
            bank.eta,
            bank.dim,
            bank.m_star,
            eff_frac - 3 * eff_frac_err,
            eff_frac + 3 * eff_frac_err,
        )
    )
    plt.xlabel("Effectualness")
    plt.ylabel("Frequency")

    plt.savefig("figures/owens-effectualnesses.pdf")
    plt.close()


if __name__ == "__main__":
    key = random.PRNGKey(528)

    bank = get_bank(key)
    _, key = random.split(key)

    plot_bank(bank)
    plot_effectualnesses(key, bank)
