from math import log
import os
import re
from typing import Dict, Tuple

from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt

from diffbank.bank import Bank
from diffbank.noise import Sn_LIGOI
from diffbank.utils import get_m1_m2_sampler
from diffbank.waveforms.threePN_simple import Psi, amp

plt.style.use("../plot_style.mplstyle")


"""
Plots scaling properties of the 3.5PN-2D bank.

The requisite banks and data files are generated by running the following
scripts:
    - job-threePN-est-p.sh: estimates the covering probability `p`.
    - job-threePN-scaling.sh: generates banks with a range of different `mm`
      and `eta` values.

Generates the following plots:
    - figures/threePN-n_templates-scaling.pdf
    - figures/threePN-time-scaling.pdf

To reproduce the plots:

    >>> python plot_scaling.py

"""


def parse_ps() -> Dict[float, Tuple[float, float]]:
    """
    Parses output from job-threePN-est-p.sh.

    Returns
        Dict mapping `mm` to an MC estimate of the covering probability `p` and
        associated error `p_err`.
    """
    with open("../scripts/threePN-p.txt", "r") as f:
        lines = f.readlines()

    ps = {}
    for i in range(len(lines) // 2):
        mm = float(re.search("mm=(\d\.\d*)", lines[2 * i]).group(1))

        match = re.search("p = (\d\.\d*) \+\/\- (\d\.\d*)", lines[2 * i + 1])
        p = float(match.group(1))
        p_err = float(match.group(2))

        ps[mm] = (p, p_err)

    return ps


key = random.PRNGKey(183)
ps = parse_ps()
fs = jnp.linspace(20.0, 2000.0, 1000)
m_range = (1.4, 5.0)
sampler = get_m1_m2_sampler(m_range, m_range)
eta_ref = 0.9
mm_ref = 0.9


def cs_cr_pred(p, eta, n_eff=1000):
    return (
        -2 + eta + 1 / p + ((-1 + p) * jnp.log(1 - eta)) / (eta * jnp.log(1 - p))
    ) / ((-1 + eta) * n_eff)


def cs_cr_pred_err(p_err, p, eta, n_eff=1000):
    return (
        (
            -(p ** (-2))
            + ((-1 + p) * jnp.log(1 - eta)) / (eta * (1 - p) * jnp.log(1 - p) ** 2)
            + jnp.log(1 - eta) / (eta * jnp.log(1 - p))
        )
        / ((-1 + eta) * n_eff)
        * p_err
    )


def plot_n_templates_scaling(axs):
    ns, n_ests, n_est_errs = {}, {}, {}
    ns_s = {}

    # Varying mm
    mms = [0.95, 0.90, 0.85, 0.80, 0.75]
    for seed, mm in enumerate(mms, 5):
        path = os.path.join(
            "../scripts/threePN-banks-scaling",
            f"3pn-random-{seed}-mm={mm}-eta_star={eta_ref}-n_eff=1000.npz",
        )
        # Measurement
        ns[(mm, eta_ref)] = Bank.load(path, amp, Psi, Sn_LIGOI, sampler).n_templates
        # Prediction
        p, p_err = ps[mm]
        n_ests[(mm, eta_ref)] = log(1 - eta_ref) / log(1 - p)
        n_est_errs[(mm, eta_ref)] = (
            log(1 - eta_ref) / ((1 - p) * log(1 - p) ** 2) * p_err
        )

        # Stochastic
        path = os.path.join(
            "../scripts/threePN-banks-scaling",
            f"3pn-stochastic-{seed + 5}-mm={mm}-eta_star={eta_ref}-n_eff=1000.npz",
        )
        # Measurement
        ns_s[(mm, eta_ref)] = Bank.load(path, amp, Psi, Sn_LIGOI, sampler).n_templates

    # Varying eta
    etas = [0.975, 0.95, 0.925, 0.900, 0.875, 0.850]
    for seed, eta in enumerate(etas, 15):
        path = os.path.join(
            "../scripts/threePN-banks-scaling",
            f"3pn-random-{seed}-mm={mm_ref}-eta_star={eta}-n_eff=1000.npz",
        )
        # Measurement
        ns[(mm_ref, eta)] = Bank.load(path, amp, Psi, Sn_LIGOI, sampler).n_templates
        # Prediction
        p, p_err = ps[mm_ref]
        n_ests[(mm_ref, eta)] = log(1 - eta) / log(1 - p)
        n_est_errs[(mm_ref, eta)] = log(1 - eta) / ((1 - p) * log(1 - p) ** 2) * p_err

        # Stochastic
        path = os.path.join(
            "../scripts/threePN-banks-scaling",
            f"3pn-stochastic-{seed + 6}-mm={mm_ref}-eta_star={eta}-n_eff=1000.npz",
        )
        # Measurement
        ns_s[(mm_ref, eta)] = Bank.load(path, amp, Psi, Sn_LIGOI, sampler).n_templates

    ax = axs[0]
    ax.scatter(mms, [ns[(mm, eta_ref)] for mm in mms], c="C0", s=10, label="Random")
    ax.errorbar(
        mms,
        [n_ests[(mm, eta_ref)] for mm in mms],
        2 * jnp.array([n_est_errs[(mm, eta_ref)] for mm in mms]),
        fmt=".",
        c="C1",
        label=r"Estimated ($\pm 2\sigma$)",
    )
    ax.scatter(
        mms,
        [ns_s[(mm, eta_ref)] for mm in mms],
        marker="x",
        c="C2",
        s=10,
        label="Stochastic",
    )

    ax = axs[1]
    ax.scatter(
        etas,
        [ns[(mm_ref, eta)] for eta in etas],
        c="C0",
        s=10,
        label="Random",
    )
    ax.errorbar(
        etas,
        [n_ests[(mm_ref, eta)] for eta in etas],
        2 * jnp.array([n_est_errs[(mm_ref, eta)] for eta in etas]),
        fmt=".",
        c="C1",
        label=r"Estimated ($\pm 2\sigma$)",
    )
    ax.scatter(
        etas,
        [ns_s[(mm_ref, eta)] for eta in etas],
        marker="x",
        c="C2",
        s=10,
        label="Stochastic",
    )
    ax.set_ylim(0, 2100)


def plot_time_scaling(axs):
    runtimes = {}
    filenames = os.listdir("../scripts/threePN-outputs-scaling/")

    # Parse tqdm output to get timing information
    for fn in filenames:
        kind = fn.split("-")[1]
        mm = float(re.search("mm=(\d*\.\d*)-", fn).group(1))
        eta = float(re.search("eta_star=(\d*\.\d*)\.txt", fn).group(1))

        with open(os.path.join("../scripts/threePN-outputs-scaling", fn)) as f:
            raw = f.read()
            last_pbar = raw.split("\n")[-4]

            # Convert to runtime [s]
            search_result = re.search("\[(.*)<", last_pbar)
            if search_result is not None:
                raw_time = search_result.group(1)
            else:
                raw_time = re.search("\[(.*?),", last_pbar).group(1)
            raw_time = raw_time.split(":")
            assert len(raw_time) <= 3
            time = sum([float(rt) * 60 ** i for i, rt in enumerate(reversed(raw_time))])

        runtimes[(kind, mm, eta)] = time

    # mm scaling
    eta_ref = 0.9
    mms = jnp.array([0.75, 0.80, 0.85, 0.90, 0.95])
    p_ests = jnp.array([ps[mm][0] for mm in mms])
    p_errs = jnp.array([ps[mm][1] for mm in mms])
    # eta scaling
    etas = jnp.array([0.850, 0.875, 0.900, 0.925, 0.950, 0.975])
    mm_ref = 0.9
    p_est_ref = ps[mm_ref][0]
    p_err_ref = ps[mm_ref][1]

    ax = axs[0]
    c_ss = jnp.array([runtimes[("stochastic", mm, eta_ref)] for mm in mms])
    c_rs = jnp.array([runtimes[("random", mm, eta_ref)] for mm in mms])
    ax.scatter(mms, c_ss / c_rs, c="C4", s=10, label="Measured")
    ax.errorbar(
        mms,
        cs_cr_pred(p_ests, eta_ref),
        2 * cs_cr_pred_err(p_errs, p_ests, eta_ref),
        fmt=".",
        c="C1",
        label=r"Estimated ($\pm 2\sigma$)",
    )

    ax = axs[1]
    c_ss = jnp.array([runtimes[("stochastic", mm_ref, eta)] for eta in etas])
    c_rs = jnp.array([runtimes[("random", mm_ref, eta)] for eta in etas])
    print(cs_cr_pred(p_est_ref, etas))
    ax.scatter(etas, c_ss / c_rs, c="C4", s=10, label="Measured")
    ax.errorbar(
        etas,
        cs_cr_pred(p_est_ref, etas),
        2 * cs_cr_pred_err(p_err_ref, p_est_ref, etas),
        fmt=".",
        c="C1",
        label=r"Estimated ($\pm 2\sigma$)",
    )
    ax.set_ylim(0, 30)


def run():
    fig, axes = plt.subplots(2, 2, sharey="row")

    plot_n_templates_scaling([axes[0, 0], axes[0, 1]])
    plot_time_scaling([axes[1, 0], axes[1, 1]])

    axes[1, 0].set_xlabel(r"$1 - m_*$")
    axes[1, 1].set_xlabel(r"$\eta$")
    axes[0, 0].set_ylabel(r"$N$")
    axes[1, 0].set_ylabel(r"$C_\mathcal{S} / C_\mathcal{R}$")
    axes[0, 0].set_title(r"$\eta = %g$" % eta_ref)
    axes[0, 1].set_title(r"$1 - m_* = %g$" % mm_ref)
    axes[1, 0].legend(loc="upper left")
    # Custom legend ordering
    ax = axes[0, 0]
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 2, 1]
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig("figures/scaling.pdf")


if __name__ == "__main__":
    run()
