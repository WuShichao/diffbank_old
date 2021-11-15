from math import log
import os
import re

from diffbank.bank import Bank
from diffbank.noise import Sn_aLIGO
from diffbank.utils import get_m1_m2_sampler
from diffbank.waveforms.threePN_simple import Psi, amp
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt

key = random.PRNGKey(183)

# Coverage probabilities
# mm -> (p_est, p_est_err), from threePN-p.txt
ps = {
    0.95: (0.0022500000000000003, 0.0003350326476628807),
    0.9: (0.004, 0.000446318272088428),
    0.85: (0.00585, 0.0005392484353245727),
    0.8: (0.01025, 0.000712212661358951),
    0.75: (0.01, 0.0007035623639735144),
}
fs = jnp.linspace(20.0, 2000.0, 1000)
m_range = (1.4, 5.0)
sampler = get_m1_m2_sampler(m_range, m_range)


def cs_cr_pred(p, eta_star, n_eff=1000):
    return 1 / (n_eff * p) * (1 + jnp.log(1 - eta_star)) / (eta_star - 1)


def cs_cr_pred_err(p_err, p, eta_star, n_eff=1000):
    return cs_cr_pred(p, eta_star, n_eff) / p * p_err


def plot_time_scaling():
    runtimes = {}
    filenames = os.listdir("../scripts/threePN-outputs-scaling/")

    for fn in filenames:
        kind = fn.split("-")[1]
        mm = float(re.search("mm=(\d*\.\d*)-", fn).group(1))
        eta_star = float(re.search("eta_star=(\d*\.\d*)\.txt", fn).group(1))

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

        runtimes[(kind, mm, eta_star)] = time

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    # mm scaling
    eta_star_ref = 0.9
    mms = jnp.array([0.75, 0.80, 0.85, 0.90, 0.95])
    p_ests = jnp.array([ps[mm][0] for mm in mms])
    p_errs = jnp.array([ps[mm][1] for mm in mms])
    # eta_star scaling
    eta_stars = jnp.array([0.850, 0.875, 0.900, 0.925, 0.950, 0.975])
    mm_ref = 0.9
    p_est_ref = ps[mm_ref][0]
    p_err_ref = ps[mm_ref][1]

    ax = axs[0]
    c_ss = jnp.array([runtimes[("stochastic", mm, eta_star_ref)] for mm in mms])
    c_rs = jnp.array([runtimes[("random", mm, eta_star_ref)] for mm in mms])
    ax.plot(mms, c_ss / c_rs, ".", label="Measured")
    ax.errorbar(
        mms,
        cs_cr_pred(p_ests, eta_star_ref),
        2 * cs_cr_pred_err(p_errs, p_ests, eta_star_ref),
        label=r"Estimated ($\pm 2\sigma$)",
        fmt=".",
    )
    ax.set_xlabel("Minimum match")
    ax.set_ylabel(r"$C_\mathcal{S} / C_\mathcal{R}$")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", frameon=False)

    ax = axs[1]
    c_ss = jnp.array(
        [runtimes[("stochastic", mm_ref, eta_star)] for eta_star in eta_stars]
    )
    c_rs = jnp.array([runtimes[("random", mm_ref, eta_star)] for eta_star in eta_stars])
    ax.plot(eta_stars, c_ss / c_rs, ".", label="Measured")
    ax.errorbar(
        eta_stars,
        cs_cr_pred(p_est_ref, eta_stars),
        2 * cs_cr_pred_err(p_err_ref, p_est_ref, eta_stars),
        label=r"Estimated ($\pm 2\sigma$)",
        fmt=".",
    )
    ax.set_xlabel(r"$\eta_*$")
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig("figures/threePN-time-scaling.pdf")


def plot_n_templates_scaling():
    ns, n_ests, n_est_errs = {}, {}, {}

    # Varying mm
    eta_star_ref = 0.9
    mms = [0.95, 0.90, 0.85, 0.80, 0.75]
    for seed, mm in enumerate(mms, 5):
        path = os.path.join(
            "../scripts/threePN-banks-scaling",
            f"3pn-random-{seed}-mm={mm}-eta_star={eta_star_ref}-n_eff=1000.npz",
        )
        # Measurement
        ns[(mm, eta_star_ref)] = Bank.load(
            path, amp, Psi, Sn_aLIGO, sampler
        ).n_templates
        # Prediction
        p, p_err = ps[mm]
        n_ests[(mm, eta_star_ref)] = log(1 - eta_star_ref) / log(1 - p)
        n_est_errs[(mm, eta_star_ref)] = (
            log(1 - eta_star_ref) / ((1 - p) * log(1 - p) ** 2) * p_err
        )

    # Varying eta_star
    mm_ref = 0.9
    eta_stars = [0.975, 0.95, 0.925, 0.900, 0.875, 0.850]
    for seed, eta_star in enumerate(eta_stars, 15):
        path = os.path.join(
            "../scripts/threePN-banks-scaling",
            f"3pn-random-{seed}-mm={mm_ref}-eta_star={eta_star}-n_eff=1000.npz",
        )
        # Measurement
        ns[(mm_ref, eta_star)] = Bank.load(
            path, amp, Psi, Sn_aLIGO, sampler
        ).n_templates
        # Prediction
        p, p_err = ps[mm_ref]
        n_ests[(mm_ref, eta_star)] = log(1 - eta_star) / log(1 - p)
        n_est_errs[(mm_ref, eta_star)] = (
            log(1 - eta_star) / ((1 - p) * log(1 - p) ** 2) * p_err
        )

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharey=True)

    ax = axs[0]
    ax.scatter(mms, [ns[(mm, eta_star_ref)] for mm in mms], c="C0", label="Generated")
    ax.errorbar(
        mms,
        [n_ests[(mm, eta_star_ref)] for mm in mms],
        2 * jnp.array([n_est_errs[(mm, eta_star_ref)] for mm in mms]),
        fmt=".",
        c="C1",
        label=r"Estimated ($\pm 2\sigma$)",
    )
    ax.set_xlabel("Minimum match")
    ax.set_ylabel("Number of templates")
    ax.legend(loc="upper left", frameon=False)

    ax = axs[1]
    ax.scatter(
        eta_stars,
        [ns[(mm_ref, eta_star)] for eta_star in eta_stars],
        c="C0",
        label="Generated",
    )
    ax.errorbar(
        eta_stars,
        [n_ests[(mm_ref, eta_star)] for eta_star in eta_stars],
        2 * jnp.array([n_est_errs[(mm_ref, eta_star)] for eta_star in eta_stars]),
        fmt=".",
        c="C1",
        label=r"Estimated ($\pm 2\sigma$)",
    )
    ax.set_xlabel(r"$\eta_*$")
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig("figures/threePN-n_templates-scaling.pdf")


if __name__ == "__main__":
    plot_time_scaling()
    plot_n_templates_scaling()
