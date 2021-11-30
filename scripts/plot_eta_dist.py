from math import log
from operator import itemgetter

import jax.numpy as jnp
import matplotlib.pyplot as plt

from plot_scaling import parse_ps

plt.style.use("../plot_style.mplstyle")

"""
Plots distribution of `eta` (coverage) values for a set of 3.5PN-2D banks.

The requisite banks are generated by running `job-threePN-coverage.sh` and
`job-threePN-stochastic.sh`.

To reproduce the plot:

    >>> python plot_eta_dist.py

"""

mm = 0.95
eta = 0.9
eta_est_s, eta_est_err_s, n_templates_s = itemgetter(
    "eta_est", "eta_est_err", "n_templates"
)(
    jnp.load(
        f"banks/3pn-stochastic-1000-mm={mm}-eta_star={eta}-n_eff=500.npz",
        allow_pickle=True,
    )["bank"].item()
)


def plot_eta_dist():
    # Load random banks
    eta_ests = []
    eta_est_errs = []
    n_templatess = []
    for i in range(100, 131):
        if i == 119:  # this bank got interrupted
            continue

        eta_est, eta_est_err, n_templates = itemgetter(
            "eta_est", "eta_est_err", "n_templates"
        )(
            jnp.load(
                f"banks/3pn-random-{i}-mm={mm}-eta_star={eta}-n_eff=500.npz",
                allow_pickle=True,
            )["bank"].item()
        )
        eta_ests.append(eta_est)
        eta_est_errs.append(eta_est_err)
        n_templatess.append(n_templates)

    eta_ests = jnp.array(eta_ests)
    eta_est_errs = jnp.array(eta_est_errs)
    n_templatess = jnp.array(n_templatess)

    # Random banks
    plt.errorbar(
        n_templatess,
        eta_ests,
        yerr=2 * eta_est_errs,
        fmt=".",
        c="C0",
        label=r"Random ($\pm 2\sigma$)",
    )
    # Stochastic bank
    plt.errorbar(
        n_templates_s,
        eta_est_s,
        yerr=2 * eta_est_err_s,
        fmt=".",
        c="C1",
        label=r"Stochastic ($\pm 2\sigma$)",
    )

    # Target eta
    plt.axhline(0.9, color="k", linestyle="--", linewidth=1)

    plt.xlabel(r"$N$")
    plt.ylabel(r"$\eta$")
    plt.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig("figures/eta-dist.pdf")


if __name__ == "__main__":
    plot_eta_dist()
