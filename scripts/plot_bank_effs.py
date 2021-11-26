import matplotlib.pyplot as plt
import jax.numpy as jnp

plt.style.use("../plot_style.mplstyle")

"""
Plots effectualness histograms for the main three banks in the diffbank paper.

The banks are generated by running the following scripts:

    >>> python genbank_2D_threePN.py --seed 1 --kind random
    >>> python genbank_3D_taylorf2reducedspin.py --seed 1 --kind random
    >>> python genbank_4D_taylorf2.py --seed 1 --kind random
"""


if __name__ == "__main__":
    effs_3pn = jnp.load(
        "../scripts/banks/3pn-random-1-mm=0.95-eta_star=0.9-n_eff=1000.npz",
        allow_pickle=True,
    )["bank"].item()["effectualnesses"]
    effs_tf2rs = jnp.load(
        "../scripts/banks/tf2rs-random-1-mm=0.95-eta_star=0.9-n_eff=1000.npz",
        allow_pickle=True,
    )["bank"].item()["effectualnesses"]
    effs_tf2 = jnp.load(
        "../scripts/banks/tf2-random-1-mm=0.95-eta_star=0.9-n_eff=1000.npz",
        allow_pickle=True,
    )["bank"].item()["effectualnesses"]

    hist_kwargs = dict(
        cumulative=True, density=True, histtype="step", bins=jnp.linspace(0, 1.0, 101)
    )
    plt.hist(effs_3pn, label="3.5PN-2D", **hist_kwargs)
    plt.hist(effs_tf2rs, label="2.5PN-3D", **hist_kwargs)
    plt.hist(effs_tf2, label="3.5PN-4D", **hist_kwargs)

    plt.axvline(0.95, color="k", linestyle="--", linewidth=1)
    plt.axhline(1 - 0.9, color="k", linestyle="--", linewidth=1)
    plt.text(0.954, 1e-2, r"$m_*$")
    plt.text(0.87, 1.3e-1, r"$1 - \eta$")

    plt.xlim(0.7, 1.0)
    plt.yscale("log")
    plt.legend(loc="upper left", frameon=False)
    plt.xlabel("Effectualness")
    plt.ylabel("CDF")

    plt.tight_layout()
    plt.savefig("figures/bank_effs.pdf")
