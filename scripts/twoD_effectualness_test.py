import time

from diffbank.bank import Bank
import jax.numpy as jnp
from jax import random

# from diffbank.waveforms.threePN_simple import amp, Psi

from diffbank.waveforms.twoPN_simple import amp, Psi
from diffbank.utils import get_M_eta_sampler, Sn_func
from diffbank.metric import get_metric_ellipse
import matplotlib.pyplot as plt


def gen_effectualness():
    key = random.PRNGKey(89)

    fs = jnp.linspace(30.0, 512.0, 1000)
    M_range = (4.0, 6.0)
    eta_range = (0.13888, 0.25)
    vol = jnp.array((M_range[1] - M_range[0]) * (eta_range[1] - eta_range[0]))
    sampler = get_M_eta_sampler(M_range, eta_range)
    bank = Bank(
        amp,
        Psi,
        fs,
        Sn_func,
        sampler,
        naive_vol=vol,
        m_star=1 - 0.9,
        eta=0.99,
        name="3PN",
    )

    bank.density_max = jnp.array(
        bank.get_density(jnp.array([M_range[0], eta_range[0]]))
    )

    bank.compute_n_templates(key, 1000)
    _, key = random.split(key)

    assert bank.n_templates > 0 and bank.n_templates < 1e5
    print("Number of templates in the bank:", bank.n_templates)

    # Now lets fill the bank
    t0 = time.time()
    bank.fill_bank(key)
    _, key = random.split(key)
    assert len(bank.templates) == bank.n_templates
    t1 = time.time()

    total = t1 - t0
    print("Time to fill bank:", total)
    print("Time spent per template:", total / bank.n_templates)

    # Finally we can test the effectualness and save
    print("Computing Effectualness")
    N_effectualness_points = 1000
    effectualness_points = bank.gen_templates_rejection(key, N_effectualness_points)
    mask = effectualness_points
    # effectualness_points = sampler(key, N_effectualness_points)
    bank.compute_effectualnesses(effectualness_points)
    print(
        "Fraction of templates above 0.95:",
        sum(bank.effectualnesses > 0.95) / bank.effectualnesses.size,
    )
    bank.save_bank()

    # Plot
    # plt.figure(figsize=(6, 5))
    # plt.scatter(bank.templates[:, 0], bank.templates[:, 1], s=10, c="C1", marker="x")
    # plt.scatter(
    #     effectualness_points[:, 0], effectualness_points[:, 1], s=10, c="C0", marker="o"
    # )
    # plt.xlabel("M")
    # plt.ylabel("eta")
    # plt.savefig("figures/M_eta_highmass.pdf", bbox_inches="tight")


def check_metric_ellipse():
    from matplotlib.patches import Ellipse

    fs = jnp.linspace(30.0, 512.0, 1000)
    theta = jnp.array([4.0, 0.13888])
    p = get_metric_ellipse(theta, amp, Psi, fs, Sn_func)

    plt.figure()
    ax = plt.gca()
    print(p[0].real, p[1].real, jnp.rad2deg(p[2].real))
    ellipse = Ellipse(
        xy=theta, width=p[0].real, height=p[1].real, angle=jnp.rad2deg(p[2].real)
    )
    plt.scatter(theta[0], theta[1], c="C1", marker="x")
    ax.add_patch(ellipse)
    ax.set_xlim(3.5, 4.5)
    ax.set_ylim(0.11, 0.18)

    plt.savefig("figures/test_ellipse.pdf", bbox_inches="tight")
    print(p)


if __name__ == "__main__":
    gen_effectualness()
    # check_metric_ellipse()