import time

from diffbank.bank import Bank
import jax.numpy as jnp
from jax import random, jit
from typing import Callable

from diffbank.waveforms.threePN_simple import amp, Psi

# from diffbank.waveforms.twoPN_simple import amp, Psi
from diffbank.utils import get_M_eta_sampler
from diffbank.noise import Sn_LIGOI
from diffbank.metric import get_metric_ellipse
import matplotlib.pyplot as plt


def gen_effectualness():
    key = random.PRNGKey(89)

    fs = jnp.linspace(30.0, 512.0, 1000)
    M_range = (4.0, 6.0)
    eta_range = (0.13888, 0.25)
    vol = jnp.array((M_range[1] - M_range[0]) * (eta_range[1] - eta_range[0]))
    sampler = get_M_eta_sampler(M_range, eta_range)

    @jit
    def is_in_bounds(pt):
        """
        Acceptance probability for tau rejection sampling.
        """
        M, eta = pt
        # There's got to be a better way to write this.
        return jnp.where(
            M < M_range[0],
            jnp.array(0.0),
            jnp.where(
                M > M_range[1],
                jnp.array(0.0),
                jnp.where(
                    eta < eta_range[0],
                    jnp.array(0.0),
                    jnp.where(
                        eta > eta_range[1],
                        jnp.array(0.0),
                        jnp.array(1.0),
                    ),
                ),
            ),
        )

    mismatch = 0.95
    eta = 0.99
    bank = Bank(
        amp,
        Psi,
        fs,
        Sn_LIGOI,
        sampler,
        m_star=1 - mismatch,
        eta=eta,
        name="3PN",
    )

    bank.density_max = jnp.array(
        bank.density_fun(jnp.array([M_range[0], eta_range[0]]))
    )

    t0 = time.time()
    key, subkey = random.split(key)
    bank.fill_bank(subkey, n_eff=500)
    assert len(bank.templates) == bank.n_templates
    t1 = time.time()
    total = t1 - t0
    print("Time to fill bank:", total)
    print("Time spent per template:", total / bank.n_templates)

    # Finally we can test the effectualness and save
    print("Computing effectualness")
    key, subkey = random.split(key)
    bank.calc_bank_effectualness(subkey, 1000)
    bank.save()


def check_metric_ellipse():
    from matplotlib.patches import Ellipse

    fs = jnp.linspace(30.0, 512.0, 1000)
    theta = jnp.array([4.0, 0.13888])
    p = get_metric_ellipse(theta, amp, Psi, fs, Sn_LIGOI)

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
    check_metric_ellipse()
