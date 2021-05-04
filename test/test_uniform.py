import click
from math import sqrt

from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np

from diffbank.utils import get_n_templates
import jax
from jax import random
import jax.numpy as jnp
from tqdm.auto import trange

"""
Tests consistency of random template bank using a simple waveform model with
Euclidean mismatch and metric.

More precisely, generates a bank, computes effectualnesses for signal
injections, and checks that the +/- 5 sigma MC estimate of the fraction with
effectualness greater than the minimum match includes eta.
"""

# Globals

# Euclidean metric
def density_fun(_):
    return jnp.array(1.0)


@jax.jit
def get_eff(x, y):
    """
    Consistent with the metric being the Euclidean one. This is the same as the
    match since there are no extrinsic parameters in this toy scenario.
    """
    return 1 - jnp.sum((x - y) ** 2)


@click.command()
@click.option("--key", default=8539, help="PRNG key")
@click.option("--m_max", default=1.5, help="maximum parameter value")
@click.option("--eta", default=0.99)
@click.option("--mm", default=0.9, help="minimum match")
@click.option("--plot/--no-plot", default=False, help="save plot")
def run(key, m_max, eta, mm, plot):
    m_min, m_max = jnp.array(0.0), jnp.array(m_max)
    naive_vol = (m_max - m_min) ** 2
    m_star = 1 - mm  # maximum mismatch

    # Uniform sampler
    def sampler(key, n):
        return random.uniform(key, (n, 2), minval=m_min, maxval=m_max)

    # Generate bank
    key = random.PRNGKey(key)
    n_templates = get_n_templates(
        key, naive_vol, 1000, density_fun, sampler, eta, m_star
    )
    _, key = random.split(key)

    print(f"{n_templates} templates required")
    n_templates_Zn = int(naive_vol / (2 * m_star / sqrt(2)) ** 2)
    print(f"{n_templates_Zn} templates required for a Z_n lattice")

    templates = sampler(key, n_templates)
    _, key = random.split(key)

    # Compute effectualness
    n_test = 250
    points = sampler(key, n_test)
    _, key = random.split(key)
    effectualnesses = np.zeros(n_test)

    for i in trange(n_test):
        effectualnesses[i] = jax.lax.map(
            lambda template: get_eff(template, points[i]), templates
        ).max()

    eff_frac = jnp.mean(effectualnesses >= mm)
    eff_frac_err = jnp.std(effectualnesses >= mm) / sqrt(len(effectualnesses))

    # Test that the bank was generated consistently
    assert eff_frac - 5 * eff_frac_err < eta and eta < eff_frac + 5 * eff_frac_err

    if plot:
        # Plot
        plt.figure(figsize=(8, 3.2))

        # Effectualnesses
        plt.subplot(1, 2, 1)
        plt.hist(effectualnesses)
        plt.axvline(mm, color="r")
        plt.xlabel("Effectualness")
        plt.ylabel("Frequency")
        plt.title(r"$%.3f \pm %.3f$ above %g" % (eff_frac, eff_frac_err, m_star))

        # Templates
        plt.subplot(1, 2, 2)
        circles = []
        for x, y in templates:
            circles.append(plt.Circle((x, y), radius=sqrt(m_star)))
        circles = PatchCollection(circles, zorder=0, alpha=0.2)
        plt.gca().add_collection(circles)
        plt.scatter(*templates.T, s=15, c="C1", marker="x")
        plt.scatter(*points.T, s=1, c="r")
        plt.xlim(m_min, m_max)
        plt.ylim(m_min, m_max)
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.title("Template bank")
        plt.gca().set_aspect("equal")

        plt.suptitle(r"$^{%g}\mathcal{R}_{%i}(%g)$" % (eta, 2, mm))
        plt.tight_layout()
        plt.savefig("test-uniform.png")


def test():
    run(45389, 2.0, 0.99, 0.95, False)


if __name__ == "__main__":
    run()
