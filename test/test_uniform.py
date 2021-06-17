from math import sqrt
import os

import click
import jax
from jax import random
import jax.numpy as jnp
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import trange

from diffbank.utils import get_n_templates, get_template_frac_in_bounds

"""
Tests consistency of random template bank using a simple waveform model with
Euclidean mismatch and metric.

More precisely, generates a bank, computes effectualnesses for signal
injections, and checks that the +/- 5 sigma MC estimate of the fraction with
effectualness greater than the minimum match includes eta.
"""

# Globals


def get_g(_):
    return jnp.identity(2)


def density_fun(_):
    return jnp.array(1.0)


@jax.jit
def get_eff(x, y):
    """
    Consistent with the metric being the Euclidean one. This is the same as the
    match since there are no extrinsic parameters in this toy scenario.
    """
    return 1 - jnp.sum((x - y) ** 2)


def _run(key, m1_max, m2_max, eta, mm, plot, suffix="", save_dir=""):
    key = random.PRNGKey(key)
    m_min, m1_max, m2_max = jnp.array(0.0), jnp.array(m1_max), jnp.array(m2_max)
    naive_vol = (m1_max - m_min) * (m2_max - m_min)
    m_star = 1 - mm  # maximum mismatch

    # Uniform sampler
    def sampler(key, n):
        return random.uniform(
            key,
            (n, 2),
            minval=jnp.stack([m_min, m_min]),
            maxval=jnp.stack([m1_max, m2_max]),
        )

    def is_in_bounds(theta):
        m1, m2 = theta
        return jnp.where(
            m1 < m_min,
            jnp.array(0.0),
            jnp.where(
                m1 > m1_max,
                jnp.array(0.0),
                jnp.where(
                    m2 < m_min,
                    jnp.array(0.0),
                    jnp.where(m2 > m2_max, jnp.array(0.0), jnp.array(1.0)),
                ),
            ),
        )

    # Correct for boundary effects
    key, pts_key, ell_key = random.split(key, 3)
    pts_ib = sampler(pts_key, 1000)
    frac_in_bounds, frac_in_bounds_err = get_template_frac_in_bounds(
        ell_key, pts_ib, get_g, m_star, is_in_bounds, 100
    )
    assert frac_in_bounds_err / frac_in_bounds < 0.1
    print(
        f"Fraction of template volume in bounds: {frac_in_bounds} +/- {frac_in_bounds_err}"
    )

    # Generate bank
    key, subkey = random.split(key)
    n_templates, n_templates_err = get_n_templates(
        subkey,
        1000,
        density_fun,
        sampler,
        eta,
        m_star,
        naive_vol,
        frac_in_bounds,
        frac_in_bounds_err=frac_in_bounds_err,
    )
    assert n_templates_err / n_templates < 0.1
    print(f"{n_templates} +/- {n_templates_err} templates required")

    n_templates_Zn = int(naive_vol / (2 * m_star / sqrt(2)) ** 2)
    print(f"{n_templates_Zn} templates required for a Z_n lattice")

    # Create the bank
    key, subkey = random.split(key)
    templates = sampler(subkey, n_templates)

    # Compute effectualness
    n_test = 500
    key, subkey = random.split(key)
    points = sampler(subkey, n_test)
    effectualnesses = np.zeros(n_test)

    for i in trange(n_test):
        effectualnesses[i] = jax.lax.map(
            lambda template: get_eff(template, points[i]), templates
        ).max()

    eff_frac = jnp.mean(effectualnesses >= mm)
    eff_frac_err = jnp.std(effectualnesses >= mm) / sqrt(len(effectualnesses))

    # Test that the bank was generated consistently
    assert eff_frac - 5 * eff_frac_err < eta and eta < eff_frac + 5 * eff_frac_err
    print(
        f"Effectualness: [{eff_frac - 2 * eff_frac_err}, {eff_frac + 2 * eff_frac_err}]"
    )

    if plot:
        # Plot
        plt.figure(figsize=(8, 3.2))

        # Effectualnesses
        plt.subplot(1, 2, 1)
        plt.hist(effectualnesses)
        plt.axvline(mm, color="r")
        plt.xlabel("Effectualness")
        plt.ylabel("Frequency")
        plt.title(
            f"[{eff_frac - 2 * eff_frac_err:.3f}, {eff_frac + 2 * eff_frac_err:.3f}]"
            f" above {m_star:.3f}"
        )

        # Templates
        plt.subplot(1, 2, 2)
        circles = []
        for x, y in templates:
            circles.append(plt.Circle((x, y), radius=sqrt(m_star)))
        circles = PatchCollection(circles, zorder=0, alpha=0.2)
        plt.gca().add_collection(circles)
        plt.scatter(*templates.T, s=15, c="C1", marker="x")

        # Effectualness points
        plt.scatter(*points.T, s=1, c="r")

        # Boundary
        plt.plot(
            [m_min, m1_max, m1_max, m_min, m_min],
            [m_min, m_min, m2_max, m2_max, m_min],
            "k",
            linewidth=0.5,
        )

        plt.xlim(m_min - 0.5, m1_max + 0.5)
        plt.ylim(m_min - 0.5, m2_max + 0.5)
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.title("Template bank")
        # plt.gca().set_aspect("equal")

        plt.suptitle(r"$^{%g}\mathcal{R}_{%i}(%g)$" % (eta, 2, mm))
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"test-uniform{suffix}.png")
        plt.savefig(os.path.join(save_dir, fig_path))
        print(f"Saved figure to {fig_path}")


def test():
    _run(45389, 2.0, 2.0, 0.99, 0.95, False)


@click.command()
@click.option("--key", default=8539, help="PRNG key")
@click.option("--m1_max", default=1.5, help="maximum parameter value")
@click.option("--m2_max", default=1.5, help="maximum parameter value")
@click.option("--eta", default=0.99)
@click.option("--mm", default=0.9, help="minimum match")
@click.option("--plot/--no-plot", default=False, help="save plot")
@click.option("--suffix", default="", help="suffix for plot filename")
@click.option("--save-dir", default="", help="directory in which to save plot")
def run(key, m1_max, m2_max, eta, mm, plot, suffix, save_dir):
    if suffix != "":
        plot = True
    _run(key, m1_max, m2_max, eta, mm, plot, suffix, save_dir)


if __name__ == "__main__":
    m1_max = 100.0
    m2_max = 0.05
    save_dir = f"{m1_max:g}x{m2_max:g}"
    for key in [0]:  # jnp.arange(0, 10, 1, dtype=int):
        suffix = f"-{m1_max:g}x{m2_max:g}-{key}"
        _run(key, m1_max, m2_max, 0.99, 0.95, True, suffix, save_dir)
