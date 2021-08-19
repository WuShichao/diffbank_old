from math import pi

import click
from scipy.optimize import minimize_scalar

from diffbank.bank import Bank
from diffbank.constants import C, G, MSUN
from diffbank.noise import Sn_aLIGOZeroDetHighPower as Sn
from diffbank.waveforms.taylorf2reducedspin import Psi, amp, get_th_boundary_interps
import jax
from jax import random
import jax.numpy as jnp

"""
Generate a TaylorF2ReducedSpin bank for comparison with Ajith et al 2014,
https://arxiv.org/abs/1210.6666.

Note: this is not checking that chi is in bounds correctly yet.
"""


def get_th3S_max(th0, th3):
    """
    Gets max value of th3S at a given `(th0, th3)` point. This computes the
    component masses, gets the corresponding `chi1`, `chi2` values, computes
    the max value `chi` can take and converts this to a max value for `th3S`.
    """
    M_chirp = 1 / (16 * pi * f_0) * (125 / (2 * th0 ** 3)) ** (1 / 5) * C ** 3 / G
    eta = (16 * pi ** 5 / 25 * th0 ** 2 / th3 ** 5) ** (1 / 3)
    q = (1 + jnp.sqrt(1 - 4 * eta) - 2 * eta) / (2 * eta)
    m2 = (1 + q) ** (1 / 5) / q ** (3 / 5) * M_chirp
    m1 = q * m2
    delta = (m1 - m2) / (m1 + m2)
    chi1_max = jnp.where(m1 > m_thresh, chi_bh_max, chi_ns_max)
    chi2_max = jnp.where(m2 > m_thresh, chi_bh_max, chi_ns_max)
    chi_s_max = (chi1_max + chi2_max) / 2
    chi_a_max = (chi1_max - chi2_max) / 2
    chi_max = chi_s_max * (1 - 76 * eta / 113) + delta * chi_a_max
    th3S_max = 113 * th3 * chi_max / (48 * pi)
    return th3S_max


key = random.PRNGKey(100)

minimum_match = 0.95
m_star = 1 - minimum_match
eta_star = 0.99
fs = jnp.linspace(20.0, 2000.0, 1000)
f_0 = 20.0  # Hz
m_range = (1.4 * MSUN, 20.0 * MSUN)
m_thresh = 2.0 * MSUN
chi_bh_max = 0.98
chi_ns_max = 0.4

th0_range, th3_interp_low, th3_interp_high = get_th_boundary_interps(*m_range, f_0)
# Figure out where th3 attains its maximum
bracket = (th0_range[0], 5e3)  # NOTE: need to change if m_range changes!
res = minimize_scalar(lambda th0: -th3_interp_high(th0), bracket, bracket)
assert res.success
th0_th3_max = res.x
th3_max = -res.fun
th3_range = (th3_interp_low(th0_range[0]), th3_max)
# Maximum value of th3
th3S_max = get_th3S_max(th0_th3_max, th3_max)


def is_in_bounds(theta: jnp.ndarray) -> jnp.ndarray:
    """
    Checks if a point is in bounds.
    """
    th0, th3, th3S = theta[..., 0], theta[..., 1], theta[..., 2]
    return jnp.logical_and(
        th3 > th3_interp_low(th0),
        jnp.logical_and(
            th3 < th3_interp_high(th0), jnp.abs(th3S) < get_th3S_max(th0, th3)
        ),
    )


def base_sample_1(key: jnp.ndarray) -> jnp.ndarray:
    """
    Sample uniformly over maximum parameter ranges.
    """
    return random.uniform(
        key,
        (3,),
        minval=jnp.array([th0_range[0], th3_range[0], -th3S_max]),
        maxval=jnp.array([th0_range[1], th3_range[1], th3S_max]),
    )


@jax.jit
def sample_1(key: jnp.ndarray) -> jnp.ndarray:
    """
    Samples a single point with rejection sampling.
    """
    cond_fun = lambda val: jnp.logical_not(is_in_bounds(val[1]))

    def body_fun(val):
        key = val[0]
        key, subkey = random.split(key)
        return (key, base_sample_1(subkey))

    key, subkey = random.split(key)
    init_val = (key, base_sample_1(subkey))
    return jax.lax.while_loop(cond_fun, body_fun, init_val)[1]


def sampler(key, n):
    return jax.lax.map(sample_1, random.split(key, n))


@click.command()
@click.option("--seed", type=int, help="PRNG seed")
@click.option("--kind", type=str, help="kind of bank ('random' or 'stochastic')")
def run(seed, kind):
    key = random.PRNGKey(seed)

    bank = Bank(
        amp, Psi, fs, Sn, m_star, eta_star, sampler, name=f"tf2rs-{kind}-{seed}"
    )

    # Get max density
    # NOTE: need to change if m_range changes!
    th0s = jnp.linspace(10.5e3, 11e3, 500)
    th3s = th3_interp_low(th0s) * 1.0001
    th3Ss = -get_th3S_max(th0s, th3s)
    boundary_densities = jax.lax.map(
        bank.density_fun, jnp.stack([th0s, th3s, th3Ss], -1)
    )
    bank.ratio_max = boundary_densities.max()

    # Fill bank
    key, subkey = random.split(key)
    bank.fill_bank(subkey, kind)
    bank.save("banks/")

    # Get effectualnesses
    key, subkey = random.split(key)
    bank.calc_bank_effectualness(subkey, 500)
    bank.save("banks/")


if __name__ == "__main__":
    run()
