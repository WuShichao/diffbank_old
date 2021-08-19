from math import pi

import click
import jax
from jax import random
import jax.numpy as jnp

from diffbank.bank import Bank
from diffbank.constants import C, G, MSUN
from diffbank.noise import Sn_aLIGOZeroDetHighPower as Sn
from diffbank.waveforms.taylorf2reducedspin import Psi, amp, get_th_boundary_interps

"""
Generate a TaylorF2ReducedSpin bank.

Note: this is not checking that chi is in bounds correctly yet.
"""


@click.command()
@click.option("--seed", type=int, help="PRNG seed")
@click.option("--kind", type=str, help="kind of bank ('random' or 'stochastic')")
def run(seed, kind):
    key = random.PRNGKey(seed)

    minimum_match = 0.95
    m_star = 1 - minimum_match
    eta_star = 0.99
    f_0 = 20.0  # Hz
    fs = jnp.linspace(f_0, 2000.0, 1000)
    m_range = (2.0 * MSUN, 5.0 * MSUN)
    chi_range = (-0.98, 0.98)

    th0_range, th3_interp_low, th3_interp_high = get_th_boundary_interps(*m_range, f_0)
    th3_range = (th3_interp_low(th0_range[0]), th3_interp_low(th0_range[1]))
    th3S_max = (
        113
        * chi_range[1]
        / (
            192
            * f_0 ** (2 / 3)
            * pi ** (2 / 3)
            * (
                m_range[0] ** 2
                / (m_range[0] + m_range[0]) ** (4 / 3)
                * (G / C ** 3) ** (2 / 3)
            )
        )
    )
    th3S_range = (-th3S_max, th3S_max)

    @jax.jit
    def sampler_1(key):
        return random.uniform(
            key,
            (3,),
            minval=jnp.array([th0_range[0], th3_range[0], th3S_range[0]]),
            maxval=jnp.array([th0_range[1], th3_range[1], th3S_range[1]]),
        )

    @jax.jit
    def is_in_bounds(theta):
        chi = 48 * pi * theta[..., 2] / (113 * theta[..., 1])
        return jnp.logical_and(
            theta[..., 1] > th3_interp_low(theta[..., 0]),
            jnp.logical_and(
                theta[..., 1] < th3_interp_high(theta[..., 0]),
                jnp.abs(chi) < chi_range[1],
            ),
        )

    @jax.jit
    def sample_1(key):
        cond_fun = lambda val: jnp.logical_not(is_in_bounds(val[1]))

        def body_fun(val):
            key = val[0]
            key, subkey = random.split(key)
            return (key, sampler_1(subkey))

        key, subkey = random.split(key)
        init_val = (key, sampler_1(subkey))
        return jax.lax.while_loop(cond_fun, body_fun, init_val)[1]

    def sampler(key, n):
        return jax.lax.map(sample_1, random.split(key, n))

    bank = Bank(
        amp, Psi, fs, Sn, m_star, eta_star, sampler, name=f"tf2rs-{kind}-{seed}"
    )

    # Get max density
    ratio_max = bank.density_fun(jnp.stack([th0_range[1], th3_range[1], th3S_range[0]]))
    bank.ratio_max = ratio_max

    # Fill bank
    key, subkey = random.split(key)
    bank.fill_bank(subkey, kind)

    # Get effectualnesses
    key, subkey = random.split(key)
    bank.calc_bank_effectualness(subkey, 500)
    bank.save("banks/")


if __name__ == "__main__":
    run()
