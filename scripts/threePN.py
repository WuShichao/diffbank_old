import click
from jax import random
import jax.numpy as jnp
from scipy.optimize import minimize_scalar

from diffbank.bank import Bank
from diffbank.noise import Sn_aLIGO
from diffbank.utils import get_m1_m2_sampler
from diffbank.waveforms.threePN_simple import Psi, amp

"""
Generate a 3PN bank.
"""


@click.command()
@click.option("--seed", type=int, help="PRNG seed")
@click.option("--kind", type=str, help="kind of bank ('random' or 'stochastic')")
def run(seed, kind):
    key = random.PRNGKey(seed)

    minimum_match = 0.95
    m_star = 1 - minimum_match
    eta_star = 0.99
    fs = jnp.linspace(20.0, 2000.0, 1000)
    m_range = (1.4, 5.0)
    sampler = get_m1_m2_sampler(m_range, m_range)

    bank = Bank(
        amp, Psi, fs, Sn_aLIGO, m_star, eta_star, sampler, name=f"3pn-{kind}-{seed}"
    )

    # Get max density
    fun = lambda m1: -bank.density_fun(jnp.array([m1, m_range[0]]))
    res = minimize_scalar(fun, bracket=m_range, bounds=m_range)
    assert res.success
    theta_dmax = jnp.array([res.x, m_range[0]])
    ratio_max = bank.density_fun(theta_dmax)
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
