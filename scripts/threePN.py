import click
import os
import jax
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
@click.option(
    "--n-eta",
    default=500,
    type=int,
    help="number of new points at which to compute effectualnesses",
)
@click.option("--mm", default=0.95, help="minimum match")
@click.option("--eta-star", default=0.99, help="eta*")
@click.option("--n-eff", default=1000)
@click.option("--savedir", default="banks", help="directory in which to save the bank")
def run(seed, kind, n_eta, mm, eta_star, n_eff, savedir):
    os.makedirs(savedir, exist_ok=True)

    key = random.PRNGKey(seed)
    m_star = 1 - mm

    fs = jnp.linspace(20.0, 2000.0, 1000)
    m_range = (1.4, 5.0)
    sampler = get_m1_m2_sampler(m_range, m_range)

    bank = Bank(
        amp,
        Psi,
        fs,
        Sn_aLIGO,
        m_star,
        eta_star,
        sampler,
        name=f"3pn-{kind}-{seed}-mm={mm}-eta_star={eta_star}-n_eff={n_eff}",
    )

    # Get max density
    fun = lambda m1: -bank.density_fun(jnp.array([m1, m_range[0]]))
    res = minimize_scalar(fun, bracket=m_range, bounds=m_range, method="bounded")
    assert res.success
    theta_dmax = jnp.array([res.x, m_range[0]])
    ratio_max = bank.density_fun(theta_dmax)
    bank.ratio_max = ratio_max

    # Fill bank
    key, subkey = random.split(key)
    bank.fill_bank(subkey, kind, n_eff)
    bank.save(savedir)
    print(f"Saved bank to {os.path.join(savedir, bank.name + '.npz')}")

    # Get effectualnesses
    if n_eta > 0:
        key, subkey = random.split(key)
        bank.calc_bank_effectualness(subkey, n_eta)
        bank.save(savedir)
    else:
        print("Skipping effectualnesses calculation")


if __name__ == "__main__":
    run()
