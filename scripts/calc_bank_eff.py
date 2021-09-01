from os.path import split, splitext

import click

from diffbank.bank import Bank
from jax import random
import jax.numpy as jnp

"""
Update eta estimate for an existing bank using more points and resave.
"""


def load_bank(path):
    waveform = splitext(split(path)[1])[0].split("-")[0]

    if waveform == "3pn":
        from diffbank.noise import Sn_aLIGO as Sn
        from diffbank.utils import get_m1_m2_sampler
        from diffbank.waveforms.threePN_simple import Psi, amp

        m_range = (1.4, 5.0)
        sampler = get_m1_m2_sampler(m_range, m_range)
    else:
        raise ValueError(f"need to implement loading for '{waveform}'")

    return Bank.load(path, amp, Psi, Sn, sampler)


@click.command()
@click.option("--path", type=str, help="path to bank")
@click.option("--seed", type=int)
@click.option(
    "--n-eta", type=int, help="number of new points at which to compute effectualnesses"
)
def run(path, seed, n_eta):
    key = random.PRNGKey(seed)
    bank = load_bank(path)

    print(f"eta = {bank.eta_est:.4f} +/- {bank.eta_est_err:.4f}")

    # Append new effectualnesses to existing ones
    effectualness_points = bank.effectualness_points
    effectualnesses = bank.effectualnesses
    bank.calc_bank_effectualness(key, n_eta)
    bank.effectualness_points = jnp.concatenate(
        (effectualness_points, bank.effectualness_points)
    )
    bank.effectualnesses = jnp.concatenate((effectualnesses, bank.effectualnesses))
    # Recompute eta
    bank.eta_est = jnp.mean(bank.effectualnesses)
    bank.eta_est_err = jnp.std(bank.effectualnesses) / jnp.sqrt(
        len(bank.effectualnesses)
    )

    print(f"eta = {bank.eta_est:.4f} +/- {bank.eta_est_err:.4f}")

    bank.save("banks/")


if __name__ == "__main__":
    run()
