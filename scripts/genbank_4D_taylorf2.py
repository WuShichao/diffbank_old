import click
import numpy as np
import jax.numpy as jnp
import jax
import os

from diffbank.bank import Bank
from diffbank.waveforms import taylorF2
from jax import random
from typing import Callable


"""
Generate a TaylorF2 bank which can be compared with the BNS section of https://arxiv.org/abs/1904.01683
"""
# To generate results for paper use command
# python3 genbank_4D_taylorf2.py --seed 1 --kind random

##### Frequency settings
f_u = 512.0  # Hz
f_l = 24.0  # Hz
N_fbins = 4880
#####

m1_range = (1.0001, 3.0)
m2_range = (1.0, 3.0)

chi1_range = (-0.99, 0.99)
chi2_range = (-0.99, 0.99)


def get_Sn_O3a() -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Get interpolator for noise curve.
    """
    xp, yp = np.loadtxt("O3a_Livingston_ASD.txt", unpack=True)
    PSD = yp ** 2
    return lambda f: jnp.interp(f, xp, PSD, left=jnp.inf, right=jnp.inf)


def sampler(key, n):
    ms = random.uniform(
        key,
        minval=jnp.array([m1_range[0], m2_range[0]]),
        maxval=jnp.array([m1_range[1], m2_range[1]]),
        shape=(n, 2),
    )
    key, subkey = random.split(key)
    chi1s = random.uniform(
        subkey,
        minval=jnp.array(chi1_range[0]),
        maxval=jnp.array(chi1_range[1]),
        shape=(n, 1),
    )
    key, subkey = random.split(key)
    chi2s = random.uniform(
        subkey,
        minval=jnp.array(chi2_range[0]),
        maxval=jnp.array(chi2_range[1]),
        shape=(n, 1),
    )
    ms_correct = jnp.stack(
        [
            ms.max(axis=1),
            ms.min(axis=1),
        ]
    ).T
    return jnp.hstack((ms_correct, chi1s, chi2s))


@click.command()
@click.option("--seed", type=int, help="PRNG seed")
@click.option("--kind", type=str, help="Type of bank")
@click.option(
    "--n-eta",
    default=1000,
    type=int,
    help="number of new points at which to compute effectualnesses",
)
@click.option("--mm", default=0.95, help="minimum match")
@click.option("--eta-star", default=0.9, help="eta*")
@click.option("--n-eff", default=1000)
@click.option("--savedir", default="banks", help="directory in which to save the bank")
@click.option("--device", default="cpu", help="device to run on")
def gen_4D_taylorf2bank(seed, kind, n_eta, mm, eta_star, n_eff, savedir, device):
    jax.config.update("jax_platform_name", device)

    key = random.PRNGKey(seed)
    fs = jnp.linspace(f_l, f_u, N_fbins)
    Sn = get_Sn_O3a()

    bank = Bank(
        taylorF2.Amp,
        taylorF2.Psi,
        fs,
        Sn,
        1 - mm,
        eta_star,
        sampler,
        name=f"tf2-{kind}-{seed}-mm={mm}-eta_star={eta_star}-n_eff={n_eff}",
    )

    theta_max = jnp.array([m1_range[0], m2_range[0], chi1_range[0], chi2_range[1]])
    bank.ratio_max = bank.density_fun(theta_max)

    key, subkey = random.split(key)
    bank.fill_bank(subkey, kind, n_eff)
    bank.save(savedir)
    print(f"Saved bank to {os.path.join(savedir, bank.name + '.npz')}")

    key, subkey = random.split(key)
    bank.calc_bank_effectualness(subkey, n_eta)

    bank.save(savedir)


if __name__ == "__main__":
    gen_4D_taylorf2bank()
