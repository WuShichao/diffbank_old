import click
from typing import Callable
import numpy as np

from diffbank.bank import Bank
from jax import random

import jax
import jax.numpy as jnp
from jax import jit
from diffbank.waveforms import kappa6D
from diffbank.metric import get_g

"""
Tests consistency of random template bank
"""
f_u = 512.0  # Hz
f_l = 32.0  # Hz

Mt_range = (2, 6)
eta_range = (0.1875, 0.2499)
chi1_range = (0.0, 0.2)
chi2_range = (0.0, 0.2)
k1_range = (1.0, 4.0)
k2_range = (1.0, 4.0)


def get_Sn_aLIGO() -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Get interpolator for noise curve.
    """
    xp, yp = np.loadtxt("LIGO-P1200087-v18-aLIGO_MID_LOW.txt", unpack=True)
    return lambda f: jnp.interp(f, xp, yp, left=jnp.inf, right=jnp.inf)


def sampler(key, n):
    """
    Proposal distribution for var rejection sampling.
    """
    return random.uniform(
        key,
        shape=(
            n,
            6,
        ),
        minval=jnp.stack(
            (
                Mt_range[0],
                eta_range[0],
                chi1_range[0],
                chi2_range[0],
                k1_range[0],
                k2_range[0],
            )
        ),
        maxval=jnp.stack(
            (
                Mt_range[1],
                eta_range[1],
                chi1_range[1],
                chi2_range[1],
                k1_range[1],
                k2_range[1],
            )
        ),
    )


def test_6D_metric(key):
    """
    Returns a filled bank.
    """

    # testing
    fs = jnp.linspace(f_l, f_u, 10000)
    Sn_aLIGO = get_Sn_aLIGO()
    mm = 0.95
    m_star = 1 - mm
    eta = 0.9
    bank = Bank(
        kappa6D.Amp,
        kappa6D.Psi,
        fs,
        Sn_aLIGO,
        m_star,
        eta,
        sampler,
        "6D",
    )

    N = 40
    thetas = sampler(key, N)

    def map_test(f, xs):
        return jnp.stack([f(x) for x in xs])

    def my_get_density(g):
        return jnp.sqrt(jnp.linalg.det(g))

    # print(jax.make_jaxpr(my_get_density)(thetas[0]))
    gs = map_test(bank.g_fun, thetas)
    densities_explicit = jax.lax.map(my_get_density, gs)
    densities_internal = jax.lax.map(bank.density_fun, thetas)
    print(densities_explicit)
    print(densities_internal)

    print(jnp.allclose(densities_explicit, densities_internal))

    return False


@click.command()
@click.option("--seed", type=int, help="PRNG seed")
def main(seed):
    key = random.PRNGKey(seed)
    test_6D_metric(key)


if __name__ == "__main__":
    main()
