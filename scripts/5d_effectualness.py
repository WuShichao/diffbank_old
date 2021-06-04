from math import pi

import click
import matplotlib.pyplot as plt

from diffbank.bank import Bank
from diffbank.constants import C, G, MSUN
from diffbank.utils import ms_to_Mc_eta
from diffbank.utils import gen_templates_rejection
from jax import random
import jax
import jax.numpy as jnp
from diffbank.waveforms import HS5d
from diffbank.metric import get_g

"""
Tests consistency of random template bank
"""
f_0 = 512.0 # Hz
f_s = 20.0  # Hz
S_0 = 1e-46  # arbitrary

m_range = (1 , 5)
#FIX ME
chiz_range=(-.8,.8)
k2_range=(1,1200)
k3_range=(1,300)


def propose(key, n):
    """
    Proposal distribution for var rejection sampling.
    """
    return random.uniform(
        key,
        shape=(
            n,
            5,
        ),
        minval=jnp.stack((m_range[0], m_range[0], chiz_range[0], k2_range[0], k3_range[0])),
        maxval=jnp.stack((m_range[1], m_range[1], chiz_range[1], k2_range[1], k3_range[1])),
    )


def accept(vars: jnp.ndarray) -> jnp.ndarray:
    """
    Returns 1 if vars is in bounds, 0 otherwise.
    """
    var1, var2, var3, var4, var5 = vars[..., 0], vars[..., 1],  vars[..., 2],  vars[..., 3],  vars[..., 4]
    var1_clipped = jnp.clip(var1, m_range[0], m_range[1])
    var2_clipped = jnp.clip(var2, m_range[0], m_range[1])
    var3_clipped = jnp.clip(var3, chiz_range[0], chiz_range[1])
    var4_clipped = jnp.clip(var4, k2_range[0], k2_range[1])
    var5_clipped = jnp.clip(var5, k3_range[0], k3_range[1])
    # Check if clipped and unclipped point are equal
    return jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_and(var1 == var1_clipped, var2 == var2_clipped), var3 == var3_clipped), var4 == var4_clipped), var5 == var5_clipped).astype(
        jnp.float64
    )


def var_sampler(key, n):
    """
    Rejection sampler for var.
    """
    return gen_templates_rejection(key, jnp.array(1.0), n, accept, propose)


def Sn_LIGO(f):
    """
    LIGO noise curve.
    """
    #load actual list, jnp.interpolate, make sure it is asd/psd
    return jnp.where(
        f > f_s, 1 / 5 * S_0 * ((f / f_0) ** (-4) + 2 * (1 + (f / f_0) ** 2)), jnp.inf
    )


def get_bank(key):
    """
    Returns a filled bank.
    """
    naive_vol_key, frac_ib_key, n_templates_key, fill_key = random.split(key, 4)

    # Estimate naive var volume to within 10%
    proposals = propose(naive_vol_key, 1000)
    proposal_vol = (m_range[1] - m_range[0]) * (m_range[1] - m_range[0]) *  (k2_range[1] - k2_range[0]) * (k3_range[1] - k3_range[0]) *  (chiz_range[1] - chiz_range[0]) 
    in_bounds = accept(proposals)
    naive_vol = in_bounds.mean() * proposal_vol
    naive_vol_err = in_bounds.std() * proposal_vol / jnp.sqrt(len(proposals))
    print(f"Naive parameter space volume: {naive_vol:.6f} +/- {naive_vol_err:.6f}")
    assert naive_vol_err / naive_vol < 0.1

    # Configure bank
    fs = jnp.linspace(f_s, 10000, 3000)
    mm = 0.95
    m_star = 1 - mm
    eta = 0.95
    bank = Bank(
        HS5d.amp, HS5d.Psi, fs, Sn_LIGO, var_sampler, naive_vol, m_star, eta, accept, "owen"
    )

    #testing
    print(var_sampler(naive_vol_key, 1))
    print(get_g(var_sampler(naive_vol_key,1)[0], HS5d.amp, HS5d.Psi, fs, Sn_LIGO))
    bank.density_max = bank.get_density(var_sampler(naive_vol_key, 1)[0])
    print(bank.get_density(var_sampler(naive_vol_key, 1)[0]))

    bank.compute_template_frac_in_bounds(frac_ib_key, 10000, 20)
    print(
        f"{bank.frac_in_bounds * 100:.3f} +/- {bank.frac_in_bounds_err * 100:.3f} % "
        "of the average template ellipse's volume is in bounds"
    )
    bank.compute_n_templates(n_templates_key, 1000)
    print(f"{bank.n_templates} +/- {bank.n_templates_err:.2f} templates required")
    # print("Filling the bank")
    # bank.fill_bank(fill_key)
    # print(f"Done: {bank}")

    return bank


@click.command()
@click.option("--seed", type=int, help="PRNG seed")
def main(seed):
    key = random.PRNGKey(seed)
    bank_key, eff_key = random.split(key)

    bank = get_bank(bank_key)

    # plot_bank(bank, seed)
    # plot_effectualnesses(eff_key, bank, seed)


if __name__ == "__main__":
    main()
