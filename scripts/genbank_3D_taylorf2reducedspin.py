import click
import jax
import jax.numpy as jnp
import numpy as np

from scipy.optimize import minimize_scalar
from diffbank.bank import Bank
from diffbank.constants import C, G, MSUN
from diffbank.waveforms.taylorf2reducedspin import Psi, amp, get_th_boundary_interps
from jax import random
from math import pi
from typing import Callable


"""
Generate a TaylorF2ReducedSpin bank for comparison with Ajith et al 2014,
https://arxiv.org/abs/1210.6666.
"""
# To generate results for paper use command
# python3 genbank_3D_taylorf2reducedspin.py --seed 1 --kind random

##### Frequency settings
f_u = 512.0  # Hz
f_l = 24.0  # Hz
N_fbins = 4880
#####
f_0 = 20.0  # Hz

m_range = (1 * MSUN, 20 * MSUN)
m_ns_thresh = 2 * MSUN
M_tot_max = m_range[0] + m_range[1]
chi_bh_max = 0.98
chi_ns_max = 0.4

th0_range, th3_interp_low, th3_interp_high = get_th_boundary_interps(*m_range, f_0)
# Figure out where th3 attains its maximum
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
    chi1_max = jnp.where(m1 > m_ns_thresh, chi_bh_max, chi_ns_max)
    chi2_max = jnp.where(m2 > m_ns_thresh, chi_bh_max, chi_ns_max)
    chi_s_max = (chi1_max + chi2_max) / 2
    chi_a_max = (chi1_max - chi2_max) / 2
    chi_max = chi_s_max * (1 - 76 * eta / 113) + delta * chi_a_max
    th3S_max = 113 * th3 * chi_max / (48 * pi)
    return th3S_max


bracket = (th0_range[0], 5e3)  # NOTE: need to change if m_range changes!
res = minimize_scalar(lambda th0: -th3_interp_high(th0), bracket, bracket)
assert res.success
th0_th3_max = res.x
th3_max = -res.fun
th3_range = (th3_interp_low(th0_range[0]), th3_max)
# Maximum value of th3
th3S_max = get_th3S_max(th0_th3_max, th3_max)


def get_Sn_O3a() -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Get interpolator for noise curve.
    """
    xp, yp = np.loadtxt("O3a_Livingston_ASD.txt", unpack=True)
    PSD = yp ** 2
    return lambda f: jnp.interp(f, xp, PSD, left=jnp.inf, right=jnp.inf)


def get_M_tot(th0, th3):
    M_chirp = 1 / (16 * pi * f_0) * (125 / (2 * th0 ** 3)) ** (1 / 5) * C ** 3 / G
    eta = (16 * pi ** 5 / 25 * th0 ** 2 / th3 ** 5) ** (1 / 3)
    q = (1 + jnp.sqrt(1 - 4 * eta) - 2 * eta) / (2 * eta)
    m2 = (1 + q) ** (1 / 5) / q ** (3 / 5) * M_chirp
    m1 = q * m2
    return m1 + m2


def is_in_bounds(theta: jnp.ndarray) -> jnp.ndarray:
    """
    Checks if a point is in bounds using the `th` values and total mass.
    """
    th0, th3, th3S = theta[..., 0], theta[..., 1], theta[..., 2]
    return jnp.logical_and(
        th3 > th3_interp_low(th0),
        jnp.logical_and(
            th3 < th3_interp_high(th0),
            jnp.logical_and(
                jnp.abs(th3S) < get_th3S_max(th0, th3), get_M_tot(th0, th3) < M_tot_max
            ),
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
def gen_2D_tf2rs(seed, kind, n_eta, mm, eta_star, n_eff, savedir, device):
    jax.config.update("jax_platform_name", device)

    key = random.PRNGKey(seed)
    m_star = 1 - mm
    fs = jnp.linspace(f_l, f_u, N_fbins)
    Sn = get_Sn_O3a()

    bank = Bank(
        amp,
        Psi,
        fs,
        Sn,
        m_star,
        eta_star,
        sampler,
        name=f"tf2rs-{kind}-{seed}-mm={mm}-eta_star={eta_star}-n_eff={n_eff}",
    )

    # Get max density
    # NOTE: need to change if m_range changes!
    th0s = jnp.linspace(1.0001 * th0_range[0], 0.9999 * th0_range[1], 500)
    th3s = th3_interp_high(th0s) * 0.99999
    th3Ss = -get_th3S_max(th0s, th3s)
    boundary_densities = jax.lax.map(
        bank.density_fun, jnp.stack([th0s, th3s, th3Ss], -1)
    )
    bank.ratio_max = jnp.nanmax(boundary_densities)

    # Fill bank
    key, subkey = random.split(key)
    bank.fill_bank(subkey, kind, n_eff)
    bank.save(savedir)
    print(f"Saved bank to {os.path.join(savedir, bank.name + '.npz')}")

    # Get effectualnesses
    key, subkey = random.split(key)
    bank.calc_bank_effectualness(subkey, n_eta)
    bank.save(savedir)


if __name__ == "__main__":
    gen_2D_tf2rs()
