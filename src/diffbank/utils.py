from math import pi
from typing import Callable

import jax
from jax import random
import jax.numpy as jnp
from jax.scipy.special import gammaln

from .constants import C, G


def ms_to_Mc_eta(m):
    m1, m2 = m
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5), m1 * m2 / (m1 + m2) ** 2


def get_f_isco(m):
    return 1 / (6 ** (3 / 2) * pi * m / (C ** 3 / G))


def get_m1_m2_sampler(m1_range, m2_range) -> Callable[[jnp.ndarray, int], jnp.ndarray]:
    """
    Uniformly samples over the specified ranges, with the restriction that the
    first variable is larger than the second.
    """

    def sampler(key, n):
        ms = random.uniform(
            key,
            minval=jnp.array([m1_range[0], m2_range[0]]),
            maxval=jnp.array([m1_range[1], m2_range[1]]),
            shape=(n, 2),
        )
        return jnp.stack([ms.max(axis=1), ms.min(axis=1)]).T

    return sampler


def Sn_func(f):
    """
    Noise function for aLIGO?

    TODO: fix!
    """
    fs = 40  # Hz
    f_theta = 150  # Hz
    x = f / f_theta
    normalization = 1e-46
    return jnp.where(
        f > fs,
        normalization
        * 9
        * ((4.49 * x) ** (-56) + 0.16 * x ** (-4.52) + 0.52 + 0.32 * x ** 2),
        jnp.inf,
    )


def get_effectualness(theta1, theta2, amp, Psi, f, Sn):
    """
    Calculates the effectualness between two waveforms at theta1 and theta2.

    Assumes f's entries are linearly spaced!
    """
    Sns = Sn(f)
    df = f[1] - f[0]

    # Calculating the best fit tc
    wf1 = amp(f, theta1) * jnp.exp(1j * Psi(f, theta1))
    wf2 = amp(f, theta2) * jnp.exp(1j * Psi(f, theta2))

    norm1 = jnp.sqrt(4.0 * jnp.sum((wf1 * wf1.conj() / Sns) * df).real)
    norm2 = jnp.sqrt(4.0 * jnp.sum((wf2 * wf2.conj() / Sns) * df).real)

    norm = norm1 * norm2

    overlap_tc = jnp.fft.fft((4.0 * wf1 * wf2.conj() * df) / Sns / norm)

    return jnp.abs(overlap_tc).max()


def get_sphere_vol(n) -> jnp.ndarray:
    """
    Volume of n-sphere.
    """
    return pi ** (n / 2) / jnp.exp(gammaln((n / 2) + 1))


def get_vol_from_samples(
    naive_vol, thetas, density_fun: Callable[[jnp.ndarray], jnp.ndarray]
) -> jnp.ndarray:
    """
    Computes parameter space volume using MC integration given some samples.
    """
    densities = jax.lax.map(density_fun, thetas)
    return jnp.mean(densities) * naive_vol


def get_vol(
    key,
    naive_vol,
    n_samples,
    density_fun: Callable[[jnp.ndarray], jnp.ndarray],
    sampler: Callable[[jnp.ndarray, int], jnp.ndarray],
) -> jnp.ndarray:
    """
    Computes parameter space volume using MC integration.
    """
    thetas = sampler(key, n_samples)
    return get_vol_from_samples(naive_vol, thetas, density_fun)


def get_n_templates(
    key,
    naive_vol,
    n_samples,
    density_fun: Callable[[jnp.ndarray], jnp.ndarray],
    sampler: Callable[[jnp.ndarray, int], jnp.ndarray],
    eta,
    m_star,
) -> jnp.ndarray:
    """
    This function computes the number of templates for a specific waveform
    model specified by the phase and amplitude

    The waveform models dimension is inferred from prange which should be nx2 array

    To do this we just need to compute equation 15 from
    https://arxiv.org/pdf/0809.5223.pdf
    """
    dim = sampler(key, 1).shape[-1]  # fine to reuse key here!
    vol_space = get_vol(key, naive_vol, n_samples, density_fun, sampler)
    vol_sphere = get_sphere_vol(dim)
    return jnp.ceil(
        jnp.log(1 - eta) / jnp.log(1 - m_star ** (dim / 2) * vol_sphere / vol_space)
    ).astype(int)


def _gen_template_rejection(
    key,
    density_max,
    density_fun: Callable[[jnp.ndarray], jnp.ndarray],
    sampler: Callable[[jnp.ndarray, int], jnp.ndarray],
) -> jnp.ndarray:
    """
    Generates a single template using rejection sampling.
    """

    def cond_fun(val):
        theta, u_key = val[0], val[2]
        u = random.uniform(u_key)
        return u >= density_fun(theta) / density_max

    def body_fun(val):
        theta_key = val[1]
        _, theta_key, u_key = random.split(theta_key, 3)
        theta = sampler(theta_key, 1)[0]
        return (theta, theta_key, u_key)

    init_val = body_fun((key, key))  # only second element of init_val matters

    return jax.lax.while_loop(cond_fun, body_fun, init_val)[0]


def gen_templates_rejection(
    key,
    density_max,
    n_templates,
    density_fun: Callable[[jnp.ndarray], jnp.ndarray],
    sampler,
) -> jnp.ndarray:
    """
    Generates a bank with n_templates samples using rejection sampling.

    TODO: add tqdm somehow.
    """
    keys = random.split(key, n_templates + 1)[1:]
    f = lambda key: _gen_template_rejection(key, density_max, density_fun, sampler)
    return jax.lax.map(f, keys)
