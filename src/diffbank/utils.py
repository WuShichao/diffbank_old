from math import pi
from typing import Callable, Optional, Tuple

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


# def update_fields(nt: NamedTuple, **updates):
#     """
#     Returns a new named tuple with the specified fields updated.
#     """
#     d = nt._asdict()
#     for k in updates:
#         d.pop(k)
#     return type(nt)(**updates, **d)
#
#


def get_M_eta_sampler(M_range, eta_range) -> Callable[[jnp.ndarray, int], jnp.ndarray]:
    """
    Uniformly samples over the specified ranges.
    """

    def sampler(key, n):
        M_eta = random.uniform(
            key,
            minval=jnp.array([M_range[0], eta_range[0]]),
            maxval=jnp.array([M_range[1], eta_range[1]]),
            shape=(n, 2),
        )
        return M_eta

    return sampler


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


def get_n_templates(
    key,
    naive_vol,
    n_samples,
    density_fun: Callable[[jnp.ndarray], jnp.ndarray],
    sampler: Callable[[jnp.ndarray, int], jnp.ndarray],
    eta,
    m_star,
    frac_in_bounds: jnp.ndarray = jnp.array(1.0),
    frac_in_bounds_err: jnp.ndarray = jnp.array(0.0),
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Estimates number of templates required to cover a parameter space with a
    random bank.

    Args:
      naive_vol: parameter space volume ignoring the metric. For example, a
        waveform parametrized by (x, y) with 0 < x < x_max and 0 < y < x would
        have naive_vol = 1/2 * x_max**2 regardless of the metric.

    Returns:
      MC estimate (and error) for required number of templates.


    Reference:
      Eq. 14 in https://arxiv.org/abs/0809.5223
    """
    dim = sampler(key, 1).shape[-1]  # fine to reuse key here!

    # Uncorrected template ellipsoid volume
    vol_template = m_star ** (dim / 2) * get_sphere_vol(dim)

    # MC samples of number of templates
    thetas = sampler(key, n_samples)
    space_vols = naive_vol * jax.lax.map(density_fun, thetas)
    vol_ratios = vol_template / space_vols
    ns = jnp.log(1 - eta) / jnp.log(1 - frac_in_bounds * vol_ratios)
    n_mean = jnp.mean(ns).astype(int)
    n_err_vol = jnp.std(ns) / jnp.sqrt(n_samples)

    # Propagate the error on the fraction of the average template's volume that
    # lies in bounds
    vol_ratio_mean = jnp.mean(vol_ratios)
    n_err_ib = (
        frac_in_bounds_err
        * vol_ratio_mean
        * jnp.log(1 - eta)
        / (
            (1 - frac_in_bounds * vol_ratio_mean)
            * jnp.log(1 - frac_in_bounds * vol_ratio_mean) ** 2
        )
    )

    return n_mean, jnp.sqrt(n_err_vol ** 2 + n_err_ib ** 2)


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
        theta, u_key = val[0], val[3]
        u = random.uniform(u_key)
        return u >= density_fun(theta) / density_max

    def body_fun(val):
        key = val[1]
        key, theta_key, u_key = random.split(key, 3)
        theta = sampler(theta_key, 1)[0]
        return (theta, key, theta_key, u_key)  # new val

    # Only second element of init_val matters
    init_val = body_fun((None, key, None, None))

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

    TODO: add tqdm somehow?
    """
    keys = random.split(key, n_templates)
    f = lambda key: _gen_template_rejection(key, density_max, density_fun, sampler)
    return jax.lax.map(f, keys)


def sample_uniform_ball(key, dim, shape: Tuple[int] = (1,)) -> jnp.ndarray:
    """
    Uniformly sample from the unit ball.
    """
    xs = random.normal(key, shape + (dim,))
    abs_xs = jnp.sqrt(jnp.sum(xs ** 2, axis=-1, keepdims=True))
    sphere_samples = xs / abs_xs
    rs = random.uniform(key, shape + (1,)) ** (1 / dim)
    return sphere_samples * rs


def sample_uniform_metric_ellipse(
    key: jnp.ndarray,
    g: jnp.ndarray,
    # theta: jnp.ndarray,
    # get_g: Callable[[jnp.ndarray], jnp.ndarray],
    # m_star,
    n: int,
) -> jnp.ndarray:
    """
    Uniformly sample inside a metric ellipse centered at the origin.
    """
    dim = g.shape[1]
    # radius = jnp.sqrt(m_star)
    ball_samples = sample_uniform_ball(key, dim, (n,))
    trafo = jnp.linalg.inv(jnp.linalg.cholesky(g))
    return ball_samples @ trafo.T


def get_template_frac_in_bounds(
    key: jnp.ndarray,
    thetas: jnp.ndarray,
    get_g: Callable[[jnp.ndarray], jnp.ndarray],
    m_star,
    is_in_bounds: Callable[[jnp.ndarray], jnp.ndarray],
    n: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Estimates average fraction of a template's metric ellipse lying inside the
    parameter space.

    Args:
      is_in_bounds: callable that takes a point and returns 1 if it is in the
        parameter space and 0 if not.
      n: number of points to sample in each template ellipse.

    Returns:
      MC estimate (along with error) of the fraction of the volume of the
      templates centered on ``theta`` that lies in the parameter space.
    """

    def helper(x):
        """
        Rescale metric ellipse samples to have radius ``sqrt(m_star)`` and
        recenter on ``theta``.
        """
        key = x[0].astype(jnp.uint32)
        theta = x[1]
        ellipse_samples_0 = sample_uniform_metric_ellipse(key, get_g(theta), n)
        return jnp.sqrt(m_star) * ellipse_samples_0 + theta

    keys = random.split(key, len(thetas))
    dim = thetas.shape[1]
    ellipse_samples = jax.lax.map(helper, jnp.stack([keys, thetas], axis=1)).reshape(
        [-1, dim]
    )
    in_bounds = jax.lax.map(is_in_bounds, ellipse_samples)
    return in_bounds.mean(), in_bounds.std() / jnp.sqrt(len(thetas) * n)
