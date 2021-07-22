from contextlib import nullcontext
from math import pi
from math import sqrt
from typing import Callable, Optional, Tuple
import warnings

from tqdm.auto import tqdm

import jax
from jax import random
import jax.numpy as jnp

from .constants import C, G
from .metric import get_density, get_g


def n_eff_pts(eta, r: float = 1):
    """
    Gets number of `eff_pts` to use during bank generation. This is computed by
    setting

        p(n_eff_pts | > eta) / p(n_eff_pts | < eta) > r,

    where the likelihoods are those for n_eff_pts being within a minimum match
    threshold of a template.
    """
    return int(jnp.ceil(jnp.log(1 / (1 + r)) / jnp.log(eta) - 1))


def ms_to_Mc_eta(m):
    m1, m2 = m
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5), m1 * m2 / (m1 + m2) ** 2


def get_f_isco(m):
    return 1 / (6 ** (3 / 2) * pi * m / (C ** 3 / G))


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


def get_match(theta1, theta2, amp, Psi, fs, Sn):
    """
    Calculates the match between two waveforms at theta1 and theta2.

    Assumes fs's entries are linearly spaced!
    """
    Sns = Sn(fs)
    df = fs[1] - fs[0]

    # Calculating the best fit tc
    wf1 = amp(fs, theta1) * jnp.exp(1j * Psi(fs, theta1))
    wf2 = amp(fs, theta2) * jnp.exp(1j * Psi(fs, theta2))

    norm1 = jnp.sqrt(4.0 * jnp.sum((wf1 * wf1.conj() / Sns) * df).real)
    norm2 = jnp.sqrt(4.0 * jnp.sum((wf2 * wf2.conj() / Sns) * df).real)

    norm = norm1 * norm2

    overlap_tc = jnp.fft.fft((4.0 * wf1 * wf2.conj() * df) / Sns / norm)

    return jnp.abs(overlap_tc).max()


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
    theta: jnp.ndarray,
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
    # Rescale metric ellipse samples to have radius ``sqrt(m_star)`` and
    # recenter on ``theta``
    ellipse_samples_0 = sample_uniform_metric_ellipse(key, get_g(theta), n)
    ellipse_samples = jnp.sqrt(m_star) * ellipse_samples_0 + theta
    in_bounds = jnp.concatenate((jnp.array([1.]), jax.lax.map(is_in_bounds, ellipse_samples)))
    return in_bounds.mean(), in_bounds.std() / jnp.sqrt(n + 1)


def gen_template_rejection(
    key,
    density_max,
    density_fun: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    base_dist: Callable[[jnp.ndarray, int], jnp.ndarray],
) -> jnp.ndarray:
    """
    Generates a single template using rejection sampling.
    """

    def cond_fun(val):
        theta, cond_key = val[0], val[3]
        u_key, density_key = random.split(cond_key)
        u = random.uniform(u_key)
        return u >= density_fun(density_key, theta) / density_max

    def body_fun(val):
        key = val[1]
        key, theta_key, cond_key = random.split(key, 3)
        theta = base_dist(theta_key, 1)[0]
        return (theta, key, theta_key, cond_key)  # new val

    # Only second element of init_val matters
    init_val = body_fun((None, key, None, None))

    return jax.lax.while_loop(cond_fun, body_fun, init_val)[0]


def gen_templates_rejection(
    key,
    density_max,
    n_templates,
    density_fun: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    sampler,
) -> jnp.ndarray:
    """
    Generates a bank with n_templates samples using rejection sampling.
    TODO: add tqdm somehow?
    """
    keys = random.split(key, n_templates)
    f = lambda key: gen_template_rejection(key, density_max, density_fun, sampler)
    return jax.lax.map(f, keys)


def _update_uncovered_eff(pt, eff, template, minimum_match, amp, Psi, fs, Sn):
    """
    Computes match for point not covered by a template
    """
    uncovered = eff < minimum_match

    def true_fun(pt):
        return get_match(template, pt, amp, Psi, fs, Sn)

    def false_fun(_):
        return eff

    return jax.lax.cond(uncovered, true_fun, false_fun, pt)


def gen_bank(
    key: jnp.ndarray,
    density_max: jnp.ndarray,
    base_dist: Callable[[jnp.ndarray, int], jnp.ndarray],
    amp,
    Psi,
    fs,
    Sn: Callable[[jnp.ndarray], jnp.ndarray],
    minimum_match: int,
    eta: float,
    show_progress: bool = True,
    eff_pt_sampler: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    r: float = 1,
    is_in_bounds: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    n_fib: int = 1000,
) -> jnp.ndarray:
    """
    Arguments
    - density_max: maximum value of density function. This is either sqrt(|g|)
      or sqrt(|g|) / f_ib, depending on whether `is_in_bounds` is provided.
    - r: likelihood ratio p(> eta | n_eff_pts) / p(< eta | n_eff_pts) for
      setting the number of effectualness points.
    - is_in_bounds: function returning 1 if the argument is in bounds and 0
      otherwise. If `None`, the fraction of points in each metric ellipse will
      not be used to reweight the template density.
    - n_fib: number of points used to estimate fraction of each template in
      bounds.

    TODO: jax-ify
    """
    if is_in_bounds is not None:
        warnings.warn("using f_ib probably won't improve your bank")

        if n_fib <= 0: raise ValueError("n_fib must be positive")

        _get_g = lambda theta: get_g(theta, amp, Psi, fs, Sn)
        density_fun = (
            lambda key, theta: get_density(theta, amp, Psi, fs, Sn)
            / get_template_frac_in_bounds(
                key, theta, _get_g, 1 - minimum_match, is_in_bounds, n_fib
            )[0]
        )
    else:
        density_fun = lambda _, theta: get_density(theta, amp, Psi, fs, Sn)

    gen_template = jax.jit(
        lambda key: gen_template_rejection(key, density_max, density_fun, base_dist)
    )

    if eff_pt_sampler is None: eff_pt_sampler = gen_template

    n_eff = n_eff_pts(eta, r)
    keys = random.split(key, 1 + n_eff)
    key_bank, keys_eff = keys[0], keys[1:]

    # Generate points for effectualness monitoring
    eff_pts = jax.lax.map(eff_pt_sampler, keys_eff)
    effs = jnp.zeros(n_eff)

    # Compile in the eff_pts
    @jax.jit
    def update_uncovered_effs(template, effs):
        return jax.lax.map(
            lambda ep: _update_uncovered_eff(
                ep["point"], ep["eff"], template, minimum_match, amp, Psi, fs, Sn
            ),
            {"point": eff_pts, "eff": effs},
        )

    # Fill the bank!
    templates = []
    n_covered = 0
    progress = tqdm(total=int(n_eff)) if show_progress else nullcontext()
    with progress as pbar:
        while n_covered < n_eff:
            # Make template
            key_bank, key_template = random.split(key_bank)
            template = gen_template(key_template)
            templates.append(template)

            # Compute matches
            effs = update_uncovered_effs(template, effs)

            # Update coverage count
            dn_covered = (effs > minimum_match).sum() - n_covered
            n_covered += dn_covered

            if show_progress:  # pbar is a tqdm
                pbar.update(int(dn_covered))  # type: ignore
                pbar.set_description(f"{len(templates)} templates")  # type: ignore

    return jnp.array(templates), eff_pts


def get_bank_effectualness(
    key: jnp.ndarray,
    templates,
    amp,
    Psi,
    fs,
    Sn: Callable[[jnp.ndarray], jnp.ndarray],
    minimum_match: int,
    n: int = 100,
    show_progress: bool = True,
    eff_pt_sampler: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    base_dist: Optional[Callable[[jnp.ndarray, int], jnp.ndarray]] = None,
    density_max: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    TODO: jax-ify
    """
    # Compile in the templates and waveform model
    @jax.jit
    def get_bank_eff(pt):
        return jax.lax.map(
            lambda template: get_match(template, pt, amp, Psi, fs, Sn), templates
        ).max()

    # Set up sampler
    if eff_pt_sampler is None:
        if base_dist is None or density_max is None:
            raise ValueError(
                "must provide base_dist and density_max to sample points with "
                "density sqrt(|g|)"
            )
        density_fun = lambda theta: get_density(theta, amp, Psi, fs, Sn)
        eff_pt_sampler = jax.jit(
            lambda key: gen_template_rejection(key, density_max, density_fun, base_dist)  # type: ignore
        )
    elif base_dist is not None:
        warnings.warn("base_dist will be ignored since eff_pt_sampler was provided")
    elif density_max is not None:
        warnings.warn("density_max will be ignored since eff_pt_sampler was provided")

    # Sample points and compute effectualnesses
    keys = random.split(key, n)
    eff_pts = jax.lax.map(eff_pt_sampler, keys)
    effs = []
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    eta, eta_err, M_2 = 0.0, 0.0, 0.0
    progress = tqdm(eff_pts) if show_progress else nullcontext()
    with progress as pbar:
        for n, pt in enumerate(pbar, start=1):  # type: ignore
            effs.append(get_bank_eff(pt))

            x = effs[-1] > minimum_match
            eta_prev = eta
            eta = eta_prev + (x - eta_prev) / n
            M_2 = M_2 + (x - eta_prev) * (x - eta)
            if n > 1:
                eta_err = jnp.sqrt(M_2 / (n - 1))

            if show_progress:  # pbar is a tqdm
                pbar.set_description(f"eta = {eta:.4f} +/- {eta_err:.4f}")  # type: ignore

    return jnp.array(effs), eff_pts, eta, eta_err
