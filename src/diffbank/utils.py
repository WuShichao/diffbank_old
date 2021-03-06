from contextlib import nullcontext
from math import pi
from typing import Callable, Tuple, Union

import jax
from jax import random
# from jax._src.prng import PRNGKeyArray
import jax.numpy as jnp
from tqdm.auto import tqdm, trange

from .constants import C, G


Array = jnp.ndarray
PRNGKeyArray = Array


def ms_to_Mc_eta(m):
    m1, m2 = m
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5), m1 * m2 / (m1 + m2) ** 2


def get_f_isco(m):
    return 1 / (6 ** (3 / 2) * pi * m / (C ** 3 / G))


def get_M_eta_sampler(M_range, eta_range) -> Callable[[PRNGKeyArray, int], Array]:
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


def get_m1_m2_sampler(m1_range, m2_range) -> Callable[[PRNGKeyArray, int], Array]:
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


def get_eff_pads(fs: Array) -> Tuple[Array, Array]:
    """
    Returns padding arrays required for accurate effectualness calculation.
    Padding `fs` with the returned arrays (almost) doubles its size and extends
    it down to 0.
    """
    df = fs[1] - fs[0]
    N = 2 * jnp.array(fs[-1] / df - 1).astype(int)
    pad_low = jnp.zeros(jnp.array(fs[0] / df).astype(int))
    pad_high = jnp.zeros(N - jnp.array(fs[-1] / df).astype(int))
    return pad_low, pad_high


def _get_effectualness(
    theta1: Array,
    theta2: Array,
    amp: Callable[[Array, Array], Array],
    Psi: Callable[[Array, Array], Array],
    fs: Array,
    Sn: Callable[[Array], Array],
    pad_low: Array,
    pad_high: Array,
):
    """
    Calculates the match between two waveforms at theta1 and theta2.

    Assumes fs's entries are linearly spaced!
    """
    wf1 = amp(fs, theta1) * jnp.exp(1j * Psi(fs, theta1))
    wf2 = amp(fs, theta2) * jnp.exp(1j * Psi(fs, theta2))
    Sns = Sn(fs)

    # Factors of 4 and df drop out due to linearity
    norm1 = jnp.sqrt(jnp.sum(jnp.abs(wf1) ** 2 / Sns))
    norm2 = jnp.sqrt(jnp.sum(jnp.abs(wf2) ** 2 / Sns))

    # Use IFFT trick to maximize over t_c. Ref: Maggiore's book, eq. 7.171.
    integrand_padded = jnp.concatenate((pad_low, wf1.conj() * wf2 / Sns, pad_high))
    # print(low_padding, high_padding, len(fs), N)
    return jnp.abs(len(integrand_padded) * jnp.fft.ifft(integrand_padded)).max() / (
        norm1 * norm2
    )


def sample_uniform_ball(key: PRNGKeyArray, dim: int, shape: Tuple[int] = (1,)) -> Array:
    """
    Uniformly sample from the unit ball.
    """
    xs = random.normal(key, shape + (dim,))
    abs_xs = jnp.sqrt(jnp.sum(xs ** 2, axis=-1, keepdims=True))
    sphere_samples = xs / abs_xs
    rs = random.uniform(key, shape + (1,)) ** (1 / dim)
    return sphere_samples * rs


def sample_uniform_metric_ellipse(
    key: PRNGKeyArray,
    g: Array,
    n: int,
) -> Array:
    """
    Uniformly sample inside a metric ellipse centered at the origin.
    """
    dim = g.shape[1]
    # radius = jnp.sqrt(m_star)
    ball_samples = sample_uniform_ball(key, dim, (n,))
    trafo = jnp.linalg.inv(jnp.linalg.cholesky(g))
    return ball_samples @ trafo.T


def get_template_frac_in_bounds(
    key: PRNGKeyArray,
    theta: Array,
    get_g: Callable[[Array], Array],
    m_star,
    is_in_bounds: Callable[[Array], Array],
    n: int,
) -> Tuple[Array, Array]:
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
    in_bounds = jnp.concatenate(
        (jnp.array([1.0]), jax.lax.map(is_in_bounds, ellipse_samples))
    )
    return in_bounds.mean(), in_bounds.std() / jnp.sqrt(n + 1)


def est_ratio_max(
    key: PRNGKeyArray,
    density_fun: Callable[[Array], Array],
    sample_base: Callable[[PRNGKeyArray, int], Array],
    density_fun_base: Callable[[Array], Array] = lambda _: jnp.array(1.0),
    n_iter: int = 1000,
    n_init: int = 200,
    show_progress: bool = True,
) -> Tuple[Array, Array]:
    """
    Estimate maximum ratio of target to base density using empirical supremum
    rejection sampling.

    Arguments
    - density_fun: density for rejection sampling. Must be jit-able.
    - sample_base: distribution for rejection sampling. Takes a key and number
      of samples. Must be jit-able.
    - density_fun_base: (possibly unnormalized) density for `sample_base`. Must
      be jit-able.
    - n_iter: number of iterations of rejection sampling to use.
    - n_init: as an initial guess, the maximum ratio will be computed over
      `n_init` points sampled from `sample_base`.

    Returns
    - The estimated maximum ratio and point at which it was attained.

    Reference: https://bookdown.org/rdpeng/advstatcomp/rejection-sampling.html
    """
    # Get initial guess for ratio_max by computing it at random points
    key, subkey = random.split(key)
    thetas = sample_base(subkey, n_init)
    densities = jax.lax.map(density_fun, thetas)
    densities_base = jax.lax.map(density_fun_base, thetas)
    ratios = densities / densities_base
    idx_max = jnp.argmax(ratios)
    ratio_max = ratios[idx_max]
    theta_max = thetas[idx_max]

    @jax.jit
    def rejection_sample(key, ratio_max):
        """Generate ratio and point by rejection sampling."""

        def cond_fun(val):
            cond_key, ratio = val[1], val[2]
            u = random.uniform(cond_key)
            return u >= ratio / ratio_max

        def body_fun(val):
            key = val[0]
            key, theta_key, cond_key = random.split(key, 3)
            theta = sample_base(theta_key, 1)[0]
            ratio = density_fun(theta) / density_fun_base(theta)
            return (key, cond_key, ratio, theta)

        # Only first element of init_val matters
        init_val = body_fun((key, None, None, None))
        # Get max ratio and point at which it is attained
        _, _, ratio, theta = jax.lax.while_loop(cond_fun, body_fun, init_val)
        return ratio, theta

    iterator = trange(n_iter) if show_progress else range(n_iter)
    for _ in iterator:
        key, subkey = random.split(key)
        ratio, theta = rejection_sample(subkey, ratio_max)

        if ratio > ratio_max:
            ratio_max = ratio
            theta_max = theta

            if show_progress:
                iterator.set_postfix_str(f"{ratio:.3e} at {theta}")  # type: ignore

    if show_progress:
        iterator.close()  # type: ignore

    return ratio_max, theta_max


def gen_template_rejection(
    key: PRNGKeyArray,
    ratio_max,
    density_fun: Callable[[Array], Array],
    sample_base: Callable[[PRNGKeyArray, int], Array],
    density_fun_base: Callable[[Array], Array] = lambda _: jnp.array(1.0),
) -> Array:
    """
    Generates a single template using rejection sampling.

    Arguments
    - ratio_max: maximum value of the ratio of density_fun to density_fun_base.
    - density_fun: density for rejection sampling. Must be jit-able.
    - sample_base: distribution for rejection sampling. Takes a key and number
      of samples. Must be jit-able.
    - density_fun_base: (possibly unnormalized) density for `sample_base`. Must
      be jit-able.

    Returns
    - A single sample from the distribution with density `density_fun`.
    """

    def cond_fun(val):
        cond_key, theta = val[1], val[2]
        u = random.uniform(cond_key)
        return u >= density_fun(theta) / (ratio_max * density_fun_base(theta))

    def body_fun(val):
        key = val[0]
        key, theta_key, cond_key = random.split(key, 3)
        theta = sample_base(theta_key, 1)[0]
        return (key, cond_key, theta)  # new val

    # Only first element of init_val matters
    init_val = body_fun((key, None, None))

    return jax.lax.while_loop(cond_fun, body_fun, init_val)[2]


def gen_templates_rejection(
    key: PRNGKeyArray,
    n_templates,
    ratio_max,
    density_fun: Callable[[Array], Array],
    sample_base: Callable[[PRNGKeyArray, int], Array],
    density_fun_base: Callable[[Array], Array] = lambda _: jnp.array(1.0),
) -> Array:
    """
    Generates a bank with n_templates samples using rejection sampling.

    Arguments
    - ratio_max: maximum value of the ratio of density_fun to density_fun_base.
    - density_fun: density for rejection sampling. Must be jit-able.
    - sample_base: distribution for rejection sampling. Takes a key and number
      of samples. Must be jit-able.
    - density_fun_base: (possibly unnormalized) density for `sample_base`. Must
      be jit-able.

    TODO: add tqdm
    """
    keys = random.split(key, n_templates)
    f = lambda key: gen_template_rejection(
        key, ratio_max, density_fun, sample_base, density_fun_base
    )
    return jax.lax.map(f, keys)


def _update_uncovered_eff(
    pt: Array,
    eff: Array,
    template: Array,
    minimum_match: Union[float, Array],
    effectualness_fun: Callable[[Array, Array], Array],
) -> Array:
    """
    Computes effectualness for a point not already covered by a template.
    Jit-able.
    """
    uncovered = eff < minimum_match

    def true_fun(pt):
        return effectualness_fun(template, pt)

    def false_fun(_):
        return eff

    return jax.lax.cond(uncovered, true_fun, false_fun, pt)


def gen_bank_random(
    key: PRNGKeyArray,
    minimum_match: Union[float, Array],
    eta: Union[float, Array],
    effectualness_fun: Callable[[Array, Array], Array],
    ratio_max: Array,
    density_fun: Callable[[Array], Array],
    sample_base: Callable[[PRNGKeyArray, int], Array],
    density_fun_base: Callable[[Array], Array] = lambda _: jnp.array(1.0),
    eff_pt_sampler: Callable[[PRNGKeyArray], Array] = None,
    n_eff: int = 1000,
    show_progress: bool = True,
) -> Tuple[Array, Array]:
    """
    Arguments
    - effectualness_fun: function computing effectualness between points. Must
      be jit-able.
    - ratio_max: maximum value of the ratio of density_fun to density_fun_base.
    - density_fun: density for rejection sampling. Must be jit-able.
    - sample_base: distribution for rejection sampling. Takes a key and number
      of samples. Must be jit-able.
    - density_fun_base: (possibly unnormalized) density for base sampler. Must
      be jit-able.
    - eff_pt_sampler: sampler for effectualness points. Need not be jit-able.
      If None, the template rejection sampler will be used.

    Reference: Messenger, Prix & Papa 2008, https://arxiv.org/abs/0809.5223
    """
    # Function for rejection sampling of templates
    gen_template = jax.jit(
        lambda key: gen_template_rejection(
            key, ratio_max, density_fun, sample_base, density_fun_base
        )
    )
    if eff_pt_sampler is None:
        eff_pt_sampler = gen_template

    # Generate points for effectualness monitoring
    key, subkey = random.split(key)
    eff_pts = jnp.array([eff_pt_sampler(k) for k in random.split(subkey, n_eff)])
    effs = jnp.zeros(n_eff)
    n_covered = 0

    # Close over eff_pts
    @jax.jit
    def update_uncovered_effs(template, effs):
        update = lambda ep: _update_uncovered_eff(
            ep["point"], ep["eff"], template, minimum_match, effectualness_fun
        )
        return jax.vmap(update)({"point": eff_pts, "eff": effs})

    # Fill the bank!
    templates = []
    n_ko = int(jnp.ceil(n_eff * eta))
    with tqdm(total=n_ko) if show_progress else nullcontext() as pbar:
        while n_covered < n_ko:
            # Make template
            key, key_template = random.split(key)
            template = gen_template(key_template)
            templates.append(template)

            # Compute matches
            effs = update_uncovered_effs(template, effs)
            # Update coverage count
            dn_covered = (effs > minimum_match).sum() - n_covered
            n_covered += dn_covered

            if show_progress:  # pbar is a tqdm
                pbar.update(int(dn_covered))  # type: ignore
                pbar.set_postfix_str(f"n_templates = {len(templates)}")  # type: ignore

    return jnp.array(templates), eff_pts


def gen_bank_stochastic(
    key: PRNGKeyArray,
    minimum_match: Union[float, Array],
    eta: Union[float, Array],
    effectualness_fun: Callable[[Array, Array], Array],
    propose_template: Callable[[PRNGKeyArray], Array],
    eff_pt_sampler: Callable[[PRNGKeyArray], Array],
    n_eff: int = 1000,
    show_progress: bool = True,
    n_acc_monitoring: int = 1,  # number of iterations for acc rate moving average
) -> Tuple[Array, Array]:
    """
    Arguments
    - effectualness_fun: function computing effectualness between points. Must
      be jit-able.
    - eff_pt_sampler: sampler for effectualness points. Need not be jit-able.
    - n_acc_monitoring: number of iterations to use for moving average
      calculation of the acceptance rate

    Reference: https://arxiv.org/abs/0908.2090
    """
    effectualness_fun = jax.jit(effectualness_fun)

    # Generate points for effectualness monitoring
    key, subkey = random.split(key)
    eff_pts = jnp.array([eff_pt_sampler(k) for k in random.split(subkey, n_eff)])
    effs = jnp.zeros(n_eff)
    n_covered = 0

    # Close over eff_pts
    @jax.jit
    def update_uncovered_effs(template, effs):
        update = lambda ep: _update_uncovered_eff(
            ep["point"], ep["eff"], template, minimum_match, effectualness_fun
        )
        return jax.vmap(update)({"point": eff_pts, "eff": effs})

    # Returns True if point is far from all templates (ie, has a low
    # effectualness to the bank)
    def accept(pt, templates):
        effs = map(lambda template: effectualness_fun(template, pt), templates)
        return max(effs) < minimum_match

    # Add first template
    key, subkey = random.split(key)
    templates = [propose_template(subkey)]
    n_proposals = 1
    acc_rates = []
    n_ko = int(jnp.ceil(n_eff * eta))
    with tqdm(total=n_ko) if show_progress else nullcontext() as pbar:
        while n_covered < n_ko:
            # Make a template
            n_proposal_it = 0
            while True:
                key, subkey = random.split(key)
                template = propose_template(subkey)
                n_proposal_it += 1
                if accept(template, templates):
                    templates.append(template)
                    break

            # Update monitoring
            n_proposals += n_proposal_it
            acc_rate_it = 1 / n_proposal_it  # for the round
            if len(acc_rates) == n_acc_monitoring:
                acc_rates = acc_rates[1:]
            acc_rates.append(acc_rate_it)
            acc_rate = sum(acc_rates) / len(acc_rates)

            # Compute matches
            effs = update_uncovered_effs(template, effs)
            # Update coverage count
            dn_covered = (effs > minimum_match).sum() - n_covered
            n_covered += dn_covered

            if show_progress:  # pbar is a tqdm
                pbar.update(int(dn_covered))  # type: ignore
                pbar.set_postfix_str(  # type: ignore
                    f"acc rate = {acc_rate:.3f}, {len(templates)} / {n_proposals}"
                )

    return jnp.array(templates), eff_pts


def get_bank_effectualness(
    key: PRNGKeyArray,
    minimum_match: Union[float, Array],
    templates: Array,
    effectualness_fun: Callable[[Array, Array], Array],
    eff_pt_sampler: Callable[[PRNGKeyArray], Array],
    n: int = 100,
    show_progress: bool = True,
) -> Tuple[Array, Array, Array, Array]:
    """
    Computes effectualness of a bank at random points.

    Arguments
    - effectualness_fun: function computing effectualness between points. Must
      be jit-able.
    - eff_pt_sampler: sampler for effectualness points. Need not be jit-able.
    """
    # Compile in the templates and waveform model
    @jax.jit
    def get_bank_eff(pt):
        return jax.lax.map(
            lambda template: effectualness_fun(template, pt), templates
        ).max()

    # Sample points and compute effectualnesses
    eff_pts = jnp.array([eff_pt_sampler(k) for k in random.split(key, n)])
    effs = []
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    eta_est, eta_est_err, M_2 = jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)
    with tqdm(eff_pts) if show_progress else nullcontext() as pbar:
        for n, pt in enumerate(pbar, start=1):  # type: ignore
            effs.append(get_bank_eff(pt))

            x = effs[-1] > minimum_match
            eta_prev = eta_est
            eta_est = eta_prev + (x - eta_prev) / n
            M_2 = M_2 + (x - eta_prev) * (x - eta_est)
            if n > 1:
                # Standard deviation of the mean
                eta_est_err = jnp.sqrt(M_2 / (n - 1)) / jnp.sqrt(n)

            if show_progress:  # pbar is a tqdm
                pbar.set_postfix_str(f"eta = {eta_est:.3f} +/- {eta_est_err:.3f}")  # type: ignore

    return jnp.array(effs), eff_pts, eta_est, eta_est_err
