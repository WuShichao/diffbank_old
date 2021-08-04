from typing import Callable
import jax
from jax import random
import jax.numpy as jnp
import jax.experimental.host_callback as hcb
from tqdm.auto import tqdm
from .utils import get_effectualness


def pbar_factory(total):
    """
    Returns closures for manipulating a tqdm progress bar.
    """
    pbar = tqdm(initial=1, total=total)

    def incr(n_templates):
        hcb.id_tap(lambda *_: pbar.update(1), arg=n_templates)

    def set_n_rejects(n_rejects):
        hcb.id_tap(
            lambda n_rejects, _: pbar.set_description(f"n_rejects = {n_rejects:d}"),
            n_rejects,
        )

    def close():
        hcb.id_tap(lambda *_: pbar.close(), None)

    return incr, set_n_rejects, close


def gen_bank_stochastic(
    key: jnp.ndarray,
    gen_template: Callable[[jnp.ndarray], jnp.ndarray],
    amp,
    Psi,
    fs,
    Sn: Callable[[jnp.ndarray], jnp.ndarray],
    minimum_match: int,
    n: int,
):
    """
    Generate a stochastic template bank.

    Algorithm
    - Sample a point from some distribution
    - Check the maximum match with all templates in the bank. Can speed this up
      if the metric is available.
    - Accept if maximum match is below some threshold
    - Stop sampling when some convergence criterion is met. Currently this is
      controlled by n, the number of templates to generate.

    Reference: https://arxiv.org/abs/0908.2090
    """
    n_dim = 2  # len(gen_template(key))
    _get_match = jax.jit(
        lambda theta1, theta2: get_effectualness(theta1, theta2, amp, Psi, fs, Sn)
    )
    incr, set_n_rejects, close = pbar_factory(n)

    def body_fun(i, val):
        """
        Generate template i
        """
        key, templates, n_rejects = val
        key, key_template = random.split(key)

        def cond(val):
            """
            Check if max match with existing templates is small enough
            """
            _, template, _, _ = val
            # Hacky, and would be better to only compute match with templates
            # that were already generated. Unclear how to do this in jax. My
            # attempts to do this with cond or fori_loop have failed, giving
            # nan matches.
            max_match = jnp.where(
                jnp.isnan(templates).any(-1),
                0.0,
                jax.lax.map(lambda t: _get_match(template, t), templates),
            ).max()
            return jax.lax.gt(max_match, minimum_match)

        def body(val):
            """
            Generate a template
            """
            key, _, _, n_rejects = val
            key, key_template = random.split(key)
            template = gen_template(key_template)
            return key, template, 0.0, n_rejects + 1

        # Loop until a template is generated with maximum match below the threshold
        init_val = (key, gen_template(key_template), 0.0, n_rejects)
        key, template, _, n_rejects = jax.lax.while_loop(cond, body, init_val)
        templates = templates.at[i - 1].set(template)

        # Update pbar
        incr(i)
        set_n_rejects(n_rejects)

        return key, templates, n_rejects

    # Add the first template
    key, key_template = random.split(key)
    templates = jnp.full((n, n_dim), jnp.nan)
    templates.at[0].set(gen_template(key_template))

    init_val = (key, templates, 0)
    _, templates, n_rejects = jax.lax.fori_loop(1, n + 1, body_fun, init_val)
    # Close pbar
    close()

    return templates, n_rejects
