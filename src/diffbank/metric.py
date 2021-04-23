from functools import partial
from math import pi

from jax import jacfwd, jacrev
from jax import jit
import jax.numpy as jnp


def hessian(fun):
    return jit(jacfwd(jacrev(fun)))


@partial(jit, static_argnums=(3, 4, 6))
def match(theta, del_t, del_theta, amp, Psi, fs, Sn):
    # Evaluate all functions over frequency gradient
    amps = amp(fs, theta)
    amp_del = amp(fs, theta + del_theta)
    amp_prods = amps * amp_del
    del_phases = (2.0 * pi * fs * del_t) + (Psi(fs, theta + del_theta) - Psi(fs, theta))
    Sns = Sn(fs)

    # Normalize both waveforms
    norm = jnp.sqrt(4 * jnp.trapz(amps ** 2 / Sns, fs))
    norm_del = jnp.sqrt(4 * jnp.trapz(amp_del ** 2 / Sns, fs))

    # Compute unnormalized match, maximizing over phi_0 by taking the absolute value
    re_integral = 4.0 * jnp.trapz(amp_prods * jnp.cos(del_phases) / Sns, fs)
    im_integral = 4.0 * jnp.trapz(amp_prods * jnp.sin(del_phases) / Sns, fs)
    match_un = jnp.sqrt(re_integral ** 2 + im_integral ** 2)

    return match_un / (norm * norm_del)


@partial(jit, static_argnums=(1, 2, 4))
def get_gam(theta, amp, Psi, fs, Sn):
    """
    Metric for t₀ and intrinsic parameters.
    """
    hess_func = lambda delta: match(theta, delta[0], delta[1:], amp, Psi, fs, Sn)
    del_theta = jnp.zeros(theta.size + 1)
    return -1 / 2 * hessian(hess_func)(del_theta)


@partial(jit, static_argnums=(1, 2, 4))
def get_g(theta, amp, Psi, fs, Sn) -> jnp.ndarray:
    """
    Metric for intrinsic parameters.
    """
    gam = get_gam(theta, amp, Psi, fs, Sn)
    # Maximize over Delta t_0
    return gam[1:, 1:] - jnp.outer(gam[0, 1:], gam[0, 1:]) / gam[0, 0]


@partial(jit, static_argnums=(1, 2, 4))
def get_density(theta, amp, Psi, f, Sn) -> jnp.ndarray:
    """
    Gets function returning metric determinant.
    """
    return jnp.sqrt(jnp.linalg.det(get_g(theta, amp, Psi, f, Sn)))
