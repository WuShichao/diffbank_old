import time

from diffbank.metric import match
from diffbank.utils import _get_effectualness, get_eff_pads
from diffbank.noise import Sn_LIGOI as Sn
from diffbank.waveforms.threePN_simple import Psi, amp
from jax import jit, vmap
import jax.numpy as jnp


"""
Runs various tests on the effectualness calculation to make sure it works and
that the FFT properly maximizes over del_t.
"""


def test_match():
    """
    Checks that match function works and times it.
    """
    theta_1 = jnp.array([3.0, 0.8])
    theta_2 = jnp.array([3.0, 0.8])
    fs = jnp.linspace(10, 500, 10000)
    pad_low, pad_high = get_eff_pads(fs)

    get_match = jit(_get_effectualness, static_argnums=(2, 3, 5))

    t0 = time.time()
    for _ in range(1000):
        get_match(theta_1, theta_2, amp, Psi, fs, Sn, pad_low, pad_high)

    t1 = time.time()

    print((t1 - t0) / 1000)


def test_matches():
    """
    Makes sure the match function with an explicit del_t argument agrees with
    the FFT one which maximizes over del_t.
    """
    theta_1 = jnp.array([2.20013935, 1.13180361])
    theta_2 = jnp.array([2.27775711, 1.30239947])
    del_ts = jnp.linspace(-0.5, 0.1, 2000)
    fs = jnp.linspace(10, 500, 10000)
    pad_low, pad_high = get_eff_pads(fs)

    matches = vmap(match, in_axes=(None, 0, None, None, None, None, None))(
        theta_1, del_ts, theta_2 - theta_1, amp, Psi, fs, Sn
    ).max()
    match_fft = _get_effectualness(
        theta_1, theta_2, amp, Psi, fs, Sn, pad_low, pad_high
    )

    assert jnp.allclose(match_fft, matches.max(), rtol=5e-3)


if __name__ == "__main__":
    test_match()
    test_matches()
