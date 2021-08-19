try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from . import noise_resources


def Sn_aLIGO(f):
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


def load_noise(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Loads noise curve from dat file with columns containing frequencies and PSD
    values.

    Arguments
    - name: name of datafile in `noise_resources` without the `.dat` extension.

    Returns
        Interpolator for noise curve returning zero above and below
        frequency range in the datafile.
    """
    path_context = pkg_resources.path(noise_resources, f"{name}.dat")
    with path_context as path:
        fs, Sns = np.loadtxt(path, unpack=True)
    Sn = jax.jit(lambda f: jnp.interp(f, fs, Sns, 0, 0))
    return Sn


Sn_aLIGOZeroDetHighPower = load_noise("aLIGOZeroDetHighPower")
