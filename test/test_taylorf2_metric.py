from diffbank.metric import get_g, get_gam
from diffbank.noise import Sn_aLIGO as Sn_func
from diffbank.waveforms.taylorF2 import Psi as _Psi, Amp as _amp
from jax import jit
import jax.numpy as jnp

"""
Makes sure the TaylorF2 metric calculation doesn't give infinities or nans.
"""


def test_taylorf2_dets():
    amp = lambda f, theta: jit(_amp)(f, theta)
    Psi = lambda f, theta: jit(_Psi)(f, theta)

    m1 = jnp.array(3.0)
    m2 = jnp.array(1.5)

    f_min, f_max = jnp.array(10.0), jnp.array(512.0)
    fs = jnp.linspace(f_min, f_max, 10000)

    gam = get_gam(jnp.array([m1, m2, 0.3, 0.5]), amp, Psi, fs, Sn_func)
    g = get_g(jnp.array([m1, m2, 0.3, 0.5]), amp, Psi, fs, Sn_func)

    det_g = jnp.linalg.det(g)

    assert not (jnp.isinf(det_g) or jnp.isnan(det_g) or det_g <= 0)
    print(g, jnp.linalg.det(gam), det_g)
    return jnp.linalg.det(gam), det_g


if __name__ == "__main__":
    test_taylorf2_dets()
