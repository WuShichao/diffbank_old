from jax import jit
import jax.numpy as jnp

from diffbank.metric import get_g, get_gam
from diffbank.noise import Sn_aLIGO as Sn_func
from diffbank.waveforms.taylorf2 import amp as _amp, Psi as _Psi

"""
Makes sure the TaylorF2 metric calculation works.
"""

amp = lambda f, theta: jit(_amp)(
    f, M=theta[0], eta=theta[1], s1z=theta[2], s2z=theta[3]
)

Psi = lambda f, theta: jit(_Psi)(
    f, M=theta[0], eta=theta[1], s1z=theta[2], s2z=theta[3]
)


def test_taylorf2_dets():
    m1 = jnp.array(3.0)
    m2 = jnp.array(1.5)
    M = m1 + m2
    eta = m1 * m2 / (m1 + m2) ** 2

    f_min, f_max = jnp.array(10.0), jnp.array(512.0)
    fs = jnp.linspace(f_min, f_max, 10000)

    gam = get_gam(jnp.array([M, eta, 0.3, 0.5]), amp, Psi, fs, Sn_func)
    g = get_g(jnp.array([M, eta, 0.3, 0.5]), amp, Psi, fs, Sn_func)

    det_g = jnp.linalg.det(g)

    assert not (jnp.isinf(det_g) or jnp.isnan(det_g) or det_g <= 0)
    print(g, jnp.linalg.det(gam), det_g)
    return jnp.linalg.det(gam), det_g


if __name__ == "__main__":
    test_taylorf2_dets()
