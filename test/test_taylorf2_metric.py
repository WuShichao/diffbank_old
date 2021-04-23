from jax import jit
import jax.numpy as jnp

from core.metric import get_g, get_gam
from core.utils import Sn_func
from taylorf2_waveform import Af3hPN, Phif3hPN


amp = lambda f, theta: jit(Af3hPN)(
    f, M=theta[0], eta=theta[1], s1z=theta[2], s2z=theta[3]
)

Psi = lambda f, theta: jit(Phif3hPN)(
    f, M=theta[0], eta=theta[1], s1z=theta[2], s2z=theta[3]
)


def test_taylorf2_dets():
    """
    Makes sure the TaylorF2 metric calculation works.
    """
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


# def test_taylorf2_Ntemplate():
#     m_star = 1 - 0.95
#     p_ranges = jnp.array([[1, 6], [0.1875, 0.25], [0.0, 0.1], [0.0, 0.1]])
#     N_test = 5000
#     eta = 0.9999
#
#     f_min, f_max = jnp.array(10.0), jnp.array(512.0)
#     fs = jnp.linspace(f_min, f_max, 10000)
#
#     N_check = N_templates(
#         p_ranges,
#         N_test,
#         eta,
#         m_star,
#         amp,
#         Psi,
#         fs,
#         Sn_func,
#         gen_samples_from_ranges_Meta,
#     )
#     print(N_check)


test_taylorf2_dets()
# test_taylorf2_Ntemplate()
