# import numpy as np
from math import pi
import jax.numpy as jnp


def Psi_35PN(f, theta):

    m1, m2, s1, s2 = theta
    # m1, m2 = theta
    # Mt, eta = theta
    gt = 4.92549094830932e-6  # GN*Msun/c**3 in seconds
    Mt = m1 + m1
    eta = m1 * m2 / (m1 + m2) ** 2
    EulerGamma = 0.57721566490153286060
    vlso = 1.0 / jnp.sqrt(6.0)

    chi_s = 0.5 * (s1 + s2)
    chi_a = 0.5 * (s1 - s2)
    k_s = 0.5 * (1.0 + 1.0)
    k_a = 0.5 * (1.0 - 1.0)

    lambda_s = 0.5 * (1.0 + 1.0)
    lambda_a = 0.5 * (1.0 - 1.0)
    L = 1.0
    delta = jnp.sqrt(1.0 - 4.0 * eta)

    v = (pi * Mt * (f + 1e-100) * gt) ** (1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v3 * v3
    v7 = v3 * v4
    eta2 = eta ** 2
    eta3 = eta ** 3

    # # ------------------------- Non spinning part of the waveform
    psi_NS_0PN = 1.0
    psi_NS_1PN = (3715.0 / 756.0 + 55.0 * eta / 9.0) * v2
    psi_NS_15PN = -16.0 * pi * v3
    psi_NS_2PN = (
        15293365.0 / 508032.0 + 27145.0 * eta / 504.0 + 3085.0 * eta2 / 72.0
    ) * v4
    psi_NS_25PN = (
        pi * (38645.0 / 756.0 - 65.0 * eta / 9.0) * (1 + 3.0 * jnp.log(v / vlso)) * v5
    )
    psi_NS_3PN = (
        (
            11583231236531.0 / 4694215680.0
            - 640.0 * pi ** 2 / 3.0
            - 6848.0 * EulerGamma / 21.0
        )
        + (2255.0 * pi ** 2 / 12.0 - 15737765635.0 / 3048192.0) * eta
        + 76055.0 * eta2 / 1728.0
        - 127825.0 * eta3 / 1296.0
        - 6848.0 * jnp.log(4.0 * v) / 21.0
    ) * v6
    psi_NS_35PN = (
        pi
        * (77096675.0 / 254016.0 + 378515.0 * eta / 1512.0 - 74045.0 * eta2 / 756.0)
        * v7
    )

    # # ------------------------- Spining part of the waveform
    # 1.5 PN
    psi_S_15PN = (
        (113.0 / 3.0 - 76.0 * eta / 3.0) * (chi_s * L)
        + 113.0 * delta * (chi_a * L) / 3.0
    ) * v3

    # 2 PN
    psi_S_2PN_t1 = (
        -(5.0 / 8.0)
        * (1.0 + 156.0 * eta + 80.0 * delta * k_a + 80.0 * (1.0 - 2.0 * eta) * k_s)
        * ((chi_s * L) ** 2)
    )
    psi_S_2PN_t2 = (
        (5.0 / 8.0)
        * (1.0 - 160.0 * eta + 80.0 * delta * k_a + 80.0 * (1.0 - 2.0 * eta) * k_s)
        * ((chi_a * L) ** 2)
    )
    psi_S_2PN_t3 = (
        (5.0 / 4.0)
        * (delta + 80.0 * delta * k_s + 80.0 * (1.0 - 2.0 * eta) * k_a)
        * (chi_s * L)
        * (chi_a * L)
    )
    psi_S_2PN = (psi_S_2PN_t1 - psi_S_2PN_t2 - psi_S_2PN_t3) * v4

    # 2.5 PN
    psi_S_25PN = (
        -(732985.0 / 2268.0 - 24260.0 * eta / 81.0 - 340.0 * eta2 / 9.0) * (chi_s * L)
        - (732985.0 / 2268.0 + 140.0 * eta / 9.0) * delta * (chi_a * L)
    ) * v5
    psi_S_25PN_log = 3.0 * psi_S_25PN * jnp.log(v / vlso)

    # 3 PN
    psi_S_3PN_t1 = (2270.0 / 3.0 - 520.0 * eta) * pi * (chi_s * L) + (
        2270.0 * pi / 3.0
    ) * delta * (chi_a * L)
    psi_S_3PN_t2 = (
        (
            (26015.0 / 14.0 - 88510.0 * eta / 21.0 - 480.0 * eta2) * k_a
            + delta
            * (
                -1344475.0 / 1008.0
                + 745.0 * eta / 18.0
                + (26015.0 / 14.0 - 1495.0 * eta / 3.0) * k_s
            )
        )
        * (chi_s * L)
        * (chi_a * L)
    )
    psi_S_3PN_t3 = (
        -1344475.0 / 2016.0
        + 829705.0 * eta / 504.0
        + 3415.0 * eta2 / 9.0
        + (26015.0 / 28.0 - 44255.0 * eta / 21.0 - 240.0 * eta2) * k_s
        + delta * (26015.0 / 28.0 - 1495.0 * eta / 6.0) * k_a
    ) * ((chi_s * L)) ** 2
    psi_S_3PN_t4 = (
        -1344475.0 / 2016.0
        + 267815.0 * eta / 252.0
        - 240.0 * eta2
        + (26015.0 / 28.0 - 44255.0 * eta / 21.0 - 240.0 * eta2) * k_s
        + delta * (26015.0 / 28.0 - 1495.0 * eta / 6.0) * k_a
    ) * ((chi_a * L)) ** 2
    psi_S_3PN = (psi_S_3PN_t1 + psi_S_3PN_t2 + psi_S_3PN_t3 + psi_S_3PN_t4) * v6

    # 3.5 PN
    psi_S_35PN_t1 = (
        -25150083775.0 / 3048192.0
        + 10566655595.0 * eta / 762048.0
        - 1042165 * eta2 / 3024.0
        + 5345.0 * eta3 / 36.0
    ) * (chi_s * L)
    psi_S_35PN_t2 = (
        (-25150083775.0 / 3048192.0 + 26804935.0 * eta / 6048.0 - 1985.0 * eta2 / 48.0)
        * delta
        * (chi_a * L)
    )
    psi_S_35PN_t3 = (
        265.0 / 24.0
        + 4035.0 * eta / 2.0
        - 20.0 * eta2 / 3.0
        + (3110.0 / 3.0 - 10250.0 * eta / 3.0 + 40.0 * eta2) * k_s
        - 440.0 * (1.0 - 3.0 * eta) * lambda_s
        + delta
        * ((3110.0 / 3.0 - 4030.0 * eta / 3.0) * k_a - 440.0 * (1.0 - eta) * lambda_a)
    ) * ((chi_s * L)) ** 3
    psi_S_35PN_t4 = (
        (3110.0 / 3.0 - 8470.0 * eta / 3.0) * k_a
        - 440.0 * (1.0 - 3.0 * eta) * lambda_a
        + delta
        * (
            265.0 / 24.0
            - 2070.0 * eta
            + (3110.0 / 3.0 - 750.0 * eta) * k_s
            - 440.0 * (1 - eta) * lambda_s
        )
    ) * ((chi_a * L)) ** 3
    psi_S_35PN_t5 = (
        (3110.0 - 28970.0 * eta / 3.0 + 80.0 * eta2) * k_a
        - 1320.0 * (1.0 - 3.0 * eta) * lambda_a
        + delta
        * (
            265.0 / 8.0
            + 12055.0 * eta / 6.0
            + (3110.0 - 10310.0 * eta / 3.0) * k_s
            - 1320.0 * (1.0 - eta) * lambda_s
        )
    ) * ((chi_s * L) ** 2 * (chi_a * L))
    psi_S_35PN_t6 = (
        265.0 / 8.0
        - 6500.0 * eta / 3.0
        + 40.0 * eta2
        + (3110.0 - 27190.0 * eta / 3.0 + 40.0 * eta2) * k_s
        - 1320.0 * (1.0 - 3 * eta) * lambda_s
        + delta
        * ((3110.0 - 8530.0 * eta / 3.0) * k_a - 1320.0 * (1.0 - eta) * lambda_a)
    ) * ((chi_a * L) ** 2 * (chi_s * L))

    psi_S_35PN = (
        psi_S_35PN_t1
        + psi_S_35PN_t2
        + psi_S_35PN_t3
        + psi_S_35PN_t4
        + psi_S_35PN_t5
        + psi_S_35PN_t6
    ) * v7

    psi_NS = (
        psi_NS_0PN
        + psi_NS_1PN
        + psi_NS_15PN
        + psi_NS_2PN
        + psi_NS_25PN
        + psi_NS_3PN
        + psi_NS_35PN
    )

    psi_S = (
        psi_S_15PN + psi_S_2PN + psi_S_25PN + psi_S_25PN_log + psi_S_3PN + psi_S_35PN
    )

    return 3.0 / 128.0 / eta / v5 * (psi_NS + psi_S)
    # return 3.0 / 128.0 / eta / v5 * psi_NS


def Amp_35PN(f, theta):
    return f ** (-7 / 6)
