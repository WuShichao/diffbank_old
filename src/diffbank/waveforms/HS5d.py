from math import pi
import numpy as np
import jax.numpy as jnp


"""
hs 5d waveform metric parametrized by the black hole masses (m1, m2).
"""

def Psi(f: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    
    mass1,mass2,chiz,kappa2PN_eff,kappa3PN_eff = theta

    Mt = mass1 + mass2
    eta = mass1 * mass2 / (mass1 + mass2) ** 2

    gt = 4.92549094830932e-6  # GN*Msun/c^3 in seconds
    EulerGamma = 0.57721566490153286060
    vlso = 1.0 / jnp.sqrt(6.0)

    L = jnp.array([0.0, 0.0, 1.0])

    chi = jnp.array([0.0, 0.0, chiz])
    delta = jnp.sqrt(1.0 - 4.0 * eta)

    v = (jnp.pi * Mt * (f + 1e-100) * gt) ** (1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v3 * v3
    v7 = v3 * v4
    v10 = v5 * v5
    v12 = v10 * v2
    eta2 = eta ** 2.0
    eta3 = eta ** 3.0

    # rho parameters for 3PN terms
    rho0 = (
        113.0
        * jnp.pi
        / (113.0 - 76.0 * eta)
        * (2270.0 / 3.0 - 520.0 * eta)
    )
    rho1 = (
        (113.0 / (113.0 - 76.0 * eta)) ** 2.0
        / (8475.0 + 1444.0 * eta)
        * (
            4008512167.0 / 672.0
            - 10713456163.0 * eta / 504.0
            + 2094712507.0 * eta2 / 126.0
            - 1545080.0 * eta3 / 9.0
        )
    )
    rho2 = 3760.0 * jnp.pi * delta * eta / (339.0 - 228.0 * eta)
    rho3 = (
        113.0
        * delta
        * eta
        * (12198367.0 - 16731218.0 * eta)
        / (84.0 * (113.0 - 76.0 * eta) ** 2.0)
    )
    rho4 = (
        eta2
        * (109852619.0 - 341495546.0 * eta - 89556880.0 * eta2)
        / (21.0 * (113.0 - 76.0 * eta) ** 2.0)
    )

    # # ------------------------- Non spinning part of the waveform
    psi_NS_0PN = 1.0
    psi_NS_1PN = (3715.0 / 756.0 + 55.0 * eta / 9.0) * v2
    psi_NS_15PN = -16.0 * jnp.pi * v3
    psi_NS_2PN = (
        15293365.0 / 508032.0 + 27145.0 * eta / 504.0 + 3085.0 * eta2 / 72.0
    ) * v4
    psi_NS_25PN = (
        jnp.pi
        * (38645.0 / 756.0 - 65.0 * eta / 9.0)
        * (1 + 3.0 * jnp.log(v / vlso))
        * v5
    )
    psi_NS_3PN = (
        (
            11583231236531.0 / 4694215680.0
            - 640.0 * jnp.pi ** 2 / 3.0
            - 6848.0 * EulerGamma / 21.0
        )
        + (2255.0 * jnp.pi ** 2 / 12.0 - 15737765635.0 / 3048192.0) * eta
        + 76055.0 * eta2 / 1728.0
        - 127825.0 * eta3 / 1296.0
        - 6848.0 * jnp.log(4.0 * v) / 21.0
    ) * v6
    psi_NS_35PN = (
        jnp.pi
        * (
            77096675.0 / 254016.0
            + 378515.0 * eta / 1512.0
            - 74045.0 * eta2 / 756.0
        )
        * v7
    )

    # ------------------------- Spining part of the waveform for the truncated waveform

    # Effective 1.5PN param
    psi_S_15PN = (113.0 / 3.0 * chi.dot(L)) * v3

    # Effective 2PN param
    psi_S_2PN = -50.0 * kappa2PN_eff
    psi_S_2PN += (
        -395.0
        / 8.0
        * (75.0 * delta ** 2.0 / 113.0 + 35344.0 * eta / 12769.0) ** (-1.0)
        * (chi.dot(L)) ** 2.0
    )
    psi_S_2PN *= v4

    # Truncated 2.5PN terms

    psi_S_25PN = (
        -(113.0 / (113.0 - 76.0 * eta))
        * (732985.0 / 2268.0 - 24260.0 * eta / 81.0 - 340.0 * eta2 / 9.0)
        * chi.dot(L)
    )
    # psi_S_25PN += (-(113.0/(113.0-76.0*eta))*(2086535.0/21357.0 + 9260.0*eta/339.0 )*delta*eta*chi_a.dot(L))*v5 # Truncation error at 2.5PN order
    psi_S_25PN *= v5
    psi_S_25PN_log = 3.0 * psi_S_25PN * jnp.log(v / vlso)

    # Effective 3PN param

    psi_S_3PN = 5.0 / 84.0 * (15609.0 - 4186.0 * eta) * kappa2PN_eff
    psi_S_3PN += -2215.0 / 6.0 * eta2 * kappa3PN_eff
    psi_S_3PN += rho0 * chi.dot(L) + rho1 * (chi.dot(L)) ** 2.0
    psi_S_3PN *= v6

    # Truncated 3.5PN terms

    A35 = 1130.0 * (-311.0 + 403.0 * eta) / (3.0 * (-113.0 + 76.0 * eta))
    B35 = 448610.0 * eta / (3.0 * (-113.0 + 76.0 * eta))

    S35lineff = (
        (113.0 / (113.0 - 76.0 * eta))
        * (
            -25150083775.0 / 3048192.0
            + 10566655595.0 * eta / 762048.0
            - 1042165.0 * eta2 / 3024.0
            + 5345.0 * eta3 / 36.0
        )
        * (chi.dot(L))
    )  # OK
    S35cubeff_kappa = A35 * kappa2PN_eff * (
        chi.dot(L)
    ) + B35 * eta * kappa3PN_eff * (chi.dot(L))
    S35cubeff = (
        1442897.0
        * (
            -223835711.0
            + 572749036.0 * eta
            - 1266382384.0 * eta2
            + 93340160.0 * eta3
        )
        * (chi.dot(L)) ** 3.0
        / (24.0 * (-113.0 + 76.0 * eta) ** 3.0 * (8475.0 + 1444.0 * eta))
    )

    # #Truncation errors
    # delta_S35lineff = -(5.0/(113.0-76.0*eta))*(5575530433.0/63504.0 + 15249235.0*eta/252.0 + 24542.0*eta2/9.0 )*delta*eta*(chi_a.dot(L)) # OK
    # delta_S35cubeff_kappa = (40.0/(3.0*(113.0-76.0*eta)*Mt**3.0)) * ((16691.0*m1 + 20627.0*m2)* m1**2.0 * kappa1*(chi_1.dot(L))**2.0 - (16691.0*m2 + 20627.0*m1)* m2**2.0*kappa2*(chi_2.dot(L))**2.0 ) * eta * chi_a.dot(L)
    # delta_S35cubeff = (160.0/(113.0 - 76.0*eta)**3.0) *(54927553.0 + 4308896.0*eta)* eta3 * delta * (chi_a.dot(L))**3.0
    # delta_S35cubeff += (452.0/(3101.0*(-113.0+76.0*eta)**3.0))*(10853245537.0 + 183323602308.0*eta) *eta2 * chi.dot(L) * (chi_a.dot(L))**2.0
    # delta_S35cubeff += - (674709440.0/(1329.0*(113.0-76.0*eta)**2.0)) * jnp.pi * delta * eta * chi_a.dot(L) * chi.dot(L)
    # delta_S35cubeff += (12769.0/(18606.0*(-113.0+76.0*eta)**3.0)) * (13380385211.0 + 2443362404.0*eta) * delta *eta * chi_a.dot(L) * (chi.dot(L))**2.0

    psi_S_35PN = S35lineff + S35cubeff_kappa + S35cubeff
    # psi_S_35PN += delta_S35lineff + delta_S35cubeff_kappa + delta_S35cubeff #Truncated terms
    psi_S_35PN *= v7

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
        psi_S_15PN
        + psi_S_2PN
        + psi_S_25PN
        + psi_S_25PN_log
        + psi_S_3PN
        + psi_S_35PN
    )

    return 3.0 / 128.0 / eta / v5 * (psi_NS + psi_S)

def amp(f: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:

    mass1,mass2,_,_,_ = theta
    distance=1.0
    Mt = mass1 + mass2
    eta = mass1 * mass2 / (mass1 + mass2) ** 2

    pre = 3.6686934875530996e-19  # (GN*Msun/c^3)^(5/6)/Hz^(7/6)*c/Mpc/sec
    Mchirp = Mt * eta ** 0.6
    A0 = (
        Mchirp ** (5.0 / 6.0)
        / (f + 1e-100) ** (7.0 / 6.0)
        / distance
        / jnp.pi ** (2.0 / 3.0)
        * jnp.sqrt(5.0 / 24.0)
    )

    return pre * A0

