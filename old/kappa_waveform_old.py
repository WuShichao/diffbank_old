class kappa_waveform(object):
    def __init__(
        self,
        mass1=None,
        mass2=None,
        spin1z=0.0,
        spin2z=0.0,
        kappa1=1.0,
        kappa2=1.0,
        distance=1.0,
        inclination=0.0,
        spin1x=0.0,
        spin2x=0.0,
        spin1y=0.0,
        spin2y=0.0,
        lambda1=1.0,
        lambda2=1.0,
        comp1=1.0,
        comp2=1.0,
        delta_f=0.25,
        f_lower=20.0,
        f_upper=16384.0,
        **kwargs,
    ):
        """
        This class represents extensions to standard compact binary coalescence
        scenarios where the objects are less compact. This includes astrophysical objects
        from a BH at kappa = 1, to neutron star (kappa ~ few - few tens), to more
        exotic objects (kappa > 20). Where this kappa is the dimensionless
        quadrupole, which induces variations to the GW waveform due to the spin
        of the object.
        Parameters:
        -----------
            self.frequencies: list
                list of frequencies
            M_total: float
                total gravitational mass of the binary
            eta: float
                symmetric mass ratio
            as_pycbc: bool
                flag to return waveform as a pycbc frequencyseries or
                timeseries object
            kwargs: ...
                Other optional arguments for waveform generation
        """
        if mass1 is None or mass2 is None:
            raise Exception("Mass1 and Mass2 must be specificed for this waveform.")
        # set kwargs to attributes else set default parameters
        self.mass1 = mass1
        self.mass2 = mass2
        self.Mt = mass1 + mass2
        self.eta = mass1 * mass2 / (mass1 + mass2) ** 2
        self.spin1z = spin1z
        self.spin1x = spin1x
        self.spin1y = spin1y
        self.spin2z = spin2z
        self.spin2x = spin2x
        self.spin2y = spin2y
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.comp1 = comp1
        self.comp2 = comp2
        self.distance = distance
        self.inclination = inclination
        self.delta_f = round(delta_f, 6)
        self.f_lower = f_lower
        self.f_upper = f_upper
        self.set_cutoff_frequency()
        if self.f_upper < self.f_cutoff:
            self.f_cutoff = self.f_upper
        else:
            self.f_upper = self.f_cutoff
        # TODO expand compactness impact
        if self.comp1 == 1.0:
            # print("C1 is Defaulting to BH compactness")
            self.comp1 = 1.0 / (1.0 + np.sqrt(1.0 - self.spin1z ** 2.0))

        if self.comp2 == 1.0:
            # print("C2 is defaulting to BH compactness")
            self.comp2 = 1.0 / (1.0 + np.sqrt(1.0 - self.spin2z ** 2.0))

        num_f = int((self.f_cutoff - self.f_lower) / self.delta_f) + 1
        self.frequencies = np.linspace(
            self.f_lower, self.f_cutoff, num_f, endpoint=True
        )
        self.hp_hc()

    def set_cutoff_frequency(self):
        c_l = 299792458.0
        Msun = 1.9884e30
        G_N = 6.6743e-11 * Msun
        f_ISCO = 4.4e3 * (1 / self.Mt)  # Hz
        f_touch = (
            (c_l ** 3 / G_N)
            * (1.0 / np.pi)
            * np.sqrt(
                (self.Mt) / (self.mass1 / self.comp1 + self.mass2 / self.comp2) ** 3
            )
        )  # Fix units to give Hz
        self.f_cutoff = min(f_ISCO, f_touch)

    def Phif35PN(self):
        """
        Computes the phase of the waveform. Sets time and phase of coealence to be zero.
        Returns:
        --------
        phase (array): Phase of the GW as a function of frequency
        """

        gt = 4.92549094830932e-6  # GN*Msun/c^3 in seconds
        EulerGamma = 0.57721566490153286060
        vlso = 1.0 / np.sqrt(6.0)
        chi_1 = np.array([self.spin1x, self.spin1y, self.spin1z])
        chi_2 = np.array([self.spin2x, self.spin2y, self.spin2z])

        chi_s = 0.5 * (chi_1 + chi_2)
        chi_a = 0.5 * (chi_1 - chi_2)
        k_s = 0.5 * (self.kappa1 + self.kappa2)
        k_a = 0.5 * (self.kappa1 - self.kappa2)

        lambda_s = 0.5 * (self.lambda1 + self.lambda2)
        lambda_a = 0.5 * (self.lambda1 - self.lambda2)
        L = np.array([0.0, 0.0, 1.0])
        delta = np.sqrt(1.0 - 4.0 * self.eta)

        v = (np.pi * self.Mt * (self.frequencies + 1e-100) * gt) ** (1.0 / 3.0)
        v2 = v * v
        v3 = v2 * v
        v4 = v2 * v2
        v5 = v4 * v
        v6 = v3 * v3
        v7 = v3 * v4
        v10 = v5 * v5
        v12 = v10 * v2
        eta2 = self.eta ** 2
        eta3 = self.eta ** 3

        # # ------------------------- Non spinning part of the waveform
        psi_NS_0PN = 1.0
        psi_NS_1PN = (3715.0 / 756.0 + 55.0 * self.eta / 9.0) * v2
        psi_NS_15PN = -16.0 * np.pi * v3
        psi_NS_2PN = (
            15293365.0 / 508032.0 + 27145.0 * self.eta / 504.0 + 3085.0 * eta2 / 72.0
        ) * v4
        psi_NS_25PN = (
            np.pi
            * (38645.0 / 756.0 - 65.0 * self.eta / 9.0)
            * (1 + 3.0 * np.log(v / vlso))
            * v5
        )
        psi_NS_3PN = (
            (
                11583231236531.0 / 4694215680.0
                - 640.0 * np.pi ** 2 / 3.0
                - 6848.0 * EulerGamma / 21.0
            )
            + (2255.0 * np.pi ** 2 / 12.0 - 15737765635.0 / 3048192.0) * self.eta
            + 76055.0 * eta2 / 1728.0
            - 127825.0 * eta3 / 1296.0
            - 6848.0 * np.log(4.0 * v) / 21.0
        ) * v6
        psi_NS_35PN = (
            np.pi
            * (
                77096675.0 / 254016.0
                + 378515.0 * self.eta / 1512.0
                - 74045.0 * eta2 / 756.0
            )
            * v7
        )

        # # ------------------------- Spining part of the waveform
        psi_S_15PN = (
            (113.0 / 3.0 - 76.0 * self.eta / 3.0) * chi_s.dot(L)
            + 113.0 * delta * chi_a.dot(L) / 3.0
        ) * v3

        psi_S_2PN = (
            -(5.0 / 8.0)
            * (
                1.0
                + 156.0 * self.eta
                + 80.0 * delta * k_a
                + 80.0 * (1.0 - 2.0 * self.eta) * k_s
            )
            * (chi_s.dot(L) ** 2)
        )
        psi_S_2PN -= (
            (5.0 / 8.0)
            * (
                1.0
                - 160.0 * self.eta
                + 80.0 * delta * k_a
                + 80.0 * (1.0 - 2.0 * self.eta) * k_s
            )
            * (chi_a.dot(L) ** 2)
        )
        psi_S_2PN -= (
            (5.0 / 4.0)
            * (delta + 80.0 * delta * k_s + 80.0 * (1.0 - 2.0 * self.eta) * k_a)
            * chi_s.dot(L)
            * chi_a.dot(L)
        )
        psi_S_2PN *= v4

        psi_S_25PN = (
            -(732985.0 / 2268.0 - 24260.0 * self.eta / 81.0 - 340.0 * eta2 / 9.0)
            * chi_s.dot(L)
            - (732985.0 / 2268.0 + 140.0 * self.eta / 9.0) * delta * chi_a.dot(L)
        ) * v5
        psi_S_25PN_log = 3.0 * psi_S_25PN * np.log(v / vlso)

        psi_S_3PN = (2270.0 / 3.0 - 520.0 * self.eta) * np.pi * chi_s.dot(L) + (
            2270.0 * np.pi / 3.0
        ) * delta * chi_a.dot(L)
        psi_S_3PN += (
            (
                (26015.0 / 14.0 - 88510.0 * self.eta / 21.0 - 480.0 * eta2) * k_a
                + delta
                * (
                    -1344475.0 / 1008.0
                    + 745.0 * self.eta / 18.0
                    + (26015.0 / 14.0 - 1495.0 * self.eta / 3.0) * k_s
                )
            )
            * chi_s.dot(L)
            * chi_a.dot(L)
        )
        psi_S_3PN += (
            -1344475.0 / 2016.0
            + 829705.0 * self.eta / 504.0
            + 3415.0 * eta2 / 9.0
            + (26015.0 / 28.0 - 44255.0 * self.eta / 21.0 - 240.0 * eta2) * k_s
            + delta * (26015.0 / 28.0 - 1495.0 * self.eta / 6.0) * k_a
        ) * (chi_s.dot(L)) ** 2
        psi_S_3PN += (
            -1344475.0 / 2016.0
            + 267815.0 * self.eta / 252.0
            - 240.0 * eta2
            + (26015.0 / 28.0 - 44255.0 * self.eta / 21.0 - 240.0 * eta2) * k_s
            + delta * (26015.0 / 28.0 - 1495.0 * self.eta / 6.0) * k_a
        ) * (chi_a.dot(L)) ** 2
        psi_S_3PN *= v6

        psi_S_35PN = (
            -25150083775.0 / 3048192.0
            + 10566655595.0 * self.eta / 762048.0
            - 1042165 * eta2 / 3024.0
            + 5345.0 * eta3 / 36.0
        ) * chi_s.dot(L)
        psi_S_35PN += (
            (
                -25150083775.0 / 3048192.0
                + 26804935.0 * self.eta / 6048.0
                - 1985.0 * eta2 / 48.0
            )
            * delta
            * chi_a.dot(L)
        )
        psi_S_35PN += (
            265.0 / 24.0
            + 4035.0 * self.eta / 2.0
            - 20.0 * eta2 / 3.0
            + (3110.0 / 3.0 - 10250.0 * self.eta / 3.0 + 40.0 * eta2) * k_s
            - 440.0 * (1.0 - 3.0 * self.eta) * lambda_s
            + delta
            * (
                (3110.0 / 3.0 - 4030.0 * self.eta / 3.0) * k_a
                - 440.0 * (1.0 - self.eta) * lambda_a
            )
        ) * (chi_s.dot(L)) ** 3
        psi_S_35PN += (
            (3110.0 / 3.0 - 8470.0 * self.eta / 3.0) * k_a
            - 440.0 * (1.0 - 3.0 * self.eta) * lambda_a
            + delta
            * (
                265.0 / 24.0
                - 2070.0 * self.eta
                + (3110.0 / 3.0 - 750.0 * self.eta) * k_s
                - 440.0 * (1 - self.eta) * lambda_s
            )
        ) * (chi_a.dot(L)) ** 3
        psi_S_35PN += (
            (3110.0 - 28970.0 * self.eta / 3.0 + 80.0 * eta2) * k_a
            - 1320.0 * (1.0 - 3.0 * self.eta) * lambda_a
            + delta
            * (
                265.0 / 8.0
                + 12055.0 * self.eta / 6.0
                + (3110.0 - 10310.0 * self.eta / 3.0) * k_s
                - 1320.0 * (1.0 - self.eta) * lambda_s
            )
        ) * (chi_s.dot(L) ** 2 * chi_a.dot(L))
        psi_S_35PN += (
            265.0 / 8.0
            - 6500.0 * self.eta / 3.0
            + 40.0 * eta2
            + (3110.0 - 27190.0 * self.eta / 3.0 + 40.0 * eta2) * k_s
            - 1320.0 * (1.0 - 3 * self.eta) * lambda_s
            + delta
            * (
                (3110.0 - 8530.0 * self.eta / 3.0) * k_a
                - 1320.0 * (1.0 - self.eta) * lambda_a
            )
        ) * (chi_a.dot(L) ** 2 * chi_s.dot(L))
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

        return 3.0 / 128.0 / self.eta / v5 * (psi_NS + psi_S)

    def hf3hPN(self):
        """
        Computes the Taylor F2 Frequency domain strain waveform with non-standard spin induced quadrupoole moment for object two
        Note that this waveform assumes object 1 is a BH therefore uses the chi*M_total relation to find C
        Note that this waveform also assumes that object one is the more massive. Therefore the more massive object is always considered a BH
        Returns:
        --------
        Strain (array):
        """
        pre = 3.6686934875530996e-19  # (GN*Msun/c^3)^(5/6)/Hz^(7/6)*c/Mpc/sec
        Mchirp = self.Mt * self.eta ** 0.6
        A0 = (
            Mchirp ** (5.0 / 6.0)
            / (self.frequencies + 1e-100) ** (7.0 / 6.0)
            / self.distance
            / np.pi ** (2.0 / 3.0)
            * np.sqrt(5.0 / 24.0)
        )

        Phi = self.Phif35PN()
        return pre * A0, Phi

    def hp_hc(self):
        """
        Decompose into plus and cross polarization
        """
        self.amplitude, self.phase = self.hf3hPN()
        self.phase_p = self.phase
        self.phase_c = self.phase_p + np.pi / 2.0
        self.hp = (
            self.amplitude
            * np.exp(-1.0j * self.phase_p)
            * ((1.0 + np.square(np.cos(self.inclination))) / 2.0)
        )
        self.hc = (
            self.amplitude * np.exp(-1.0j * self.phase_c) * np.cos(self.inclination)
        )