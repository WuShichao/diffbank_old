import os
from typing import Callable, Optional, Set, Union

import jax
from jax import random
import jax.numpy as jnp

from .metric import get_density, get_g, get_gam
from .utils import (
    gen_bank_effpoints,
    gen_bank_stochastic,
    gen_templates_rejection,
    get_bank_effectualness,
)


class Bank:
    """
    Template bank container.
    """

    computed_vars: Set[str] = set(
        [
            "density_max",
            "n_templates",
            "templates",
            "effectualness_points",
            "effectualnesses",
            "_eta_est",
            "_eta_est_err",
            "_dim",
        ]
    )
    provided_vars: Set[str] = set(
        [
            "fs",
            "m_star",
            "eta",
            "name",
        ]
    )

    def __init__(
        self,
        amp: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        Psi: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        fs: jnp.ndarray,
        Sn: Callable[[jnp.ndarray], jnp.ndarray],
        sampler: Callable[[jnp.ndarray, int], jnp.ndarray],
        m_star: Union[float, jnp.ndarray],
        eta: Union[float, jnp.ndarray],
        name: str = "test",
    ):
        self.amp = amp
        self.Psi = Psi
        self.fs = fs
        self.Sn = Sn
        self.sampler = sampler
        self.m_star = m_star
        self.eta = eta
        self.name = name

        self.density_max: Optional[jnp.ndarray] = None
        self.n_templates: Optional[jnp.ndarray] = None
        self.templates: Optional[jnp.ndarray] = None
        self.effectualness_points: Optional[jnp.ndarray] = None
        self.effectualnesses: Optional[jnp.ndarray] = None
        self._eta_est: Optional[jnp.ndarray] = None
        self._eta_est_err: Optional[jnp.ndarray] = None

        # Key doesn't matter
        self._dim = self.sampler(random.PRNGKey(1), 1).shape[-1]

    def __str__(self):
        return f"Bank(m_star={float(self.m_star)}, eta={float(self.eta)}, dim={self.dim}, name='{self.name}')"

    def __repr__(self):
        return str(self)  # for now

    @property
    def minimum_match(self):
        return 1 - self.m_star

    @property
    def m_star(self):
        return self._m_star

    @m_star.setter
    def m_star(self, m_star):
        if m_star < 0 or m_star > 1:
            raise ValueError("m_star must be a scalar in (0, 1)")
        self._m_star = m_star

    @property
    def eta(self):
        return self._eta

    @property
    def eta_est(self):
        return self._eta_est

    @property
    def eta_est_err(self):
        return self._eta_est_err

    @eta.setter
    def eta(self, eta):
        if eta < 0 or eta > 1:
            raise ValueError("eta must be a scalar in (0, 1)")
        self._eta = eta

    @property
    def dim(self) -> int:
        return self._dim

    def get_density(self, theta) -> jnp.ndarray:
        """
        Template density, sqrt(|g|).
        """
        return get_density(theta, self.amp, self.Psi, self.fs, self.Sn)

    def get_g(self, theta) -> jnp.ndarray:
        """
        Parameter space metric maximized over extrinsic parameters t_0 and
        Phi_0, g.
        """
        return get_g(theta, self.amp, self.Psi, self.fs, self.Sn)

    def get_gam(self, theta) -> jnp.ndarray:
        """
        t_0 and parameter space metric maximized over the extrinsic parameter
        Phi_0, gamma.
        """
        return get_gam(theta, self.amp, self.Psi, self.fs, self.Sn)

    def gen_templates_rejection(self, key: jnp.ndarray, n_templates) -> jnp.ndarray:
        """
        Generates templates using rejection sampling.
        """
        if self.density_max is None:
            raise ValueError(
                "Must set bank's 'density_max' attribute to an estimate of the"
                " maximum value of sqrt(|g|)"
            )

        return gen_templates_rejection(
            key, self.density_max, n_templates, self.get_density, self.sampler
        )

    def fill_bank(
        self,
        key: jnp.ndarray,
        show_progress: bool = True,
        method="eff_points",
        r: float = 1,
    ):
        """
        Fills the bank with the required number of templates. See docs for
        `gen_bank`.
        """
        if method == "eff_points":
            templates, _ = gen_bank_effpoints(
                key,
                self.density_max,
                self.sampler,
                self.amp,
                self.Psi,
                self.fs,
                self.Sn,
                self.minimum_match,
                self.eta,
                show_progress=show_progress,
                r=r,
            )
            self.templates = templates
            self.n_templates = templates.shape[0]

        elif method == "stochastic":
            templates = gen_bank_stochastic(
                key,
                self.density_max,
                self.sampler,
                self.amp,
                self.Psi,
                self.fs,
                self.Sn,
                self.minimum_match,
                self.eta,
            )
            self.templates = templates
            self.n_templates = templates.shape[0]

    def calculate_bank_effectualness(
        self,
        key: jnp.ndarray,
        n: int,
    ):
        """
        Computes effectualnesses for a sample of parameter points, adds the
        points as the attribute ``effectualness_points`` and uses this to
        estimate ``eta``. The estimate of ``eta`` and the associated error are
        accessible through the ``eta_est`` and ``eta_est_err`` properties.

        The points are sampled following the metric volume i.e. using the same
        rejection sampling used to make the bank
        """
        effs, eff_pts, eta, eta_err = get_bank_effectualness(
            key,
            self.templates,
            self.amp,
            self.Psi,
            self.fs,
            self.Sn,
            self.minimum_match,
            n,
            base_dist=self.sampler,
            density_max=self.density_max,
        )
        self.effectualnesses = effs
        self.effectualness_points = eff_pts
        self._eta_est = eta
        self._eta_est_err = eta_err

    def save(self, path: str = ""):
        """
        Saves template bank non-function attributes to a npz file.
        """
        d = {k: getattr(self, k) for k in self.provided_vars | self.computed_vars}
        jnp.savez(os.path.join(path, f"{self.name}.npz"), bank=d)

    @classmethod
    def load(
        cls,
        path: str,
        amp: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        Psi: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        Sn: Callable[[jnp.ndarray], jnp.ndarray],
        sampler: Callable[[jnp.ndarray, int], jnp.ndarray],
    ):
        """
        Loads template bank non-function attributes from a npz file.
        """
        d = jnp.load(path, allow_pickle=True)["bank"].item()
        if d.keys() != cls.provided_vars | cls.computed_vars:
            raise ValueError("missing or extra keys in bank file")

        fn_kwargs = {
            "amp": amp,
            "Psi": Psi,
            "Sn": Sn,
            "sampler": sampler,
        }
        bank = cls(**{**fn_kwargs, **{name: d[name] for name in cls.provided_vars}})

        for name in cls.computed_vars:
            setattr(bank, name, d[name])

        return bank
