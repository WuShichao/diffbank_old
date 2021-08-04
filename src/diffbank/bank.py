import os
from typing import Callable, Set

import jax
from jax import random
import jax.numpy as jnp

from .metric import get_density, get_g, get_gam
from .utils import (
    gen_bank_random,
    gen_bank_stochastic,
    gen_template_rejection,
    gen_templates_rejection,
    get_bank_effectualness,
    get_effectualness,
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
            "eta_est",
            "eta_est_err",
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
        m_star: float,
        eta: float,
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

        self.density_max: jnp.ndarray = None
        self.n_templates: jnp.ndarray = None
        self.templates: jnp.ndarray = None
        self.effectualness_points: jnp.ndarray = None
        self.effectualnesses: jnp.ndarray = None
        self.eta_est: jnp.ndarray = None
        self.eta_est_err: jnp.ndarray = None

        # Key doesn't matter
        self._dim = self.sampler(random.PRNGKey(1), 1).shape[-1]

    def __str__(self):
        return f"Bank(m_star={float(self.m_star)}, eta={float(self.eta)}, dim={self.dim}, name='{self.name}')"

    def __repr__(self):
        return str(self)  # for now

    @property
    def dim(self) -> int:
        return self._dim

    def effectualness_fun(
        self, theta1: jnp.ndarray, theta2: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Effectualness between two points
        """
        return get_effectualness(theta1, theta2, self.amp, self.Psi, self.fs, self.Sn)

    def density_fun(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Template density, sqrt(|g|).
        """
        return get_density(theta, self.amp, self.Psi, self.fs, self.Sn)

    def g_fun(self, theta) -> jnp.ndarray:
        """
        Parameter space metric maximized over extrinsic parameters t_0 and
        Phi_0, g.
        """
        return get_g(theta, self.amp, self.Psi, self.fs, self.Sn)

    def gam_fun(self, theta) -> jnp.ndarray:
        """
        t_0 and parameter space metric maximized over the extrinsic parameter
        Phi_0, gamma.
        """
        return get_gam(theta, self.amp, self.Psi, self.fs, self.Sn)

    def gen_template_rejection(self, key: jnp.ndarray) -> jnp.ndarray:
        if self.density_max is None:
            raise ValueError(
                "must set bank's 'density_max' attribute to an estimate of the"
                " maximum value of sqrt(|g|)"
            )
        return gen_template_rejection(
            key, self.density_max, self.density_fun, self.sampler
        )

    def gen_templates_rejection(self, key: jnp.ndarray, n_templates) -> jnp.ndarray:
        """
        Generates templates using rejection sampling.
        """
        if self.density_max is None:
            raise ValueError(
                "must set bank's 'density_max' attribute to an estimate of the"
                " maximum value of sqrt(|g|)"
            )
        return gen_templates_rejection(
            key, n_templates, self.density_max, self.density_fun, self.sampler
        )

    def fill_bank(
        self,
        key: jnp.ndarray,
        method="random",
        r: float = 3,
        show_progress: bool = True,
    ):
        """
        Fills the bank with the required number of templates. See docs for
        `gen_bank`.
        """
        if self.density_max is None:
            raise ValueError(
                "must set bank's 'density_max' attribute to an estimate of the"
                " maximum value of sqrt(|g|)"
            )

        if method == "random":
            self.templates, _ = gen_bank_random(
                key,
                1 - self.m_star,
                self.eta,
                self.effectualness_fun,
                self.density_max,
                self.sampler,
                self.density_fun,
                r=r,
                show_progress=show_progress,
            )
            self.n_templates = len(self.templates)
        elif method == "stochastic":
            propose_template = jax.jit(lambda key: self.sampler(key, 1)[0])
            eff_pt_sampler = jax.jit(self.gen_template_rejection)
            self.templates, _ = gen_bank_stochastic(
                key,
                1 - self.m_star,
                self.eta,
                self.effectualness_fun,
                propose_template,
                eff_pt_sampler,
                r,
                show_progress,
            )
            self.n_templates = len(self.templates)

    def calc_bank_effectualness(
        self, key: jnp.ndarray, n: int, show_progress: bool = True
    ):
        """
        Computes effectualnesses for a sample of parameter points, adds the
        points as the attribute ``effectualness_points`` and uses this to
        estimate ``eta``. The estimate of ``eta`` and the associated error are
        accessible through the ``eta_est`` and ``eta_est_err`` properties.

        The points are sampled following the metric density.
        """
        eff_pt_sampler = jax.jit(self.gen_template_rejection)
        (
            self.effectualnesses,
            self.effectualness_points,
            self.eta_est,
            self.eta_est_err,
        ) = get_bank_effectualness(
            key,
            1 - self.m_star,
            self.templates,
            self.effectualness_fun,
            eff_pt_sampler,
            n,
            show_progress,
        )

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
        Loads template bank's non-function attributes from a npz file.
        """
        d = jnp.load(path, allow_pickle=True)["bank"].item()
        if d.keys() != cls.provided_vars | cls.computed_vars:
            raise ValueError("missing or extra keys in bank file")

        # Instantiate with provided variables and functions
        fn_kwargs = {
            "amp": amp,
            "Psi": Psi,
            "Sn": Sn,
            "sampler": sampler,
        }
        bank = cls(**{**fn_kwargs, **{name: d[name] for name in cls.provided_vars}})

        # Set computed variables
        for name in cls.computed_vars:
            setattr(bank, name, d[name])

        return bank
