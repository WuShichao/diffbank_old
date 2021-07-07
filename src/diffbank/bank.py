import os
from typing import Callable, Optional, Set, Union

# import warnings

import jax
import jax.numpy as jnp

# import numpy as np
# from tqdm.auto import trange

from .metric import get_density, get_g, get_gam
from .utils import (
    gen_templates_rejection,
    get_bank_effectualness,
    gen_bank,
)


class Bank:
    """
    Template bank container.
    """

    computed_vars: Set[str] = set(
        [
            "density_max",
            "frac_in_bounds",
            "frac_in_bounds_err",
            "n_templates",
            # "n_templates_err",
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
            # "naive_vol",
            # "naive_vol_err",
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
        self.frac_in_bounds: Optional[jnp.ndarray] = None
        self.frac_in_bounds_err: Optional[jnp.ndarray] = None
        self.n_templates: Optional[jnp.ndarray] = None
        self.templates: Optional[jnp.ndarray] = None
        self.effectualness_points: Optional[jnp.ndarray] = None
        self.effectualnesses: Optional[jnp.ndarray] = None
        self._eta_est: Optional[jnp.ndarray] = None
        self._eta_est_err: Optional[jnp.ndarray] = None

        # Key doesn't matter
        self._dim = self.sampler(jax.random.PRNGKey(1), 1).shape[-1]

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

    def fill_bank(self, key: jnp.ndarray, show_progress: bool = True):
        """
        Fills the bank with the required number of templates.
        """
        templates, eff_pts = gen_bank(
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
        )
        self.templates = templates
        self.n_templates = templates.shape[0]

    def calculate_bank_effectualness(
        self,
        key: jnp.ndarray,
        n: int,
        # points: Optional[jnp.ndarray] = None,
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

    # def compute_effectualnesses(
    #     self,
    #     key: Optional[jnp.ndarray] = None,
    #     n: Optional[int] = None,
    #     points: Optional[jnp.ndarray] = None,
    # ):
    #     """
    #     Computes effectualnesses for a sample of parameter points, adds the
    #     points as the attribute ``effectualness_points`` and uses this to
    #     estimate ``eta``. The estimate of ``eta`` and the associated error are
    #     accessible through the ``eta_est`` and ``eta_est_err`` properties.

    #     If ``points`` is provided, the effectualness will be computed using
    #     this sample. Otherwise ``n`` points will be drawn uniformly in proper
    #     volume using the PRNG key ``key``.
    #     """
    #     if points is not None:
    #         n = len(points)
    #     elif key is not None and n is not None:
    #         points = self.gen_templates_rejection(key, n)
    #     else:
    #         raise ValueError("either 'points' or both 'key' and 'n' must be provided")

    #     effectualnesses = np.zeros(n)

    #     get_eff_jit = jax.jit(
    #         lambda template, sample: get_effectualness(
    #             template, sample, self.amp, self.Psi, self.fs, self.Sn
    #         )
    #     )

    #     pbar = trange(n)

    #     try:
    #         for i in pbar:
    #             effectualnesses[i] = jax.lax.map(
    #                 lambda template: get_eff_jit(template, points[i]), self.templates
    #             ).max()

    #             # Keep track of eta estimate. Should be fast enough to keep in
    #             # the loop.
    #             self._eta_est = jnp.mean(effectualnesses[:i] > self.minimum_match)
    #             self._eta_est_err = jnp.std(
    #                 effectualnesses[:i] > self.minimum_match
    #             ) / jnp.sqrt(i)

    #             pbar.set_postfix_str(
    #                 f"eta ~ {self.eta_est:.3f} +/- {self.eta_est_err:.3f}"
    #             )
    #     finally:
    #         self.effectualnesses = jnp.array(effectualnesses)
    #         self.effectualness_points = points

    # def compute_template_frac_in_bounds(self, key, n_pts, n_per_pt):
    #     """
    #     Sets the fraction of points in the average template's metric ellipse
    #     that lie in the parameter space.
    #     """
    #     thetas = self.gen_templates_rejection(key, n_pts)
    #     self.frac_in_bounds, self.frac_in_bounds_err = get_template_frac_in_bounds(
    #         key, thetas, self.get_g, self.m_star, self.is_in_bounds, n_per_pt
    #     )

    # def compute_n_templates(self, key: jnp.ndarray, n_samples: Union[int, jnp.ndarray]):
    #     """
    #     Sets the number of templates for the bank.
    #     """
    #     if self.frac_in_bounds is not None and self.frac_in_bounds_err is not None:
    #         frac_in_bounds = self.frac_in_bounds
    #         frac_in_bounds_err = self.frac_in_bounds_err
    #     else:
    #         warnings.warn(
    #             "'frac_in_bounds' has not been computed, so number of "
    #             "templates will be underestimated",
    #             RuntimeWarning,
    #         )
    #         frac_in_bounds = jnp.array(1.0)
    #         frac_in_bounds_err = jnp.array(0.0)

    #     self.n_templates, self.n_templates_err = get_n_templates(
    #         key,
    #         n_samples,
    #         self.get_density,
    #         self.sampler,
    #         self.eta,
    #         self.m_star,
    #         self.naive_vol,
    #         frac_in_bounds,
    #         self.naive_vol_err,
    #         frac_in_bounds_err,
    #     )
