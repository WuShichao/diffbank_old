import os
from typing import Callable, Optional, Set, Tuple, Union

import jax
from jax import random
from jax._src.prng import PRNGKeyArray
import jax.numpy as jnp

from .metric import get_density, get_g, get_gam
from .utils import (
    _get_effectualness,
    est_ratio_max,
    gen_bank_random,
    gen_bank_stochastic,
    gen_template_rejection,
    gen_templates_rejection,
    get_bank_effectualness,
    get_eff_pads,
)

Array = jnp.ndarray


class Bank:
    """
    Template bank container.
    """

    computed_vars: Set[str] = set(
        [
            "ratio_max",
            "n_templates",
            "templates",
            "effectualness_points",
            "effectualnesses",
            "eta_est",
            "eta_est_err",
            "_eff_pad_low",
            "_eff_pad_high",
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
        amp: Callable[[Array, Array], Array],
        Psi: Callable[[Array, Array], Array],
        fs: Array,
        Sn: Callable[[Array], Array],
        m_star: Union[float, Array],
        eta: Union[float, Array],
        sample_base: Callable[[PRNGKeyArray, int], Array],
        density_fun_base: Callable[[Array], Array] = lambda _: jnp.array(1.0),
        name: str = "test",
    ):
        # Validation
        if len(fs) <= 2:
            # Required for padding to work
            raise ValueError("length of frequency array must be at least three")
        if m_star > 1 or m_star < 0:
            raise ValueError("m_star must be in (0, 1)")
        if eta > 1 or eta < 0:
            raise ValueError("eta must be in (0, 1)")

        self.amp = amp
        self.Psi = Psi
        self.fs = fs
        self.Sn = Sn
        self.m_star = m_star
        self.eta = eta
        self.sample_base = sample_base
        self.density_fun_base = density_fun_base
        self.name = name

        self.ratio_max: Optional[Array] = None
        self.n_templates: Optional[int] = None
        self.templates: Optional[Array] = None
        self.effectualness_points: Optional[Array] = None
        self.effectualnesses: Optional[Array] = None
        self.eta_est: Optional[Array] = None
        self.eta_est_err: Optional[Array] = None

        # Padding for accurate effectualness calculation
        # Length of padded array
        self._eff_pad_low, self._eff_pad_high = get_eff_pads(self.fs)

        # Key doesn't matter
        self._dim = self.sample_base(random.PRNGKey(1), 1).shape[-1]

    def __str__(self):
        return f"Bank(m_star={float(self.m_star)}, eta={float(self.eta)}, dim={self.dim}, name='{self.name}')"

    def __repr__(self):
        return str(self)  # for now

    @property
    def dim(self) -> int:
        return self._dim

    def effectualness_fun(self, theta1: Array, theta2: Array) -> Array:
        """
        Effectualness between two points
        """
        return _get_effectualness(
            theta1,
            theta2,
            self.amp,
            self.Psi,
            self.fs,
            self.Sn,
            self._eff_pad_low,
            self._eff_pad_high,
        )

    def density_fun(self, theta: Array) -> Array:
        """
        Template density, sqrt(|g|).
        """
        return get_density(theta, self.amp, self.Psi, self.fs, self.Sn)

    def g_fun(self, theta) -> Array:
        """
        Parameter space metric maximized over extrinsic parameters t_0 and
        Phi_0, g.
        """
        return get_g(theta, self.amp, self.Psi, self.fs, self.Sn)

    def gam_fun(self, theta) -> Array:
        """
        t_0 and parameter space metric maximized over the extrinsic parameter
        Phi_0, gamma.
        """
        return get_gam(theta, self.amp, self.Psi, self.fs, self.Sn)

    def est_ratio_max(
        self,
        key: PRNGKeyArray,
        n_iter: int = 1000,
        n_init: int = 200,
        show_progress: bool = True,
    ) -> Tuple[Array, Array]:
        return est_ratio_max(
            key,
            self.density_fun,
            self.sample_base,
            self.density_fun_base,
            n_iter,
            n_init,
            show_progress,
        )

    def gen_template_rejection(self, key: PRNGKeyArray) -> Array:
        if self.ratio_max is None:
            raise ValueError(
                "must set bank's 'ratio_max' attribute to an estimate of the"
                " maximum value of sqrt(|g|)"
            )
        return gen_template_rejection(
            key,
            self.ratio_max,
            self.density_fun,
            self.sample_base,
            self.density_fun_base,
        )

    def gen_templates_rejection(self, key: PRNGKeyArray, n_templates) -> Array:
        """
        Generates templates using rejection sampling.
        """
        if self.ratio_max is None:
            raise ValueError(
                "must set bank's 'ratio_max' attribute to an estimate of the"
                " maximum value of sqrt(|g|)"
            )
        return gen_templates_rejection(
            key, n_templates, self.ratio_max, self.density_fun, self.sample_base
        )

    def fill_bank(
        self,
        key: PRNGKeyArray,
        method="random",
        n_eff: int = 1000,
        show_progress: bool = True,
    ):
        """
        Fills the bank with the required number of templates. See docs for
        `gen_bank`.
        """
        if self.ratio_max is None:
            raise ValueError(
                "must set bank's 'ratio_max' attribute to an estimate of the"
                " maximum value of sqrt(|g|)"
            )

        if method == "random":
            self.templates, _ = gen_bank_random(
                key,
                1 - self.m_star,
                self.eta,
                self.effectualness_fun,
                self.ratio_max,
                self.density_fun,
                self.sample_base,
                self.density_fun_base,
                n_eff=n_eff,
                show_progress=show_progress,
            )
            self.n_templates = len(self.templates)
        elif method == "stochastic":
            propose_template = jax.jit(lambda key: self.sample_base(key, 1)[0])
            sample_eff_pt = jax.jit(self.gen_template_rejection)
            self.templates, _ = gen_bank_stochastic(
                key,
                1 - self.m_star,
                self.eta,
                self.effectualness_fun,
                propose_template,
                sample_eff_pt,
                n_eff=n_eff,
                show_progress=show_progress,
            )
            self.n_templates = len(self.templates)

    def calc_bank_effectualness(
        self, key: PRNGKeyArray, n: int, show_progress: bool = True
    ):
        """
        Computes effectualnesses for a sample of parameter points, adds the
        points as the attribute ``effectualness_points`` and uses this to
        estimate ``eta``. The estimate of ``eta`` and the associated error are
        accessible through the ``eta_est`` and ``eta_est_err`` properties.

        The points are sampled following the metric density.
        """
        if self.templates is None:
            raise RuntimeError("cannot calculate effectualness of an empty bank")

        sample_eff_pt = jax.jit(self.gen_template_rejection)
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
            sample_eff_pt,
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
        amp: Callable[[Array, Array], Array],
        Psi: Callable[[Array, Array], Array],
        Sn: Callable[[Array], Array],
        sample_base: Callable[[PRNGKeyArray, int], Array],
        ignore_key_errors: bool = False,
    ):
        """
        Loads template bank's non-function attributes from a npz file.
        """
        d = jnp.load(path, allow_pickle=True)["bank"].item()
        if d.keys() != cls.provided_vars | cls.computed_vars and not ignore_key_errors:
            raise ValueError("missing or extra keys in bank file")

        # Instantiate with provided variables and functions
        fn_kwargs = {
            "amp": amp,
            "Psi": Psi,
            "Sn": Sn,
            "sample_base": sample_base,
        }

        try:
            bank = cls(**{**fn_kwargs, **{name: d[name] for name in cls.provided_vars}})

            # Set computed variables
            for name in cls.computed_vars:
                setattr(bank, name, d[name])
        except KeyError as e:
            if not ignore_key_errors:
                raise e

        return bank
