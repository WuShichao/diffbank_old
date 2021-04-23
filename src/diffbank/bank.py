import os
from typing import Callable, NamedTuple, Optional, Union

import numpy as np

import jax
import jax.numpy as jnp
from tqdm.auto import trange

from .metric import get_density
from .utils import (
    gen_templates_rejection,
    get_effectualness,
    get_n_templates,
)


class Bank:
    """
    Template bank container.
    """

    def __init__(
        self,
        amp: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        Psi: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        fs: jnp.ndarray,
        Sn: Callable[[jnp.ndarray], jnp.ndarray],
        sampler: Callable[[jnp.ndarray, int], jnp.ndarray],
        naive_vol: jnp.ndarray,
        m_star: Union[float, jnp.ndarray],
        eta: Union[float, jnp.ndarray],
        name: str = "test",
    ):
        self.amp = amp
        self.Psi = Psi
        self.fs = fs
        self.Sn = Sn
        self.sampler = sampler
        self.naive_vol = naive_vol
        self.m_star = m_star
        self.eta = eta
        self.name = name

        self.density_max: jnp.ndarray = None
        self.n_templates: jnp.ndarray = None
        self.templates: jnp.ndarray = None
        self.effectualnesses: jnp.ndarray = None

    @property
    def dim(self) -> jnp.ndarray:
        return self.sampler(jax.random.PRNGKey(1), 1).shape[-1]  # key doesn't matter

    def density_fun(self, theta) -> jnp.ndarray:
        """
        Template density, sqrt(|g|).
        """
        return get_density(theta, self.amp, self.Psi, self.fs, self.Sn)

    def compute_n_templates(self, key: jnp.ndarray, n_samples: Union[int, jnp.ndarray]):
        """
        Sets the number of templates for the bank.
        """
        self.n_templates = get_n_templates(
            key,
            self.naive_vol,
            n_samples,
            self.density_fun,
            self.sampler,
            self.eta,
            self.m_star,
        )

    def gen_templates_rejection(self, key: jnp.ndarray, n_templates) -> jnp.ndarray:
        """
        Generates templates using rejection sampling.
        """
        return gen_templates_rejection(
            key, self.density_max, n_templates, self.density_fun, self.sampler
        )

    def fill_bank(self, key: jnp.ndarray):
        """
        Fills the bank with the required number of templates.
        """
        self.templates = self.gen_templates_rejection(key, self.n_templates)

    def save_bank(self, path: str = ""):
        """
        Saves template bank non-function attributes to a npz file.
        """
        d = {
            "fs": self.fs,
            "naive_vol": self.naive_vol,
            "m_star": self.m_star,
            "eta": self.eta,
            "name": self.name,
            "density_max": self.density_max,
            "n_templates": self.n_templates,
            "templates": self.templates,
            "effectualnesses": self.effectualnesses,
        }
        jnp.savez(os.path.join(path, f"{self.name}.npz"), bank=d)

    @classmethod
    def load_bank(
        cls,
        amp: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        Psi: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        Sn: Callable[[jnp.ndarray], jnp.ndarray],
        sampler: Callable[[jnp.ndarray, int], jnp.ndarray],
        path: str,
    ):
        """
        Loads template bank non-function attributes from a npz file.
        """
        d = jnp.load(path, allow_pickle=True)["bank"].item()
        bank = cls(
            amp,
            Psi,
            d["fs"],
            Sn,
            sampler,
            d["naive_vol"],
            d["m_star"],
            d["eta"],
            d["name"],
        )
        bank.density_max = d["density_max"]
        bank.n_templates = d["n_templates"]
        bank.templates = d["templates"]
        bank.effectualnesses = d["effectualnesses"]
        return bank

    def compute_effectualnesses(self, points: jnp.ndarray):
        """
        Computes effectualnesses for a sample of parameter points.
        """
        n_test = len(points)
        effectualnesses = np.zeros(n_test)

        get_eff_jit = jax.jit(
            lambda template, sample: get_effectualness(
                template, sample, self.amp, self.Psi, self.fs, self.Sn
            )
        )

        for i in trange(n_test):
            effectualnesses[i] = jax.lax.map(
                lambda template: get_eff_jit(template, points[i]), self.templates
            ).max()

        self.effectualnesses = jnp.array(effectualnesses)
