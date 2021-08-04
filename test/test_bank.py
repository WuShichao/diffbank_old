from diffbank.bank import Bank
from diffbank.utils import get_m1_m2_sampler
from diffbank.noise import Sn_aLIGO as Sn_func
from diffbank.waveforms.threePN_simple import Psi, amp
from jax import random
import jax.numpy as jnp

"""
Make sure the bank's saving/loading, generation and effectualness calculation
functionality works.
"""


def test_save_load():
    """
    Test saving and loading.
    """
    amp = lambda f, t: None
    Psi = lambda f, t: None
    Sn = lambda f: None
    sampler = lambda k, n: jnp.array([0.0, 0.9])
    bank = Bank(
        amp,
        Psi,
        jnp.array(0.0),
        Sn,
        sampler,
        0.05,
        0.99,
        "save_test",
    )
    bank.density_max = jnp.array(100.0)
    bank.n_templates = jnp.array(2, dtype=int)
    bank.templates = jnp.array([[5.0, 4.0], [7.0, 79.0]])
    bank.effectualnesses = jnp.array([1.0, 0.2, 3.0, 6.0, 1.0, 45.0, 9.0])
    bank.effectualness_points = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
    bank.eta_est = 0.92
    bank.eta_est_err = 0.001
    bank.save()
    bank_str = str(bank)
    print(bank_str)

    loaded_bank = Bank.load("save_test.npz", amp, Psi, Sn, sampler)
    loaded_bank_str = str(loaded_bank)
    print(loaded_bank_str)

    # Computed variables
    assert bank.density_max == loaded_bank.density_max
    assert bank.n_templates == loaded_bank.n_templates
    assert jnp.all(bank.templates == loaded_bank.templates)
    assert jnp.all(bank.effectualness_points == loaded_bank.effectualness_points)
    assert jnp.all(bank.effectualnesses == loaded_bank.effectualnesses)
    assert bank.eta_est == bank.eta_est
    assert bank.eta_est_err == bank.eta_est_err
    assert bank.dim == loaded_bank.dim
    # Provided variables
    assert jnp.all(bank.fs == loaded_bank.fs)
    assert bank.m_star == loaded_bank.m_star
    assert bank.eta == loaded_bank.eta
    assert bank.name == loaded_bank.name
    # Functions
    assert bank.amp is loaded_bank.amp
    assert bank.Psi is loaded_bank.Psi
    assert bank.Sn is loaded_bank.Sn
    assert bank.sampler is loaded_bank.sampler


def test_random():
    """
    Make sure template bank generation works.
    """
    key = random.PRNGKey(84)

    fs = jnp.linspace(20.0, 2000.0, 300)
    m_range = (2.5, 3.0)
    sampler = get_m1_m2_sampler(m_range, m_range)
    bank = Bank(
        amp,
        Psi,
        fs,
        Sn_func,
        sampler,
        m_star=1 - 0.95,
        eta=0.99,
        name="3PN",
    )

    theta_dmax = jnp.array([m_range[1], m_range[0]])
    bank.density_max = bank.density_fun(theta_dmax)

    for kind in ["random", "stochastic"]:
        print(f"Testing {kind} bank")
        key, subkey = random.split(key)
        bank.fill_bank(subkey, kind, r=0.1)  # coarse bank
        assert len(bank.templates) == bank.n_templates

        # Make sure templates are in bounds
        for m1, m2 in bank.templates:
            assert m1 >= m_range[0] and m1 <= m_range[1]
            assert m2 >= m_range[0] and m2 <= m_range[1]

        key, subkey = random.split(key)
        bank.calc_bank_effectualness(subkey, 100)
        # Rough sanity checks
        assert bank.eta_est > 0.5
        assert bank.eta_est_err < 0.5
        print(f"eta = {bank.eta_est:.3f} +/- {bank.eta_est_err:.3f}\n")


if __name__ == "__main__":
    test_save_load()
    test_random()
