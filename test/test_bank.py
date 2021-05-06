from diffbank.bank import Bank
from diffbank.utils import Sn_func, get_m1_m2_sampler
from diffbank.waveforms.threePN_simple import Psi, amp
from jax import random
import jax.numpy as jnp


def test_save_load():
    """
    Test saving and loading.
    """
    amp = lambda f, t: None
    Psi = lambda f, t: None
    Sn = lambda f: None
    sampler = lambda k, n: jnp.array([0.0, 0.9])
    is_in_bounds = lambda t: jnp.array(1.0)
    bank = Bank(
        amp,
        Psi,
        jnp.array(0.0),
        Sn,
        sampler,
        jnp.array(1.0),
        0.05,
        0.99,
        is_in_bounds,
        "save_test",
    )
    bank.density_max = jnp.array(100.0)
    bank.n_templates = jnp.array(5, dtype=int)
    bank.templates = jnp.array([[5.0, 4.0], [7.0, 79.0]])
    bank.effectualnesses = jnp.array([1.0, 0.2, 3.0, 6.0, 1.0, 45.0, 9.0])
    bank.save()
    bank_str = str(bank)
    print(bank_str)

    loaded_bank = Bank.load("save_test.npz", amp, Psi, Sn, sampler, is_in_bounds)
    loaded_bank_str = str(loaded_bank)
    print(loaded_bank_str)

    assert bank.amp == loaded_bank.amp
    assert bank.Psi == loaded_bank.Psi
    assert jnp.all(bank.fs == loaded_bank.fs)
    assert bank.Sn == loaded_bank.Sn
    assert bank.sampler == loaded_bank.sampler
    assert bank.naive_vol == loaded_bank.naive_vol
    assert bank.m_star == loaded_bank.m_star
    assert bank.eta == loaded_bank.eta
    assert bank.name == loaded_bank.name
    assert bank.density_max == loaded_bank.density_max
    assert jnp.all(bank.n_templates == loaded_bank.n_templates)
    assert jnp.all(bank.templates == loaded_bank.templates)
    assert jnp.all(bank.effectualnesses == loaded_bank.effectualnesses)
    assert bank.dim == loaded_bank.dim


def test_generation():
    """
    Make sure template bank generation works.
    """
    key = random.PRNGKey(84)

    fs = jnp.linspace(20.0, 2000.0, 300)
    m_range = (2.0, 3.0)
    sampler = get_m1_m2_sampler(m_range, m_range)
    is_in_bounds = lambda _: jnp.array(1.0)  # doesn't matter for this test
    bank = Bank(
        amp,
        Psi,
        fs,
        Sn_func,
        sampler,
        naive_vol=jnp.array(4.0),
        m_star=1 - 0.95,
        eta=0.99,
        is_in_bounds=is_in_bounds,
        name="3PN",
    )

    bank.density_max = jnp.array(bank.get_density(jnp.array([m_range[0], m_range[0]])))

    key, subkey = random.split(key)
    bank.compute_template_frac_in_bounds(subkey, 20, 5)
    assert bank.frac_in_bounds == 1.0

    key, subkey = random.split(key)
    bank.compute_n_templates(subkey, 1000)

    assert bank.n_templates > 0 and bank.n_templates < 1e5

    key, subkey = random.split(key)
    bank.fill_bank(subkey)
    assert len(bank.templates) == bank.n_templates

    # Make sure templates are in bounds
    for m1, m2 in bank.templates:
        assert m1 >= m_range[0] and m1 <= m_range[1]
        assert m2 >= m_range[0] and m2 <= m_range[1]


if __name__ == "__main__":
    test_save_load()
    test_generation()
