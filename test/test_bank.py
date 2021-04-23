from diffbank.bank import Bank
import jax.numpy as jnp
from jax import random
from diffbank.waveforms.threePN_simple import amp, Psi
from diffbank.utils import get_m1_m2_sampler, Sn_func


def test_save_load():
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
        jnp.array(1.0),
        0.05,
        0.99,
        "save_test",
    )
    bank.density_max = jnp.array(100.0)
    bank.n_templates = jnp.array(5, dtype=int)
    bank.templates = jnp.array([[5.0, 4.0], [7.0, 79.0]])
    bank.effectualnesses = jnp.array([1.0, 0.2, 3.0, 6.0, 1.0, 45.0, 9.0])
    bank.save_bank()
    bank_str = str(bank)
    print(bank_str)

    loaded_bank = Bank.load_bank("save_test.npz", amp, Psi, Sn, sampler)
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


def test_generation():
    key = random.PRNGKey(84)

    fs = jnp.linspace(20.0, 2000.0, 300)
    m_range = (2.0, 3.0)
    sampler = get_m1_m2_sampler(m_range, m_range)
    bank = Bank(
        amp,
        Psi,
        fs,
        Sn_func,
        sampler,
        naive_vol=jnp.array(4.0),
        m_star=1 - 0.95,
        eta=0.99,
        name="3PN",
    )

    bank.density_max = jnp.array(bank.get_density(jnp.array([m_range[0], m_range[0]])))

    bank.compute_n_templates(key, 1000)
    _, key = random.split(key)

    assert bank.n_templates > 0 and bank.n_templates < 1e5
    print(bank.n_templates)

    bank.fill_bank(key)
    _, key = random.split(key)
    assert len(bank.templates) == bank.n_templates

    # Make sure templates are in bounds
    for m1, m2 in bank.templates:
        assert m1 >= m_range[0] and m1 <= m_range[1]
        assert m2 >= m_range[0] and m2 <= m_range[1]


if __name__ == "__main__":
    test_save_load()
    test_generation()