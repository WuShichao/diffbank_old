from jax import jit, random, vmap
from jax.lax import map as lax_map
import jax.numpy as jnp
import matplotlib.pyplot as plt

from diffbank.bank import Bank, fill_bank, save_bank, update_n_templates
from diffbank.metric import get_density
from diffbank.utils import Sn_func, get_m1_m2_sampler, update_fields
from diffbank.waveforms.threePN_simple import amp, Psi


if __name__ == "__main__":
    key = random.PRNGKey(32)

    # Set up the bank
    m_star = 1 - 0.95
    eta = 0.999
    m1_range = jnp.array([1.0, 3.0])
    m2_range = jnp.array([1.0, 3.0])
    sampler = get_m1_m2_sampler(m1_range, m2_range)
    f = jnp.linspace(10, 500, 10000)
    bank = Bank(amp, Psi, f, Sn_func, sampler, m_star, eta, name="test-35PN")

    print("Estimating maximum density")
    # Simple estimate of maximum density, which appears to lie along m2 boundary
    density_fun = jit(lambda theta: get_density(theta, bank.amp, bank.Psi, bank.f, bank.Sn))
    m1s = jnp.linspace(m1_range[0], m1_range[1], 1000)
    density_fun_m1 = lambda m1: density_fun(jnp.stack([m1, m2_range[0]]))
    densities = vmap(density_fun_m1)(m1s)
    bank = update_fields(bank, density_max=densities.max())

    print("Estimating number of templates")
    _, key, n_templates_key = random.split(key, 3)
    naive_vol = (m1_range[1] - m1_range[0]) * (m2_range[1] - m2_range[0]) / 2
    bank = update_n_templates(n_templates_key, naive_vol, 2000, bank)

    print(f"Filling the bank with {bank.n_templates} templates")
    _, key, fill_bank_key = random.split(key, 3)
    bank = fill_bank(bank, key)

    m1_ts = bank.templates[0][:, 0]
    m2_ts = bank.templates[0][:, 1]
    plt.scatter(m1_ts, m2_ts, c=lax_map(density_fun_m1, bank.templates[0]))
    plt.xlabel(r"$m_1$ [M$_\odot$]")
    plt.ylabel(r"$m_2$ [M$_\odot$]")
    plt.tight_layout()
    plt.savefig(f"figures/{bank.name}.png")

    save_bank(bank)
