import jax
import click
from jax import random
import jax.numpy as jnp
from tqdm.auto import trange, tqdm
import matplotlib.pyplot as plt

from diffbank.bank import Bank

# from diffbank.metric import get_density
from diffbank.noise import Sn_aLIGO
from diffbank.utils import (
    get_m1_m2_sampler,
)
from diffbank.waveforms.threePN_simple import Psi, amp


def test_n_eff_pts():
    eta = 0.99
    from diffbank.utils import n_eff_pts

    print(n_eff_pts(eta, alpha=0.1))


def test_eff(key):
    minimum_match = 0.9
    eta = 0.9
    fs = jnp.linspace(20.0, 2000.0, 1000)
    m_range = (2.0, 3.0)
    sampler = get_m1_m2_sampler(m_range, m_range)

    bank = Bank(
        amp,
        Psi,
        fs,
        Sn_aLIGO,
        sampler,
        m_star=1 - minimum_match,
        eta=eta,
        name="testing",
    )

    bank.density_max = jnp.array(bank.density_fun(jnp.array([m_range[0], m_range[0]])))

    # key, subkey = random.split(key)
    # bank.fill_bank(subkey, method="stochastic")
    # # bank.fill_bank(subkey)
    # # print(bank.templates)
    # plt.figure(figsize=(7, 5))
    # plt.scatter(bank.templates[:, 0], bank.templates[:, 1], s=1)
    # plt.xlabel("m_1")
    # plt.ylabel("m_2")
    # plt.savefig("figures/test_template_placement_stochastic.pdf", bbox_inches="tight")

    key, subkey = random.split(key)
    bank.fill_bank(subkey, show_progress=True)
    print(minimum_match)
    print(bank.n_templates)

    key, subkey = random.split(key)
    bank.fill_bank(subkey, show_progress=True)
    print(minimum_match)
    print(bank.n_templates)

    key, subkey = random.split(key)
    bank.fill_bank(subkey, show_progress=True)
    print(minimum_match)
    print(bank.n_templates)

    key, subkey = random.split(key)
    bank.fill_bank(subkey, show_progress=True)
    print(minimum_match)
    print(bank.n_templates)

    key, subkey = random.split(key)
    bank.fill_bank(subkey, show_progress=True)
    print(minimum_match)
    print(bank.n_templates)

    # key, subkey = random.split(key)
    # N = 1000
    # bank.calculate_bank_effectualness(subkey, N)


@click.command()
@click.option("--seed", type=int, help="PRNG seed")
def main(seed):
    key = random.PRNGKey(seed)
    test_eff(key)
    # test_n_eff_pts()


if __name__ == "__main__":
    main()