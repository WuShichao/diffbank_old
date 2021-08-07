import click
import matplotlib.pyplot as plt

from diffbank.bank import Bank
from diffbank.noise import Sn_aLIGO
from diffbank.utils import get_m1_m2_sampler
from diffbank.waveforms.threePN_simple import Psi, amp
import jax
from jax import random
import jax.numpy as jnp
from tqdm.auto import tqdm, trange


def test_eff(key):
    minimum_match = 0.95
    eta = 0.99
    fs = jnp.linspace(20.0, 2000.0, 300)
    m_range = (2.0, 3.0)
    sampler = get_m1_m2_sampler(m_range, m_range)

    bank = Bank(
        amp,
        Psi,
        fs,
        Sn_aLIGO,
        m_star=1 - minimum_match,
        eta=eta,
        sample_base=sampler,
        name="testing",
    )

    bank.ratio_max = jnp.array(bank.density_fun(jnp.array([m_range[0], m_range[0]])))

    key, subkey = random.split(key)
    bank.fill_bank(subkey, "stochastic", 1000)
    plt.figure(figsize=(7, 5))
    plt.scatter(bank.templates[:, 0], bank.templates[:, 1], s=1)
    plt.xlabel("m_1")
    plt.ylabel("m_2")
    plt.savefig("figures/test_template_placement_stochastic.pdf", bbox_inches="tight")

    key, subkey = random.split(key)
    bank.calc_bank_effectualness(subkey, 500)


@click.command()
@click.option("--seed", type=int, help="PRNG seed")
def main(seed):
    key = random.PRNGKey(seed)
    test_eff(key)


if __name__ == "__main__":
    main()
