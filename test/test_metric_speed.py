import time

from diffbank import metric, utils
from diffbank.waveforms.threePN_simple import Psi, amp
from diffbank.noise import Sn_aLIGO
from jax import jit, random
import jax.numpy as jnp


def test_metric_speed():
    """
    Tests that the metric calculation runs, and times it.
    """
    f = jnp.linspace(30.0, 512.0, 1000)

    print(
        "Match:",
        metric.match(
            jnp.array([3, 1.0]),
            0.0,
            jnp.array([0.0, 0.0]),
            amp,
            Psi,
            f,
            Sn_aLIGO,
        ),
    )

    get_g = jit(metric.get_g, static_argnums=(1, 2, 4))
    print(
        "Metric:",
        get_g(jnp.array([3, 1.0]), amp, Psi, f, Sn_aLIGO),
    )

    N = 1000
    key = random.PRNGKey(89)
    m1_key, m2_key = random.split(key)
    m1 = random.uniform(m1_key, minval=2.0, maxval=3.0, shape=(N,))
    m2 = random.uniform(m2_key, minval=1.0, maxval=2.0, shape=(N,))

    t0 = time.time()
    for i in range(N):
        pars = jnp.array([m1[i], m2[i]])
        get_g(pars, amp, Psi, f, Sn_aLIGO).block_until_ready()
    t1 = time.time()

    total = t1 - t0
    print("Time per metric evaluation:", total / N)


if __name__ == "__main__":
    test_metric_speed()
