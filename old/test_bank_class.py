import numpy as np
from functools import partial
from kappa_waveform import Amp_35PN, Psi_35PN
from math import pi
from core import metric
from core import utils
from core import bank
from threePN_simple import Amp_35PN, Psi_35PN


m_star = 1 - 0.95
p_ranges = np.array([[1, 3], [1, 3]])
N_test = 1000
eta = 0.9999

f_low = 10
f_upper = 512
sample_length = 94
sample_rate = 1024

tlen = sample_rate * sample_length
print(tlen)
flen = tlen / 2 + 1

tlist = np.arange(0, tlen + 1) * sample_length / tlen
f = np.arange(0, flen) / sample_length
fmax = flen / (tlen / sample_rate)
print(flen / fmax / sample_rate)
print(f)

# f = np.geomspace(10, 500, 10000)

# bank = genbank_rejection_sampling(
#     prange,
#     Amp_35PN,
#     Psi_35PN,
#     f,
#     utils.Sn_func,
#     utils.gen_samples_from_ranges,
#     m_star,
#     eta,
# )
