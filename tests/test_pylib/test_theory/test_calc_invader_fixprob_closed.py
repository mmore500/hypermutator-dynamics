import numpy as np
import pytest

from pylib.theory._calc_escape_prob_closed import calc_escape_prob_closed
from pylib.theory._calc_invader_fixprob_closed import (
    calc_invader_fixprob_closed,
)
from pylib.theory._calc_load_params import calc_load_params


@pytest.mark.parametrize("N", [10, 1e3, 1e5])
def test_calc_invader_fixprob_closed_unlimited_is_escape(N: float):
    assert calc_invader_fixprob_closed(N, np.inf) == pytest.approx(
        calc_escape_prob_closed(N, 1.0 / N)
    )


def test_calc_invader_fixprob_closed_neutral_at_small_sizes():
    assert 2 * calc_invader_fixprob_closed(2, 1) == pytest.approx(1, abs=0.05)


def test_calc_invader_fixprob_closed_sign_inversion():
    N_hitch = calc_load_params()["N_hitch"]
    for K in 1, 2, np.inf:
        lo, hi = N_hitch / 2, N_hitch * 2
        assert lo * calc_invader_fixprob_closed(lo, K) < 1
        assert hi * calc_invader_fixprob_closed(hi, K) > 1


@pytest.mark.parametrize("K, N", [(1, 1e6), (2, 1e7)])
def test_calc_invader_fixprob_closed_disfavored_at_large_sizes(K, N):
    assert N * calc_invader_fixprob_closed(N, K) < 1
    assert N * calc_invader_fixprob_closed(N, np.inf) > 1
