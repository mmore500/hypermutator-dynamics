import numpy as np
import pytest

from pylib.theory._calc_invader_hold_prob_closed import (
    calc_invader_hold_prob_closed,
)


def test_calc_invader_hold_prob_closed_unlimited():
    assert calc_invader_hold_prob_closed(1e6, np.inf) == 1.0


@pytest.mark.parametrize("K", [1, 2, 3])
def test_calc_invader_hold_prob_closed_decreases_with_population_size(K):
    probs = [calc_invader_hold_prob_closed(N, K) for N in (1e3, 1e5, 1e7)]
    assert 1.0 >= probs[0] > probs[1] > probs[2] >= 0.0


@pytest.mark.parametrize("N", [1e4, 1e5, 1e6])
def test_calc_invader_hold_prob_closed_increases_with_potential(N: float):
    probs = [calc_invader_hold_prob_closed(N, K) for K in (1, 2, 3, 4)]
    assert probs == sorted(probs)


def test_calc_invader_hold_prob_closed_share_law():
    # each further available mutation costs nonmutators the supply ratio
    e = [-np.log(calc_invader_hold_prob_closed(1e6, K)) for K in (2, 3, 4)]
    assert e[1] / e[0] == pytest.approx(e[2] / e[1])
    assert 1 / 100 < e[1] / e[0] < 1 / 60


def test_calc_invader_hold_prob_closed_small_populations_hold():
    assert calc_invader_hold_prob_closed(100, 1) == pytest.approx(1, abs=0.01)
