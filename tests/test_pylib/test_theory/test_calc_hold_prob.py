import numpy as np
import pytest

from pylib.theory._calc_hold_prob import calc_hold_prob


def test_calc_hold_prob_exponential_in_population_size():
    p, c = calc_hold_prob(1e5, 1)
    assert 5.0 < c < 7.0
    assert p == pytest.approx(np.exp(-1e5 * 1e-6 * c), rel=1e-3)
    assert calc_hold_prob(1e7, 1)[0] < 1e-6


def test_calc_hold_prob_increases_with_adaptive_potential():
    res = [calc_hold_prob(1e6, K)[0] for K in (1, 2, 3, 4)]
    assert res == sorted(res)
    assert res[-1] > 0.99
