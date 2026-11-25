import numpy as np
import pytest

from pylib.theory._calc_hold_from_state import calc_hold_from_state
from pylib.theory._calc_hold_prob import calc_hold_prob


def make_state(K: int, x0: float) -> np.ndarray:
    f = np.zeros((2, K + 1, 5))
    f[0, 0, 0], f[1, 0, 0] = 1.0 - x0, x0
    return f


@pytest.mark.parametrize("N, K", [(1e5, 1), (3e6, 2)])
def test_calc_hold_from_state_vs_calc_hold_prob(N: float, K: int):
    actual = calc_hold_from_state(
        make_state(K, 0.5), N, K, 0.1, 1e-6, 1e-4, 100.0
    )
    assert actual == pytest.approx(calc_hold_prob(N, K)[0], abs=0.1)


def test_calc_hold_from_state_more_mutators_hold_better():
    res = [
        calc_hold_from_state(make_state(2, x0), 3e6, 2, 0.1, 1e-6, 1e-4, 100.0)
        for x0 in (0.01, 0.5, 0.99)
    ]
    assert res == sorted(res)
