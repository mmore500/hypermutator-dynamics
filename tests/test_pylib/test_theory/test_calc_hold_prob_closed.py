import numpy as np
import pytest

from pylib.theory._calc_hold_prob import calc_hold_prob
from pylib.theory._calc_hold_prob_closed import calc_hold_prob_closed


@pytest.mark.parametrize("K", [1, 2, 3, 4, 5])
def test_calc_hold_prob_closed_vs_recursion(K: int):
    __, expected = calc_hold_prob(1e12, K, finite_wave=False)
    __, actual = calc_hold_prob_closed(1e12, K)
    assert actual == pytest.approx(expected, rel=0.2)


def test_calc_hold_prob_closed_share_law():
    # each further available mutation costs nonmutators supply ratio
    c = [calc_hold_prob_closed(1e6, K)[1] for K in (2, 3, 4)]
    assert c[1] / c[0] == pytest.approx(c[2] / c[1])
    assert 1 / 100 < c[1] / c[0] < 1 / 60


def test_calc_hold_prob_closed_exponential_in_population_size():
    p, c = calc_hold_prob_closed(1e5, 1)
    assert p == pytest.approx(np.exp(-1e5 * 1e-6 * c))


def test_calc_hold_prob_closed_structure_slows_sweeps():
    assert (
        calc_hold_prob_closed(1e5, 1, deme_size=256)[0]
        < calc_hold_prob_closed(1e5, 1)[0]
    )


def test_calc_hold_prob_closed_single_deme_is_wellmixed():
    assert calc_hold_prob_closed(256, 1, deme_size=256) == (
        calc_hold_prob_closed(256, 1)
    )
