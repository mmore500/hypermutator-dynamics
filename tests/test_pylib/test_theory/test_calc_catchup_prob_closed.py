import numpy as np
import pytest

from pylib.theory._calc_catchup_prob_closed import calc_catchup_prob_closed
from pylib.theory._calc_hold_prob import calc_hold_prob


# the closed form's per-class supply ratio is M q0 = 90, against about 81
# in the recursion, which solves establishment separately for each type,
# so agreement loosens by about 10% with each further class
@pytest.mark.parametrize(
    "K, rel", [(1, 0.2), (2, 0.2), (3, 0.2), (4, 0.3), (5, 0.45)]
)
def test_calc_catchup_prob_closed_vs_recursion(K: int, rel: float):
    __, expected = calc_hold_prob(1e12, K, finite_wave=False)
    __, actual = calc_catchup_prob_closed(1e12, K)
    assert actual == pytest.approx(expected, rel=rel)


def test_calc_catchup_prob_closed_share_law():
    # each further available mutation costs nonmutators supply ratio
    c = [calc_catchup_prob_closed(1e6, K)[1] for K in (2, 3, 4)]
    assert c[1] / c[0] == pytest.approx(c[2] / c[1], rel=0.02)
    assert 1 / 100 < c[1] / c[0] < 1 / 60


def test_calc_catchup_prob_closed_exponential_in_population_size():
    p, c = calc_catchup_prob_closed(1e5, 1)
    assert 1 - p == pytest.approx(np.exp(-1e5 * 1e-6 * c))


def test_calc_catchup_prob_closed_structure_slows_sweeps():
    assert (
        calc_catchup_prob_closed(1e5, 1, deme_size=256)[0]
        > calc_catchup_prob_closed(1e5, 1)[0]
    )


def test_calc_catchup_prob_closed_single_deme_is_wellmixed():
    assert calc_catchup_prob_closed(256, 1, deme_size=256) == (
        calc_catchup_prob_closed(256, 1)
    )


@pytest.mark.parametrize("K", [1, 2, 3])
def test_calc_catchup_prob_closed_single_invader(K: int):
    # nonmutators catch a single invader more easily as populations grow
    probs = [
        calc_catchup_prob_closed(N, K, x0=1 / N)[0] for N in (1e3, 1e5, 1e7)
    ]
    assert 0.0 <= probs[0] < probs[1] < probs[2] <= 1.0


def test_calc_catchup_prob_closed_invader_share_conditioned_on_escape():
    # nonmutators' odds in class 1 are N supply_non / delta for an invader
    from pylib.theory._calc_load_params import calc_load_params
    from pylib.theory._calc_sweep_time import calc_sweep_time

    p, N = calc_load_params(), 1e4
    odds = N * p["supply_non"] / p["delta"]
    share = odds / (1 + odds)
    T = calc_sweep_time(
        (1 - share) * p["supply_mut"], p["growth_mut"], p["pi_mut"]
    )
    expected = -np.expm1(-N * p["supply_non"] * share * T)
    assert calc_catchup_prob_closed(N, 2, x0=1 / N)[0] == pytest.approx(
        expected, rel=1e-3
    )
