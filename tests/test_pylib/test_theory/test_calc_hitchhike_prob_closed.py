import pytest

from pylib.theory._calc_hitchhike_prob_closed import calc_hitchhike_prob_closed
from pylib.theory._calc_load_params import calc_load_params


def test_calc_hitchhike_prob_closed_reduces_to_raynes_single_mutant():
    # Raynes et al. eq. 1c: (Uben / Udel) Pfix(sben), for x0 = 1 / N
    N = 1e5
    p = calc_load_params()
    expected = 1.0 / p["N_hitch"]
    actual = calc_hitchhike_prob_closed(N, x0=1.0 / N)
    assert actual == pytest.approx(expected, rel=0.02)


def test_calc_hitchhike_prob_closed_limits():
    assert calc_hitchhike_prob_closed(1.0) < 0.01
    assert calc_hitchhike_prob_closed(1e6) == pytest.approx(1.0)


@pytest.mark.parametrize("x0", [0.01, 0.5, 0.9])
def test_calc_hitchhike_prob_closed_threshold_independent_of_x0(x0: float):
    N_hitch = calc_load_params()["N_hitch"]
    assert calc_hitchhike_prob_closed(N_hitch, x0=x0) == pytest.approx(x0)
