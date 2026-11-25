import pytest

from pylib.theory._calc_hitchhike_prob_closed import (
    calc_hitchhike_prob_closed,
)
from pylib.theory._calc_load_params import calc_load_params


def test_calc_hitchhike_prob_closed_reduces_to_raynes_single_mutant():
    # Raynes et al. eq. 1c: (Uben / Udel) Pfix(sben), for x0 = 1 / N
    N = 1e5
    p = calc_load_params()
    # ... times small correction for the lag before load-driven decline
    expected = (1.0 + p["delta"] * p["t_lag"]) / p["N_hitch"]
    actual = calc_hitchhike_prob_closed(N, x0=1.0 / N)
    assert actual == pytest.approx(expected, rel=0.02)


def test_calc_hitchhike_prob_closed_limits():
    assert calc_hitchhike_prob_closed(1.0) < 0.01
    assert calc_hitchhike_prob_closed(1e6) == pytest.approx(1.0)
