import numpy as np
import pytest

from pylib.theory._mutator_fixprob import (
    calc_drift_fixprob,
    calc_escape_prob,
    calc_escape_prob_closed,
    calc_establishment_prob,
    calc_hitchhike_prob_closed,
    calc_hold_prob,
    calc_load_params,
    calc_mutator_fixprob,
    calc_ncrit,
)


def test_calc_establishment_prob():
    assert calc_establishment_prob(1.0) == 0.0
    assert calc_establishment_prob(0.9) == 0.0
    assert calc_establishment_prob(1.01) == pytest.approx(0.02, rel=0.05)
    assert 0.17 < calc_establishment_prob(1.1) < 0.18


def test_calc_drift_fixprob_neutral_limit():
    assert calc_drift_fixprob(1e-6, x0=0.5) == pytest.approx(0.5, abs=1e-6)
    assert calc_drift_fixprob(1e9, x0=0.5) == 0.0


def test_hitchhike_reduces_to_raynes_single_mutant():
    # Raynes et al. eq. 1c: (Uben / Udel) Pfix(sben), for x0 = 1 / N
    N = 1e5
    p = calc_load_params()
    # ... times small correction for the lag before load-driven decline
    expected = (1.0 + p["delta"] * p["t_lag"]) / p["N_hitch"]
    actual = calc_hitchhike_prob_closed(N, x0=1.0 / N)
    assert actual == pytest.approx(expected, rel=0.02)


@pytest.mark.parametrize("N", [32, 256, 2048, 20000])
def test_escape_prob_closed_vs_numeric(N: int):
    assert calc_escape_prob(N) == pytest.approx(
        calc_escape_prob_closed(N), abs=0.05
    )


def test_hold_prob():
    p, c = calc_hold_prob(1e5, 1)
    assert 5.0 < c < 7.0
    assert p == pytest.approx(np.exp(-1e5 * 1e-6 * c), rel=1e-3)
    assert calc_hold_prob(1e5, 2)[0] > p
    assert calc_hold_prob(1e7, 1)[0] < 1e-6


def test_calc_mutator_fixprob_vs_published_wellmixed():
    # mutator fixation fractions from https://osf.io/duam2
    assert calc_mutator_fixprob(256, 1) == pytest.approx(0.297, abs=0.02)
    assert calc_mutator_fixprob(2304, 1) == pytest.approx(0.875, abs=0.03)
    assert calc_mutator_fixprob(20736, 1) == pytest.approx(0.879, abs=0.03)
    assert calc_mutator_fixprob(186624, 1) == pytest.approx(0.284, abs=0.1)
    assert calc_mutator_fixprob(186624, 2) == pytest.approx(0.963, abs=0.03)
    assert calc_mutator_fixprob(1679616, 1) < 0.01
    assert calc_mutator_fixprob(1679616, np.inf) == 1.0


def test_calc_ncrit():
    lower = calc_ncrit(np.inf, "lower")
    assert 400 < lower < 800
    assert calc_ncrit(1, "lower") == pytest.approx(lower, rel=0.05)
    assert np.isnan(calc_ncrit(np.inf, "upper"))
    upper1, upper2 = calc_ncrit(1, "upper"), calc_ncrit(2, "upper")
    assert 5e4 < upper1 < 2e5
    assert 2e6 < upper2 < 1e7
