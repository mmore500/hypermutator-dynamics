import pytest

from pylib.theory._calc_drift_fixprob import calc_drift_fixprob


def test_calc_drift_fixprob_neutral_limit():
    assert calc_drift_fixprob(1e-6, x0=0.5) == pytest.approx(0.5, abs=1e-6)
    assert calc_drift_fixprob(1e-6, x0=0.1) == pytest.approx(0.1, abs=1e-6)


def test_calc_drift_fixprob_large_population():
    assert calc_drift_fixprob(1e9, x0=0.5) == 0.0


def test_calc_drift_fixprob_monotonic():
    res = [calc_drift_fixprob(N) for N in (10, 100, 1000)]
    assert res == sorted(res, reverse=True)
