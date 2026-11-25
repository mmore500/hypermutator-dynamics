import pytest

from pylib.theory._calc_denovo_drift_fixprob import calc_denovo_drift_fixprob


def test_calc_denovo_drift_fixprob():
    assert calc_denovo_drift_fixprob(256) == pytest.approx(0.0165, abs=0.003)
    assert calc_denovo_drift_fixprob(20736) < 1e-9
    assert calc_denovo_drift_fixprob(256, n_gen=0) == 0.0
