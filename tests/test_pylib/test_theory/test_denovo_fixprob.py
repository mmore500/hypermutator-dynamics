import pytest

from pylib.theory._denovo_fixprob import (
    calc_denovo_drift_fixprob,
    calc_denovo_fixprob,
    calc_denovo_step_prob,
)
from pylib.theory._resimulate_wellmixed import resimulate_wellmixed


def test_calc_denovo_step_prob():
    assert 0.007 < calc_denovo_step_prob() < 0.01


def test_calc_denovo_drift_fixprob():
    assert calc_denovo_drift_fixprob(256) == pytest.approx(0.0165, abs=0.003)
    assert calc_denovo_drift_fixprob(20736) < 1e-9


@pytest.mark.parametrize("K", [1, 8, 16])
def test_calc_denovo_fixprob_small_population_limit(K: int):
    rho = calc_denovo_step_prob()
    expected = 1.0 - (1.0 - rho) ** K
    assert calc_denovo_fixprob(2304, K) == pytest.approx(expected, rel=0.15)


def test_calc_denovo_fixprob_vs_published_wellmixed():
    # mutator fixation fractions from https://osf.io/edwbu
    assert calc_denovo_fixprob(20736, 40) == pytest.approx(0.248, abs=0.04)
    assert calc_denovo_fixprob(186624, 4) < 0.05
    assert calc_denovo_fixprob(186624, 40) == pytest.approx(0.42, abs=0.1)
    assert calc_denovo_fixprob(15116544, 4) < 0.05
    assert calc_denovo_fixprob(15116544, 12) > 0.9


def test_resimulate_wellmixed_denovo_vs_theory():
    res = resimulate_wellmixed(3162278, 10, n_rep=200, x0=0.0, mu=1e-6, seed=1)
    assert res.mean() == pytest.approx(
        calc_denovo_fixprob(3162278, 10), abs=0.12
    )
