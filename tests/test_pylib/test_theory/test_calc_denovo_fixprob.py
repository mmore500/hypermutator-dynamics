import pytest

from pylib.theory._calc_denovo_fixprob import calc_denovo_fixprob
from pylib.theory._calc_denovo_step_prob import calc_denovo_step_prob


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
