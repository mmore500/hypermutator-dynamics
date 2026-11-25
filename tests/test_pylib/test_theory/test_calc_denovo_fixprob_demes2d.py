import pytest

from pylib.theory._calc_denovo_fixprob import calc_denovo_fixprob
from pylib.theory._calc_denovo_fixprob_demes2d import (
    calc_denovo_fixprob_demes2d,
)


def test_calc_denovo_fixprob_demes2d_single_deme_is_wellmixed():
    assert calc_denovo_fixprob_demes2d(256, 4) == calc_denovo_fixprob(256, 4)


@pytest.mark.parametrize(
    "N, K, expected, abs_tol",
    [
        # mutator fixation fractions from https://osf.io/ag4ur (2D demes)
        (20736, 16, 0.155, 0.05),
        (20736, 40, 0.380, 0.08),
        (186624, 4, 0.025, 0.03),
        (186624, 12, 0.358, 0.08),
        (186624, 24, 0.679, 0.08),
        (1679616, 8, 0.0, 0.05),
        (1679616, 14, 0.778, 0.15),
        (15116544, 10, 0.0, 0.05),
        (15116544, 20, 1.0, 0.05),
    ],
)
def test_calc_denovo_fixprob_demes2d_vs_published(
    N: int, K: int, expected: float, abs_tol: float
):
    assert calc_denovo_fixprob_demes2d(N, K) == pytest.approx(
        expected, abs=abs_tol
    )


def test_calc_denovo_fixprob_demes2d_increases_with_adaptive_potential():
    res = [calc_denovo_fixprob_demes2d(1679616, K) for K in (8, 12, 16, 20)]
    assert res == sorted(res)


def test_calc_denovo_fixprob_demes2d_structure_hinders_large_populations():
    assert calc_denovo_fixprob_demes2d(15116544, 10) < calc_denovo_fixprob(
        15116544, 10
    )
