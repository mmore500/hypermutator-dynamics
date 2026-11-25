import numpy as np
import pytest

from pylib.theory._calc_mutator_fixprob import calc_mutator_fixprob


def test_calc_mutator_fixprob_vs_published_wellmixed():
    # mutator fixation fractions from https://osf.io/duam2
    assert calc_mutator_fixprob(256, 1) == pytest.approx(0.297, abs=0.02)
    assert calc_mutator_fixprob(2304, 1) == pytest.approx(0.875, abs=0.03)
    assert calc_mutator_fixprob(20736, 1) == pytest.approx(0.879, abs=0.03)
    assert calc_mutator_fixprob(186624, 1) == pytest.approx(0.284, abs=0.1)
    assert calc_mutator_fixprob(186624, 2) == pytest.approx(0.963, abs=0.03)
    assert calc_mutator_fixprob(1679616, 1) < 0.01


def test_calc_mutator_fixprob_unlimited_adaptive_potential():
    assert calc_mutator_fixprob(1679616, np.inf) == 1.0
    assert calc_mutator_fixprob(256, np.inf) == pytest.approx(
        calc_mutator_fixprob(256, 4), abs=0.01
    )
