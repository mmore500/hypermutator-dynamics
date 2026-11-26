import numpy as np
import pytest

from pylib.theory._calc_mutator_fixprob import calc_mutator_fixprob
from pylib.theory._calc_mutator_fixprob_closed import (
    calc_mutator_fixprob_closed,
)


@pytest.mark.parametrize(
    "N, K",
    [(256, 1), (2304, 1), (20736, 1), (186624, 1), (1679616, 2), (3e8, 3)],
)
def test_calc_mutator_fixprob_closed_vs_numeric(N: float, K: int):
    assert calc_mutator_fixprob_closed(N, K) == pytest.approx(
        calc_mutator_fixprob(N, K), abs=0.06
    )


def test_calc_mutator_fixprob_closed_vs_published_wellmixed():
    # mutator fixation fractions from https://osf.io/duam2
    assert calc_mutator_fixprob_closed(256, 1) == pytest.approx(
        0.297, abs=0.03
    )
    assert calc_mutator_fixprob_closed(20736, 1) == pytest.approx(
        0.879, abs=0.03
    )
    assert calc_mutator_fixprob_closed(186624, 2) == pytest.approx(
        0.963, abs=0.03
    )


def test_calc_mutator_fixprob_closed_vs_published_demes2d():
    # mutator fixation fractions from https://osf.io/8q5v6 (256 per deme)
    # and https://osf.io/cu9r6 (2048 per deme)
    assert calc_mutator_fixprob_closed(
        20736, 1, deme_size=256
    ) == pytest.approx(0.809, abs=0.05)
    assert calc_mutator_fixprob_closed(
        186624, 1, deme_size=256
    ) == pytest.approx(0.173, abs=0.1)
    assert calc_mutator_fixprob_closed(
        147456, 1, deme_size=2048
    ) == pytest.approx(0.271, abs=0.1)


def test_calc_mutator_fixprob_closed_unlimited_adaptive_potential():
    assert calc_mutator_fixprob_closed(1e7, np.inf) == pytest.approx(1.0)
