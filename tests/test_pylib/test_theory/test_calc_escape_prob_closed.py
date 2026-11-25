import pytest

from pylib.theory._calc_drift_fixprob import calc_drift_fixprob
from pylib.theory._calc_escape_prob_closed import calc_escape_prob_closed
from pylib.theory._calc_hitchhike_prob_closed import calc_hitchhike_prob_closed


def test_calc_escape_prob_closed_limits():
    assert calc_escape_prob_closed(4) == pytest.approx(0.5, abs=0.02)
    assert calc_escape_prob_closed(1e6) == pytest.approx(1.0)


@pytest.mark.parametrize("N", [32, 256, 2048])
def test_calc_escape_prob_closed_bounds(N: int):
    res = calc_escape_prob_closed(N)
    assert res >= calc_drift_fixprob(N)
    assert res >= calc_hitchhike_prob_closed(N)
    assert res <= calc_drift_fixprob(N) + calc_hitchhike_prob_closed(N)
