import pytest

from pylib.theory._calc_escape_prob import calc_escape_prob
from pylib.theory._calc_escape_prob_closed import calc_escape_prob_closed


@pytest.mark.parametrize("N", [32, 256, 2048, 20000])
def test_calc_escape_prob_vs_closed(N: int):
    assert calc_escape_prob(N) == pytest.approx(
        calc_escape_prob_closed(N), abs=0.05
    )


def test_calc_escape_prob_sign_inversion():
    # mutators disfavored at intermediate N, favored at large N
    assert calc_escape_prob(8) > calc_escape_prob(128)
    assert calc_escape_prob(128) < 0.5 < calc_escape_prob(2048)
    assert calc_escape_prob(1e7) == 1.0
