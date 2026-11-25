import numpy as np
import pytest

from pylib.theory._calc_ncrit import calc_ncrit
from pylib.theory._calc_ncrit_closed import calc_ncrit_closed


def test_calc_ncrit_closed_lower():
    assert 500 < calc_ncrit_closed(1, "lower") < 800
    assert calc_ncrit_closed(1, "lower") == calc_ncrit_closed(
        np.inf, "lower", x0=0.1, deme_size=256
    )


@pytest.mark.parametrize("K", [1, 2, 3])
def test_calc_ncrit_closed_upper_vs_numeric(K: int):
    assert calc_ncrit_closed(K, "upper") == pytest.approx(
        calc_ncrit(K, "upper"), rel=0.25
    )


def test_calc_ncrit_closed_upper_unlimited_adaptive_potential():
    assert np.isnan(calc_ncrit_closed(np.inf, "upper"))


def test_calc_ncrit_closed_upper_lower_in_demes2d():
    for K in 1, 2, 3:
        assert calc_ncrit_closed(
            K, "upper", deme_size=256
        ) < calc_ncrit_closed(K, "upper")
