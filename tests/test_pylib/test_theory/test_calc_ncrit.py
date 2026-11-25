import numpy as np
import pytest

from pylib.theory._calc_ncrit import calc_ncrit


def test_calc_ncrit_lower():
    lower = calc_ncrit(np.inf, "lower")
    assert 400 < lower < 800
    assert calc_ncrit(1, "lower") == pytest.approx(lower, rel=0.05)


def test_calc_ncrit_upper():
    assert np.isnan(calc_ncrit(np.inf, "upper"))
    assert 5e4 < calc_ncrit(1, "upper") < 2e5
    assert 2e6 < calc_ncrit(2, "upper") < 1e7
