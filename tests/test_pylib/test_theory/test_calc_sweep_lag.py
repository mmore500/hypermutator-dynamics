import numpy as np
import pytest

from pylib.theory._calc_sweep_lag import calc_sweep_lag


def test_calc_sweep_lag_increases_with_frequency():
    res = [calc_sweep_lag(4e7, f, s=0.05) for f in (0.05, 0.5, 0.9)]
    assert res == sorted(res)
    assert 200 < res[0] < 400


def test_calc_sweep_lag_logistic():
    # time from 5% to 95% is 2 ln(19) / rate
    diff = calc_sweep_lag(4e7, 0.95, s=0.05) - calc_sweep_lag(
        4e7, 0.05, s=0.05
    )
    assert diff == pytest.approx(2 * np.log(19) / 0.04, rel=1e-6)


def test_calc_sweep_lag_load_exceeds_advantage():
    assert calc_sweep_lag(4e7, 0.9, s=0.01) == np.inf
