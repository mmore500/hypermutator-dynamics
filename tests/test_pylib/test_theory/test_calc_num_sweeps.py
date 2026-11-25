import pytest

from pylib.theory._calc_num_sweeps import calc_num_sweeps


def test_calc_num_sweeps_no_time():
    assert calc_num_sweeps(1e7, 1) == pytest.approx(0.0, abs=1e-3)


def test_calc_num_sweeps_increases_with_selection_and_time():
    res = [calc_num_sweeps(4e7, 1000, s=s) for s in (0.02, 0.05, 0.1)]
    assert res == sorted(res)
    assert 3.0 < res[1] < 6.0
    assert calc_num_sweeps(4e7, 500) < calc_num_sweeps(4e7, 1000)


def test_calc_num_sweeps_capped():
    assert calc_num_sweeps(4e7, 2000, max_ben=3) == pytest.approx(
        3.0, abs=0.01
    )
