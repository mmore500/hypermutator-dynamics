import pytest

from pylib.theory._calc_fill_time import calc_fill_time


def test_calc_fill_time():
    # ln(256 * 0.18) / 0.09
    assert calc_fill_time(256, 0.09, 0.18) == pytest.approx(42.6, rel=0.01)


def test_calc_fill_time_shorter_with_stronger_selection():
    assert calc_fill_time(256, 0.2, 0.18) < calc_fill_time(256, 0.1, 0.18)


def test_calc_fill_time_longer_in_larger_demes():
    assert calc_fill_time(2048, 0.1, 0.18) > calc_fill_time(256, 0.1, 0.18)
