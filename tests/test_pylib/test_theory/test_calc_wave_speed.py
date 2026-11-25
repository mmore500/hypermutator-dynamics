import pytest

from pylib.theory._calc_wave_speed import calc_wave_speed


def test_calc_wave_speed():
    assert calc_wave_speed(256, 0.1) == pytest.approx(0.0308, rel=0.01)


def test_calc_wave_speed_faster_with_stronger_selection():
    assert calc_wave_speed(256, 0.2) > calc_wave_speed(256, 0.1)


def test_calc_wave_speed_slower_in_larger_demes():
    assert calc_wave_speed(2048, 0.1) < calc_wave_speed(256, 0.1)
