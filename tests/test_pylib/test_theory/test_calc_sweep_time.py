import pytest

from pylib.theory._calc_sweep_time import calc_sweep_time


def test_calc_sweep_time_wellmixed():
    # [ln(pi r / u) - gamma] / r
    assert calc_sweep_time(1e-5, 0.1, 0.18) == pytest.approx(69.1, rel=0.01)


def test_calc_sweep_time_demes2d():
    assert 90 < calc_sweep_time(1.43e-5, 0.09, 0.159, 256) < 115


@pytest.mark.parametrize("deme_size", [None, 32, 256, 2048])
def test_calc_sweep_time_faster_with_more_supply(deme_size: int):
    assert calc_sweep_time(1e-4, 0.1, 0.18, deme_size) < calc_sweep_time(
        1e-6, 0.1, 0.18, deme_size
    )
