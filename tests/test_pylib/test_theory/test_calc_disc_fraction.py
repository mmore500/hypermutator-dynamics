import numpy as np
import pytest

from pylib.theory._calc_disc_fraction import calc_disc_fraction


def test_calc_disc_fraction_limits():
    assert calc_disc_fraction(0.0) == 0.0
    assert calc_disc_fraction(np.sqrt(2.0)) == pytest.approx(1.0, abs=1e-3)
    assert calc_disc_fraction(10.0) == 1.0


def test_calc_disc_fraction_small_radius():
    assert calc_disc_fraction(0.05) == pytest.approx(
        np.pi * 0.05**2, rel=0.15
    )


def test_calc_disc_fraction_monotonic():
    res = calc_disc_fraction(np.linspace(0.0, 1.5, 50))
    assert (np.diff(res) >= 0).all()
