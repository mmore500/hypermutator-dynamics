import pytest

from pylib.theory._calc_establishment_prob import calc_establishment_prob


def test_calc_establishment_prob_subcritical():
    assert calc_establishment_prob(1.0) == 0.0
    assert calc_establishment_prob(0.9) == 0.0


def test_calc_establishment_prob_supercritical():
    # approximately 2s for small s
    assert calc_establishment_prob(1.01) == pytest.approx(0.02, rel=0.05)
    assert 0.17 < calc_establishment_prob(1.1) < 0.18
    assert calc_establishment_prob(1.2) > calc_establishment_prob(1.1)
