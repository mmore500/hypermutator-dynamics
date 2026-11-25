import pytest

from pylib.theory._calc_denovo_step_prob import calc_denovo_step_prob


def test_calc_denovo_step_prob():
    assert 0.007 < calc_denovo_step_prob() < 0.01


def test_calc_denovo_step_prob_scales_with_mutator_supply():
    assert calc_denovo_step_prob(mu=2e-6) == pytest.approx(
        2.0 * calc_denovo_step_prob(), rel=0.02
    )
