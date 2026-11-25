import pytest

from pylib.theory._calc_load_params import calc_load_params


def test_calc_load_params_defaults():
    res = calc_load_params()
    assert res["delta"] == pytest.approx(0.0099, rel=0.01)
    assert res["q0"] == pytest.approx(0.9, rel=0.01)
    assert 0.15 < res["pi_mut"] < res["pi_non"] < 0.18
    assert res["growth_mut"] == pytest.approx(0.09, rel=0.02)
    assert res["supply_non"] / res["supply_mut"] == pytest.approx(
        1 / 81, rel=0.05
    )
    assert 600 < res["N_hitch"] < 750


def test_calc_load_params_no_mutator_effect():
    assert calc_load_params(M=1.0)["delta"] == 0.0
