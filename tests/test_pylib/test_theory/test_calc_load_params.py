import pytest

from pylib.theory._calc_load_params import calc_load_params


def test_calc_load_params_defaults():
    res = calc_load_params()
    assert res["delta"] == pytest.approx(0.0099, rel=0.01)
    assert res["q0"] == pytest.approx(0.9, rel=0.01)
    assert res["pi_mut"] == res["pi_non"] == pytest.approx(0.1813, rel=1e-3)
    assert res["growth_mut"] == pytest.approx(0.09, rel=0.02)
    assert res["supply_mut"] / res["supply_non"] == pytest.approx(
        100 * res["q0"]
    )
    assert res["N_hitch"] == pytest.approx(603, rel=0.01)


def test_calc_load_params_no_mutator_effect():
    assert calc_load_params(M=1.0)["delta"] == 0.0
