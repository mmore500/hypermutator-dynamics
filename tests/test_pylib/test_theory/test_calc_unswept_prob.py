import numpy as np
import pytest
from scipy import integrate as sp_integrate

from pylib.theory._calc_denovo_critical_potential_closed import (
    calc_denovo_critical_potential_closed,
)
from pylib.theory._calc_denovo_mutator_frequency import (
    calc_denovo_mutator_frequency,
)
from pylib.theory._calc_load_params import calc_load_params
from pylib.theory._calc_sweep_time import calc_sweep_time
from pylib.theory._calc_unswept_prob import calc_unswept_prob


@pytest.mark.parametrize("deme_size", [None, 32, 256, 2048])
@pytest.mark.parametrize("supply", [1e-7, 1e-5])
def test_calc_unswept_prob_integrates_to_sweep_time(deme_size, supply):
    args = supply, 0.09, 0.16, deme_size
    expected = calc_sweep_time(*args)
    actual, __ = sp_integrate.quad(
        calc_unswept_prob, 0, 20 * expected, args=args, limit=500
    )
    assert actual == pytest.approx(expected, rel=0.01)


@pytest.mark.parametrize("deme_size", [None, 256])
def test_calc_unswept_prob_decreasing(deme_size):
    probs = [
        calc_unswept_prob(t, 1e-5, 0.09, 0.16, deme_size)
        for t in (0, 50, 100, 200, 400)
    ]
    assert probs[0] == 1.0
    assert probs == sorted(probs, reverse=True)
    assert probs[-1] < 0.01


@pytest.mark.parametrize("N", [1679616, 15116544, 136048896])
def test_calc_unswept_prob_critical_potential(N: int):
    # K* leaves ln 2 demes expected unreached by mutator domains
    p = calc_load_params()
    T_non = calc_sweep_time(p["supply_non"], p["growth_non"], p["pi_non"], 256)
    K = calc_denovo_critical_potential_closed(N, 256)
    unswept = calc_unswept_prob(
        K * T_non,
        calc_denovo_mutator_frequency(T_non) * p["supply_mut"],
        p["growth_mut"],
        p["pi_mut"],
        256,
    )
    assert N / 256 * unswept == pytest.approx(np.log(2), rel=1e-6)
