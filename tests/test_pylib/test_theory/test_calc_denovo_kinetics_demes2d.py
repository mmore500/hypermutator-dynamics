import numpy as np

from pylib.theory._calc_denovo_kinetics_demes2d import (
    calc_denovo_kinetics_demes2d,
)


def test_calc_denovo_kinetics_demes2d():
    res = calc_denovo_kinetics_demes2d(1679616, 4)
    t = res["t"]
    assert res["num_demes"] == 1679616 / 256
    for key in "arrived", "rate_peak", "reach_mut", "uncovered":
        assert res[key].shape == t.shape
    assert (np.diff(res["arrived"]) >= -1e-9).all()
    assert res["arrived"][0] < 1e-6 and res["arrived"][-1] > 0.99
    assert (np.diff(res["uncovered"]) <= 1e-12).all()
    assert res["uncovered"][0] == 1.0 and res["uncovered"][-1] < 1e-6
    assert res["reach_mut"].max() <= res["num_demes"]


def test_calc_denovo_kinetics_demes2d_more_classes_arrive_later():
    medians = []
    for K in 2, 4, 8:
        res = calc_denovo_kinetics_demes2d(1679616, K)
        medians.append(res["t"][np.searchsorted(res["arrived"], 0.5)])
    assert medians == sorted(medians)
