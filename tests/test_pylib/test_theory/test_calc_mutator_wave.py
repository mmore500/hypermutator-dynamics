import numpy as np

from pylib.theory._calc_mutator_wave import calc_mutator_wave


def test_calc_mutator_wave_shape():
    res = calc_mutator_wave(3)
    assert res.ndim == 2
    assert res.shape[1] == 4
    assert (res > 0.89).all() and (res < 1.11).all()


def test_calc_mutator_wave_fitter_grow_faster():
    res = calc_mutator_wave(3, N=1e6)
    assert (np.diff(res, axis=1) >= -1e-12).all()


def test_calc_mutator_wave_finite_population_slower():
    assert len(calc_mutator_wave(4, N=1e5)) > len(calc_mutator_wave(4))
