import pytest

from pylib.theory._calc_denovo_critical_potential_closed import (
    calc_denovo_critical_potential_closed,
)


def test_calc_denovo_critical_potential_closed_wellmixed():
    # about 6 to 7 by resimulation, at these population sizes
    for N in 1e7, 1e8, 1e9:
        assert 4.0 < calc_denovo_critical_potential_closed(N) < 8.0


def test_calc_denovo_critical_potential_closed_demes2d():
    # about 12 to 16 in simulation data, https://osf.io/ag4ur
    for N in 1679616, 15116544, 136048896:
        assert 11.0 < calc_denovo_critical_potential_closed(N, 256) < 16.0


def test_calc_denovo_critical_potential_closed_structure_hinders():
    assert calc_denovo_critical_potential_closed(
        15116544, 256
    ) > calc_denovo_critical_potential_closed(15116544)


def test_calc_denovo_critical_potential_closed_weak_size_dependence():
    res = [
        calc_denovo_critical_potential_closed(N, 256) for N in (1e7, 1e8, 1e9)
    ]
    assert res == sorted(res)
    assert res[-1] - res[0] < 3.0


def test_calc_denovo_critical_potential_closed_rarer_mutators_need_more():
    assert calc_denovo_critical_potential_closed(
        1e8, mu=1e-8
    ) > calc_denovo_critical_potential_closed(1e8, mu=1e-6)
