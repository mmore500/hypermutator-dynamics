import pytest

from pylib.theory._calc_num_sweeps import calc_num_sweeps
from pylib.theory._calc_selection_from_num_sweeps import (
    calc_selection_from_num_sweeps,
)


@pytest.mark.parametrize("num_sweeps", [2.0, 4.0, 7.0])
def test_calc_selection_from_num_sweeps_roundtrip(num_sweeps: float):
    s = calc_selection_from_num_sweeps(4e7, 1000, num_sweeps)
    assert calc_num_sweeps(4e7, 1000, s=s) == pytest.approx(
        num_sweeps, abs=0.1
    )


def test_calc_selection_from_num_sweeps_monotonic():
    res = [
        calc_selection_from_num_sweeps(4e7, 1000, num_sweeps)
        for num_sweeps in (2.0, 4.0, 7.0)
    ]
    assert res == sorted(res)
    assert 0.03 < res[1] < 0.06
