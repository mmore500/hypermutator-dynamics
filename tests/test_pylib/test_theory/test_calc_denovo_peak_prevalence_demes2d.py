import pytest

from pylib.theory._calc_denovo_peak_prevalence_demes2d import (
    calc_denovo_peak_prevalence_demes2d,
)


@pytest.mark.parametrize(
    "K, expected",
    # approximate peak mutator prevalence observed at 750 x 994 demes,
    # from wse-denovo-spatial2d-explicitsite-timeseries-2025-02-07-peaksweep
    [(2, 0.04), (4, 0.25), (6, 0.70), (8, 0.92), (10, 1.0)],
)
def test_calc_denovo_peak_prevalence_demes2d_vs_published(
    K: int, expected: float
):
    assert calc_denovo_peak_prevalence_demes2d(190848000, K) == pytest.approx(
        expected, abs=0.1
    )


def test_calc_denovo_peak_prevalence_demes2d_monotonic():
    res = [
        calc_denovo_peak_prevalence_demes2d(15116544, K) for K in (2, 6, 10)
    ]
    assert res == sorted(res)
    assert 0.0 <= res[0] and res[-1] <= 1.0
