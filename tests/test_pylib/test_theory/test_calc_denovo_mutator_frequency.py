import pytest

from pylib.theory._calc_denovo_mutator_frequency import (
    calc_denovo_mutator_frequency,
)
from pylib.theory._calc_load_params import calc_load_params


def test_calc_denovo_mutator_frequency_balance_limit():
    # long intervals between sweeps: mutation-selection balance
    delta = calc_load_params()["delta"]
    assert calc_denovo_mutator_frequency(1e7) == pytest.approx(
        1e-6 / delta, rel=1e-3
    )


def test_calc_denovo_mutator_frequency_frequent_sweeps_limit():
    # short intervals: mutators accumulate for about one interval, mu T
    assert calc_denovo_mutator_frequency(1.0) == pytest.approx(1e-6, rel=0.02)


def test_calc_denovo_mutator_frequency_supply_over_loss():
    delta = calc_load_params()["delta"]
    assert calc_denovo_mutator_frequency(100.0) == pytest.approx(
        1e-6 / (delta + 0.01)
    )


def test_calc_denovo_mutator_frequency_proportional_to_mu():
    assert calc_denovo_mutator_frequency(100.0, mu=1e-5) == pytest.approx(
        10 * calc_denovo_mutator_frequency(100.0, mu=1e-6)
    )
