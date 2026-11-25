import pytest

from pylib.theory._calc_denovo_fixprob import calc_denovo_fixprob
from pylib.theory._calc_mutator_fixprob import calc_mutator_fixprob
from pylib.theory._resimulate_wellmixed import resimulate_wellmixed


@pytest.mark.parametrize(
    "N, K",
    [(256, 1), (2304, 2), (186624, 1), (3162278, 2)],
)
def test_resimulate_wellmixed_5050_vs_theory(N: int, K: int):
    res = resimulate_wellmixed(N, K, n_rep=400, seed=1)
    assert res.mean() == pytest.approx(calc_mutator_fixprob(N, K), abs=0.1)


def test_resimulate_wellmixed_denovo_vs_theory():
    res = resimulate_wellmixed(3162278, 10, n_rep=200, x0=0.0, mu=1e-6, seed=1)
    assert res.mean() == pytest.approx(
        calc_denovo_fixprob(3162278, 10), abs=0.12
    )


def test_resimulate_wellmixed_deterministic():
    a = resimulate_wellmixed(2304, 1, n_rep=50, seed=7)
    b = resimulate_wellmixed(2304, 1, n_rep=50, seed=7)
    assert (a == b).all()
