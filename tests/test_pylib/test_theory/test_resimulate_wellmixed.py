import pytest

from pylib.theory._mutator_fixprob import calc_mutator_fixprob
from pylib.theory._resimulate_wellmixed import resimulate_wellmixed


@pytest.mark.parametrize(
    "N, K",
    [(256, 1), (2304, 2), (186624, 1), (3162278, 2)],
)
def test_resimulate_wellmixed_vs_theory(N: int, K: int):
    res = resimulate_wellmixed(N, K, n_rep=400, seed=1)
    assert res.mean() == pytest.approx(calc_mutator_fixprob(N, K), abs=0.1)
