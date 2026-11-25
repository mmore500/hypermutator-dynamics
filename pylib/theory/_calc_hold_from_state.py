import numpy as np

from ._calc_establishment_prob import calc_establishment_prob
from ._run_meanfield import run_meanfield


def calc_hold_from_state(
    f: np.ndarray,
    N: float,
    K: int,
    s: float,
    Ub: float,
    Ud: float,
    M: float,
) -> float:
    """P(no nonmutator lineage establishes at peak K), starting from genome
    class frequencies f. See `_mutator_fixprob.calc_hold_prob`."""

    def should_stop(f: np.ndarray, top: list, Ws: list) -> bool:
        return (
            len(Ws) > 50
            and f[0].sum() < 1e-13
            and np.abs(Ws[-1] - Ws[-2]).max() < 1e-12
        )

    W, __, __, __ = run_meanfield(
        f, N, K, s, Ub, Ud, M, True, should_stop, max_gen=50000
    )
    pb, pd = -np.expm1(-Ub), -np.expm1(-Ud)

    def step(Q: np.ndarray, W_t: np.ndarray) -> np.ndarray:
        G = np.exp(-W_t * (1.0 - Q))
        res = (1.0 - pb - pd) * G + pd
        res[:-1] += pb * G[1:]
        res[-1] += pb * G[-1]
        return res

    Q = np.ones(K + 1)
    Q[K] = 1.0 - calc_establishment_prob(W[-1, K] * (1.0 - pd))
    for __ in range(5000):
        Q_ = step(Q, W[-1])
        if np.abs(Q_ - Q).max() < 1e-16:
            break
        Q = Q_
    for W_t in W[::-1]:
        Q = step(Q, W_t)

    return float(np.exp(-N * np.dot(f[0, :, 0], 1.0 - Q)))
