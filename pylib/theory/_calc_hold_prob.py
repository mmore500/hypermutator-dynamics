import typing

import numpy as np

from ._calc_establishment_prob import calc_establishment_prob
from ._calc_mutator_wave import calc_mutator_wave


def calc_hold_prob(
    N: float,
    K: int,
    x0: float = 0.5,
    s: float = 0.1,
    Ub: float = 1e-6,
    Ud: float = 1e-4,
    M: float = 100.0,
    finite_wave: bool = True,
    sweep_slowdown: float = 1.0,
) -> typing.Tuple[float, float]:
    """P(no nonmutator lineage establishes at the fitness peak K before
    nonmutators are swept out by adapting mutators).

    Nonmutators carrying beneficial mutations are rare until they win, so
    they are treated as a multitype branching process riding the mutators'
    mean-field wave (`calc_mutator_wave`). With Q_j(t) the probability that a
    nonmutator with j beneficial mutations at generation t leaves no lineage
    established at the peak,

        Q_j(t) = (1 - ub - ud) G_j(t) + ub G_{j+1}(t) + ud,
        G_j(t) = exp(-W_j(t) (1 - Q_j(t + 1))),

    solved backward from the post-sweep stationary state. Then

        P_hold = Q_0(0)^(N (1 - x0)) = exp(-N Ub C_K).

    Population structure is accommodated through `sweep_slowdown`, the
    factor tau by which selective sweeps take longer than in a well-mixed
    population (e.g., the ratio of fixation times). Structure leaves
    establishment probabilities and the mutators' load unchanged, so
    stretching the sweep clock by tau is equivalent to solving the well-mixed
    problem with s / tau and tau * Ub (which holds mutation supply per sweep
    fixed). Nonmutators get tau times longer to catch up, during which the
    mutators' load costs them tau times more.

    Returns (P_hold, C_K).
    """
    Ub_actual = Ub
    s, Ub = s / sweep_slowdown, Ub * sweep_slowdown
    W = calc_mutator_wave(
        K, N if finite_wave else np.inf, x0=x0, s=s, Ub=Ub, Ud=Ud, M=M
    )
    pb, pd = -np.expm1(-Ub), -np.expm1(-Ud)

    def step(Q: np.ndarray, W_t: np.ndarray) -> np.ndarray:
        G = np.exp(-W_t * (1.0 - Q))
        res = (1.0 - pb - pd) * G + pd
        res[:-1] += pb * G[1:]
        res[-1] += pb * G[-1]  # capped: beneficial mutation has no effect
        return res

    Q = np.ones(K + 1)
    Q[K] = 1.0 - calc_establishment_prob(W[-1, K] * (1.0 - pd))
    for __ in range(5000):  # stationary post-sweep environment
        Q_ = step(Q, W[-1])
        if np.abs(Q_ - Q).max() < 1e-16:
            break
        Q = Q_
    for W_t in W[::-1]:
        Q = step(Q, W_t)

    p_success = 1.0 - Q[0]  # per founding nonmutator
    log_p_hold = N * (1.0 - x0) * np.log1p(-p_success)
    return float(np.exp(log_p_hold)), float((1.0 - x0) * p_success / Ub_actual)
