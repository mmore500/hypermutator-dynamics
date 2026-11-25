"""Semi-analytic fixation probability of mutators arising de novo in a
nonmutator population with finite adaptive potential.

Counterpart to `_mutator_fixprob` for the "de novo" treatment, where the
population is founded by nonmutators and each genome converts to mutator
with probability `mu` per generation.

Mutators fix by a two-stage relay,

1. *lead*: some mutator lineage establishes a beneficial mutation that puts
   it strictly ahead of every nonmutator, then
2. *hold*: no nonmutator lineage establishes at the fitness peak K before
   nonmutators are swept out (as in the 50/50 theory).

Nonmutators are common, so their adaptation is treated as a mean-field
wave. Mutators are rare until they lead, so their lineages are treated as a
multitype branching process riding that wave.
"""

import functools
import typing

import numpy as np

from ._mutator_fixprob import calc_establishment_prob

# generations elapsed since current nonmutator nose class was founded, at
# which hold probability is tabulated
_HOLD_TAUS = np.array([0, 15, 30, 45, 60, 80, 100, 130, 170, 250, 400])
_HOLD_KMAX = 7  # hold is near-certain with more levels than this remaining
_HOLD_NOSE = 2  # nonmutator nose level used to tabulate hold probability
_MAX_DEL = 4


def _run_meanfield(
    f: np.ndarray,
    N: float,
    K: int,
    s: float,
    Ub: float,
    Ud: float,
    M: float,
    block_peak: bool,
    should_stop: typing.Callable,
    max_gen: int,
    record_states: bool = False,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, typing.List[np.ndarray]]:
    """Mean-field tournament dynamics of genome class frequencies
    f[is mutator, n ben, n del], with semi-deterministic seeding of each
    type's leading fitness class: a new class opens, with 1 / pi individuals,
    once the cumulative expected number of established founders reaches one.

    Returns per-generation growth factors W[t, b] of unloaded genomes,
    nonmutator nose level nose[t], unloaded nonmutator frequencies z[t, b],
    and (optionally) the state at the start of each generation.
    """
    f = f.copy()
    max_del = f.shape[2] - 1
    p_ben = [-np.expm1(-Ub), -np.expm1(-M * Ub)]
    p_del = [-np.expm1(-Ud), -np.expm1(-M * Ud)]
    score = (
        np.arange(K + 1)[:, None] - np.arange(max_del + 1)[None, :] + max_del
    )
    n_score = K + max_del + 1
    top = [
        int(np.flatnonzero(f[m].sum(axis=1)).max()) if f[m].any() else -1
        for m in (0, 1)
    ]
    n_established = [0.0, 0.0]

    W_score = np.ones(n_score)
    Ws, noses, zs, states = [], [], [], []
    for __ in range(max_gen):
        if record_states:
            states.append(f.copy())
        for m in 0, 1:
            n = top[m]
            if n < 0:
                continue
            flux = f[m, :n] * p_ben[m]
            f[m, :n] -= flux
            f[m, 1 : n + 1] += flux
            if n < (K - 1 if m == 0 and block_peak else K):
                pi = calc_establishment_prob(
                    W_score[n + 1 + max_del] * (1.0 - p_del[m]),
                )
                n_established[m] += N * f[m, n, 0] * p_ben[m] * pi
                if n_established[m] >= 1.0:
                    seed = min(1.0 / (N * pi), f[m, n, 0])
                    f[m, n, 0] -= seed
                    f[m, n + 1, 0] += seed
                    top[m] += 1
                    n_established[m] = 0.0
            flux = f[m, :, :-1] * p_del[m]
            f[m, :, :-1] -= flux
            f[m, :, 1:] += flux

        totals = np.bincount(
            score.ravel(), weights=f.sum(axis=0).ravel(), minlength=n_score
        )
        below = np.concatenate([[0.0], np.cumsum(totals)[:-1]])
        W_score = 1.0 + s * (2.0 * below + totals - 1.0)
        Ws.append(W_score[max_del:].copy())
        noses.append(top[0])
        zs.append(f[0, :, 0].copy())
        f *= W_score[score][None]
        f /= f.sum()
        if should_stop(f, top, Ws):
            break

    return np.array(Ws), np.array(noses), np.array(zs), states


def _calc_hold_from_state(
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

    W, __, __, __ = _run_meanfield(
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


@functools.lru_cache(maxsize=None)
def calc_lead_hold_table(
    N: float,
    s: float = 0.1,
    Ub: float = 1e-6,
    Ud: float = 1e-4,
    M: float = 100.0,
) -> np.ndarray:
    """Hold probability H[k, i] for a single mutator lineage that takes the
    lead `_HOLD_TAUS[i]` generations after the nonmutators' leading class was
    founded, with k beneficial mutations available beyond that class.

    Timing matters because nonmutators are caught mid-sweep: early on, most
    still trail their own leading class and so have further to climb.
    """
    K = _HOLD_NOSE + 1
    f = np.zeros((2, K + 1, _MAX_DEL + 1))
    f[0, 0, 0] = 1.0
    __, noses, __, states = _run_meanfield(
        f,
        N,
        K,
        s,
        Ub,
        Ud,
        M,
        False,
        lambda f, top, Ws: top[0] == K,
        max_gen=10**7,
        record_states=True,
    )
    t_open = int(np.argmax(noses == _HOLD_NOSE))
    taus = np.minimum(_HOLD_TAUS, len(noses) - t_open - 1)
    pi_mut = calc_establishment_prob((1.0 + s) * np.exp(-M * Ud))

    res = np.ones((_HOLD_KMAX + 1, len(taus)))
    for k in range(1, _HOLD_KMAX + 1):
        for i, tau in enumerate(taus):
            f = np.zeros((2, _HOLD_NOSE + k + 1, _MAX_DEL + 1))
            f[0, : _HOLD_NOSE + 1] = states[t_open + tau][0, : _HOLD_NOSE + 1]
            f[1, _HOLD_NOSE + 1, 0] = 1.0 / (N * pi_mut)
            f /= f.sum()
            res[k, i] = _calc_hold_from_state(
                f, N, _HOLD_NOSE + k, s, Ub, Ud, M
            )
    return res


@functools.lru_cache(maxsize=None)
def _calc_lead_hold_initial(
    N: float, k: int, s: float, Ub: float, Ud: float, M: float
) -> float:
    """Hold probability for a mutator lineage that takes the lead before
    nonmutators have gained any beneficial mutation."""
    f = np.zeros((2, k + 1, _MAX_DEL + 1))
    f[0, 0, 0] = 1.0
    pi_mut = calc_establishment_prob((1.0 + s) * np.exp(-M * Ud))
    f[1, 1, 0] = 1.0 / (N * pi_mut)
    f /= f.sum()
    return _calc_hold_from_state(f, N, k, s, Ub, Ud, M)


def calc_denovo_drift_fixprob(
    N: float,
    Ud: float = 1e-4,
    M: float = 100.0,
    mu: float = 1e-6,
    n_gen: int = 500000,
) -> float:
    """P(some de novo mutator lineage fixes by drift alone in n_gen
    generations), per Kimura with selection -delta from a single copy."""
    delta = np.exp(-Ud) - np.exp(-M * Ud)
    if 2.0 * N * delta > 700:
        return 0.0
    p_fix = np.expm1(2.0 * delta) / np.expm1(2.0 * N * delta)
    return float(-np.expm1(-N * mu * p_fix * n_gen))


def calc_denovo_step_prob(
    s: float = 0.1,
    Ub: float = 1e-6,
    Ud: float = 1e-4,
    M: float = 100.0,
    mu: float = 1e-6,
) -> float:
    """Closed-form small-N limit: probability that a given beneficial
    mutation is swept by a mutator rather than a nonmutator.

    rho = mu Phh / (Ub pi + mu Phh), with Phh the single-mutant hitchhiking
    probability (Raynes et al. eq. 1c). Then Pfix = 1 - (1 - rho)^K.
    """
    Udm = M * Ud
    delta = np.exp(-Ud) - np.exp(-Udm)
    q0 = 1.0 - (1.0 - np.exp(-Udm)) / s
    pi_mut = calc_establishment_prob((1.0 + s) * np.exp(-Udm))
    pi_non = calc_establishment_prob((1.0 + s) * np.exp(-Ud))
    t_lag = 1.0 / (s - Udm)
    p_hitch = M * Ub * q0 * pi_mut * (1.0 / delta + t_lag)
    return float(mu * p_hitch / (Ub * pi_non + mu * p_hitch))


def calc_denovo_fixprob(
    N: float,
    K: int,
    s: float = 0.1,
    Ub: float = 1e-6,
    Ud: float = 1e-4,
    M: float = 100.0,
    mu: float = 1e-6,
    n_gen: int = 500000,
) -> float:
    """Probability that mutators arising de novo at rate `mu` are fixed
    after `n_gen` generations, in a well-mixed population of size N with K
    available beneficial mutations.

    With q_b(t) the probability that a mutator in fitness class b at
    generation t founds no lineage that takes and holds the lead,

        q_b(t) = ud' + (1 - ud') [(1 - ub') G_b(t) + ub' X_b(t)],
        G_b(t) = exp(-W_b(t) (1 - q_b(t + 1))),
        X_b(t) = G_{b+1}(t) below the nonmutator nose class n(t), and
                 1 - pi' H(t) at the nose,

    where W is the nonmutators' mean-field wave and H the hold probability.
    Summing over de novo mutator arrivals,

        P_fix = 1 - exp(-sum_t sum_b N z_b(t) mu (1 - q_b(t))).
    """
    f = np.zeros((2, K + 1, _MAX_DEL + 1))
    f[0, 0, 0] = 1.0
    W, nose, z, __ = _run_meanfield(
        f,
        N,
        K,
        s,
        Ub,
        Ud,
        M,
        False,
        lambda f, top, Ws: top[0] == K and f[0, K, 0] > 0.999,
        max_gen=n_gen,
    )

    # hold probability, should a mutator take the lead at generation t
    hold_table = calc_lead_hold_table(N, s, Ub, Ud, M)
    is_new_nose = np.concatenate([[True], nose[1:] != nose[:-1]])
    t_open = np.maximum.accumulate(
        np.where(is_new_nose, np.arange(len(nose)), 0),
    )
    hold = np.ones(len(nose))
    for t, n in enumerate(nose):
        k = K - n
        if k <= 0:
            hold[t] = 0.0
        elif k > _HOLD_KMAX:
            hold[t] = 1.0
        elif n == 0:
            hold[t] = _calc_lead_hold_initial(N, int(k), s, Ub, Ud, M)
        else:
            hold[t] = np.interp(t - t_open[t], _HOLD_TAUS, hold_table[k])

    pbm, pdm = -np.expm1(-M * Ub), -np.expm1(-M * Ud)
    pi_lead = calc_establishment_prob((1.0 + s) * (1.0 - pdm))
    levels = np.arange(K + 1)
    q = np.ones(K + 1)
    n_success = 0.0
    for t in reversed(range(len(W))):
        G = np.exp(-W[t] * (1.0 - q))
        X = np.append(G[1:], G[-1])
        if nose[t] < K:
            X[nose[t]] = 1.0 - pi_lead * hold[t]
        q = pdm + (1.0 - pdm) * ((1.0 - pbm) * G + pbm * X)
        q[levels > nose[t]] = 1.0
        n_success += N * mu * np.dot(z[t], 1.0 - q)

    p_drift = calc_denovo_drift_fixprob(N, Ud, M, mu, n_gen)
    return float(1.0 - np.exp(-n_success) * (1.0 - p_drift))
