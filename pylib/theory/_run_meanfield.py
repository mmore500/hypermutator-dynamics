import typing

import numpy as np

from ._calc_establishment_prob import calc_establishment_prob


def run_meanfield(
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
