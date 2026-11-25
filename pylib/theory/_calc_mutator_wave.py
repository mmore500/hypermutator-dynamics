import numpy as np

from ._calc_establishment_prob import calc_establishment_prob


def calc_mutator_wave(
    K: int,
    N: float = np.inf,
    x0: float = 0.5,
    s: float = 0.1,
    Ub: float = 1e-6,
    Ud: float = 1e-4,
    M: float = 100.0,
    max_del: int = 6,
    tol: float = 1e-13,
) -> np.ndarray:
    """Mean-field adaptation of mutators toward the fitness peak K, in the
    presence of nonmutators that (counterfactually) never gain a beneficial
    mutation.

    Returns array W[t, j], the tournament growth factor experienced in
    generation t by a genome with j beneficial and no deleterious mutations.

    For finite N, successive mutator fitness classes are seeded
    semi-deterministically: class j opens, with 1 / pi individuals, once the
    cumulative expected number of established founders reaches one. The clock
    starts at establishment of the first mutator beneficial mutation.
    """
    pbm, pdm = -np.expm1(-M * Ub), -np.expm1(-M * Ud)
    pdn = -np.expm1(-Ud)
    f = np.zeros((2, K + 1, max_del + 1))  # [is mutator, n ben, n del]
    f[0, 0, 0], f[1, 0, 0] = 1.0 - x0, x0
    score = (
        np.arange(K + 1)[:, None] - np.arange(max_del + 1)[None, :] + max_del
    )
    n_score = K + max_del + 1
    n_established = np.zeros(K + 1)
    is_open = np.full(K + 1, not np.isfinite(N))
    is_open[0] = True
    if np.isfinite(N) and K >= 1:
        is_open[1] = True
        seed = 1.0 / (N * calc_establishment_prob((1.0 + s) * (1.0 - pdm)))
        seed = min(seed, x0)
        f[1, 0, 0] -= seed
        f[1, 1, 0] += seed

    W_score = np.ones(n_score)
    res = []
    while len(res) < 200000:
        mut = f[1]
        for b in reversed(range(K)):
            flux = mut[b] * pbm
            if is_open[b + 1]:
                mut[b] -= flux
                mut[b + 1] += flux
                continue
            pi = calc_establishment_prob(
                W_score[b + 1 + max_del] * (1.0 - pdm)
            )
            n_established[b + 1] += N * flux[0] * pi
            if n_established[b + 1] >= 1.0:
                is_open[b + 1] = True
                seed = min(1.0 / (N * pi), mut[b, 0])
                mut[b, 0] -= seed
                mut[b + 1, 0] += seed
        for genomes, p_del in (f[0], pdn), (f[1], pdm):
            for d in reversed(range(max_del)):
                flux = genomes[:, d] * p_del
                genomes[:, d] -= flux
                genomes[:, d + 1] += flux

        totals = np.bincount(
            score.ravel(), weights=f.sum(axis=0).ravel(), minlength=n_score
        )
        below = np.concatenate([[0.0], np.cumsum(totals)[:-1]])
        W_score = 1.0 + s * (2.0 * below + totals - 1.0)
        res.append(W_score[max_del:].copy())
        f *= W_score[score][None]
        f /= f.sum()

        if (
            len(res) > 50
            and f[0].sum() < tol
            and np.abs(res[-1] - res[-2]).max() < 1e-12
        ):
            break

    return np.array(res)
