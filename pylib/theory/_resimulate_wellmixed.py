import typing

import numpy as np


def resimulate_wellmixed(
    N: int,
    K: int,
    n_rep: int,
    x0: float = 0.5,
    s: float = 0.1,
    Ub: float = 1e-6,
    Ud: float = 1e-4,
    M: float = 100.0,
    mu: float = 0.0,
    max_del: int = 4,
    max_gen: int = 500000,
    seed: typing.Optional[int] = None,
) -> np.ndarray:
    """Exact-in-distribution resimulation of the well-mixed model, tracking
    counts per (mutator, n ben, n del) class instead of individuals, so cost
    is independent of population size N.

    Mutators begin at frequency `x0` and thereafter arise de novo from
    nonmutators with per-generation probability `mu`. Use x0=0.5, mu=0 for
    the 50/50 treatment and x0=0, mu=1e-6 for the de novo treatment.

    Returns boolean array over replicates: did mutators win?

    Two approximations: at most one beneficial and one deleterious mutation
    per genome per generation; genomes with more than `max_del` deleterious
    mutations are lumped. A replicate is called for nonmutators once >= 1000
    unloaded nonmutators sit at the fitness peak (their loss is then
    vanishingly unlikely for N >~ 1000); otherwise replicates run until
    mutators fix or, absent de novo mutators, go extinct.
    """
    rng = np.random.default_rng(seed)
    shape = (2, K + 1, max_del + 1)
    counts = np.zeros((n_rep, *shape), dtype=np.int64)
    counts[:, 1, 0, 0] = rng.binomial(N, x0, size=n_rep)
    counts[:, 0, 0, 0] = N - counts[:, 1, 0, 0]

    score = (
        np.arange(K + 1)[:, None] - np.arange(max_del + 1)[None, :] + max_del
    )
    score = np.broadcast_to(score, shape).ravel()
    is_score = score[:, None] == np.arange(K + max_del + 1)[None, :]
    is_mut = np.broadcast_to(np.array([0, 1])[:, None, None], shape).ravel()
    p_ben = -np.expm1(-Ub * np.array([1.0, M]))
    p_del = -np.expm1(-Ud * np.array([1.0, M]))

    p_mu = -np.expm1(-mu)

    res = np.zeros(n_rep, dtype=bool)
    active = np.arange(n_rep)
    for __ in range(max_gen):
        for b in reversed(range(K)):
            k = rng.binomial(counts[:, :, b, :], p_ben[None, :, None])
            counts[:, :, b, :] -= k
            counts[:, :, b + 1, :] += k
        for d in reversed(range(max_del)):
            k = rng.binomial(counts[:, :, :, d], p_del[None, :, None])
            counts[:, :, :, d] -= k
            counts[:, :, :, d + 1] += k
        if mu:
            k = rng.binomial(counts[:, 0], p_mu)
            counts[:, 0] -= k
            counts[:, 1] += k

        flat = counts.reshape(len(active), -1) / N
        totals = flat @ is_score
        below = np.cumsum(totals, axis=1) - totals
        W = 1.0 + s * (2.0 * below + totals - 1.0)
        pvals = flat * W[:, score]
        pvals /= pvals.sum(axis=1, keepdims=True)
        flat = rng.multinomial(N, pvals)
        counts = flat.reshape(len(active), *shape)

        n_mut = flat @ is_mut
        mut_won = n_mut == N
        non_won = (counts[:, 0, K, 0] >= 1000) | ((n_mut == 0) & (mu == 0))
        res[active[mut_won]] = True
        keep = ~(mut_won | non_won)
        if not keep.all():
            active, counts = active[keep], counts[keep]
        if not len(active):
            break

    return res
