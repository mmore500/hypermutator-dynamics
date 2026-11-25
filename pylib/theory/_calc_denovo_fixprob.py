import numpy as np

from ._calc_denovo_drift_fixprob import calc_denovo_drift_fixprob
from ._calc_establishment_prob import calc_establishment_prob
from ._calc_lead_hold_initial import calc_lead_hold_initial
from ._calc_lead_hold_table import calc_lead_hold_table
from ._denovo_constants import HOLD_KMAX, HOLD_TAUS, MAX_DEL
from ._run_meanfield import run_meanfield


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
    f = np.zeros((2, K + 1, MAX_DEL + 1))
    f[0, 0, 0] = 1.0
    W, nose, z, __ = run_meanfield(
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
        elif k > HOLD_KMAX:
            hold[t] = 1.0
        elif n == 0:
            hold[t] = calc_lead_hold_initial(N, int(k), s, Ub, Ud, M)
        else:
            hold[t] = np.interp(t - t_open[t], HOLD_TAUS, hold_table[k])

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
