import functools

import numpy as np

from ._calc_establishment_prob import calc_establishment_prob
from ._calc_hold_from_state import calc_hold_from_state
from ._denovo_constants import HOLD_KMAX, HOLD_NOSE, HOLD_TAUS, MAX_DEL
from ._run_meanfield import run_meanfield


@functools.lru_cache(maxsize=None)
def calc_lead_hold_table(
    N: float,
    s: float = 0.1,
    Ub: float = 1e-6,
    Ud: float = 1e-4,
    M: float = 100.0,
) -> np.ndarray:
    """Hold probability H[k, i] for a single mutator lineage that takes the
    lead `HOLD_TAUS[i]` generations after the nonmutators' leading class was
    founded, with k beneficial mutations available beyond that class.

    Timing matters because nonmutators are caught mid-sweep: early on, most
    still trail their own leading class and so have further to climb.
    """
    K = HOLD_NOSE + 1
    f = np.zeros((2, K + 1, MAX_DEL + 1))
    f[0, 0, 0] = 1.0
    __, noses, __, states = run_meanfield(
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
    t_open = int(np.argmax(noses == HOLD_NOSE))
    taus = np.minimum(HOLD_TAUS, len(noses) - t_open - 1)
    pi_mut = calc_establishment_prob((1.0 + s) * np.exp(-M * Ud))

    res = np.ones((HOLD_KMAX + 1, len(taus)))
    for k in range(1, HOLD_KMAX + 1):
        for i, tau in enumerate(taus):
            f = np.zeros((2, HOLD_NOSE + k + 1, MAX_DEL + 1))
            f[0, : HOLD_NOSE + 1] = states[t_open + tau][0, : HOLD_NOSE + 1]
            f[1, HOLD_NOSE + 1, 0] = 1.0 / (N * pi_mut)
            f /= f.sum()
            res[k, i] = calc_hold_from_state(f, N, HOLD_NOSE + k, s, Ub, Ud, M)
    return res
