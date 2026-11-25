import functools

import numpy as np

from ._calc_establishment_prob import calc_establishment_prob
from ._calc_hold_from_state import calc_hold_from_state
from ._denovo_constants import MAX_DEL


@functools.lru_cache(maxsize=None)
def calc_lead_hold_initial(
    N: float, k: int, s: float, Ub: float, Ud: float, M: float
) -> float:
    """Hold probability for a mutator lineage that takes the lead before
    nonmutators have gained any beneficial mutation."""
    f = np.zeros((2, k + 1, MAX_DEL + 1))
    f[0, 0, 0] = 1.0
    pi_mut = calc_establishment_prob((1.0 + s) * np.exp(-M * Ud))
    f[1, 1, 0] = 1.0 / (N * pi_mut)
    f /= f.sum()
    return calc_hold_from_state(f, N, k, s, Ub, Ud, M)
