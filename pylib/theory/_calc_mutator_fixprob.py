import numpy as np

from ._calc_escape_prob import calc_escape_prob
from ._calc_hold_prob import calc_hold_prob


def calc_mutator_fixprob(
    N: float,
    K: float,
    x0: float = 0.5,
    **kwargs,
) -> float:
    """Fixation probability of mutators started at frequency x0 in a
    well-mixed population of size N with K available beneficial mutations,
    by numerical refinement of `calc_mutator_fixprob_closed`.

    P_fix = P_escape(N) * P_hold(N, K); normalized N P_fix is P_fix / x0.
    """
    p_escape = calc_escape_prob(N, x0, **kwargs)
    if not np.isfinite(K):
        return p_escape
    return p_escape * calc_hold_prob(N, int(K), x0, **kwargs)[0]
