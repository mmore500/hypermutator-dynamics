import typing

import numpy as np

from ._calc_escape_prob_closed import calc_escape_prob_closed
from ._calc_hold_prob_closed import calc_hold_prob_closed


def calc_mutator_fixprob_closed(
    N: float,
    K: float,
    x0: float = 0.5,
    deme_size: typing.Optional[int] = None,
    **kwargs,
) -> float:
    """Fixation probability of mutators started at frequency x0 in a
    population of size N with K available beneficial mutations, in closed
    form.

        P_fix = P_escape(N) * P_hold(N, K)

    Mutators must first escape their deleterious load by hitchhiking or
    drift (`calc_escape_prob_closed`, after Raynes et al. 2018), then hold
    their lead until nonmutators are gone (`calc_hold_prob_closed`).
    Establishment of beneficial mutations is unaffected by population
    structure (Maruyama 1970), so structure, set by `deme_size`, enters only
    through P_hold. Normalized N P_fix is P_fix / x0.
    """
    p_escape = calc_escape_prob_closed(N, x0, **kwargs)
    if not np.isfinite(K):
        return p_escape
    return (
        p_escape * calc_hold_prob_closed(N, int(K), x0, deme_size, **kwargs)[0]
    )
