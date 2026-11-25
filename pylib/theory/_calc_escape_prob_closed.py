from ._calc_drift_fixprob import calc_drift_fixprob
from ._calc_hitchhike_prob_closed import calc_hitchhike_prob_closed


def calc_escape_prob_closed(N: float, x0: float = 0.5, **kwargs) -> float:
    """Closed-form analog of Raynes et al. eq. 1a for a start at x0."""
    p_drift = calc_drift_fixprob(N, x0, **kwargs)
    p_hitch = calc_hitchhike_prob_closed(N, x0, **kwargs)
    return p_drift + (1.0 - p_drift) * p_hitch
