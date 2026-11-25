from ._calc_escape_prob_closed import calc_escape_prob_closed
from ._calc_invader_hold_prob_closed import calc_invader_hold_prob_closed


def calc_invader_fixprob_closed(N: float, K: float, **kwargs) -> float:
    """Fixation probability of a single mutator lineage invading a
    population of size N with K available beneficial mutations, in closed
    form.

        P_fix = P_escape(N) * P_hold(N, K)

    With unlimited adaptive potential (K infinite), this is eq. 1 of Raynes
    et al. (2018). Normalized N P_fix exceeds one where mutators are
    favored.
    """
    return calc_escape_prob_closed(
        N, 1.0 / N, **kwargs
    ) * calc_invader_hold_prob_closed(N, K, **kwargs)
