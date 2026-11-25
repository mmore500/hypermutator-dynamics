from ._calc_mutator_fixprob_closed import calc_mutator_fixprob_closed


def calc_invader_fixprob_closed(N: float, K: float, **kwargs) -> float:
    """Fixation probability of a single mutator lineage invading a
    population of size N with K available beneficial mutations, in closed
    form: `calc_mutator_fixprob_closed` at start frequency x0 = 1 / N.

    With unlimited adaptive potential (K infinite), this is eq. 1 of Raynes
    et al. (2018). Normalized N P_fix exceeds one where mutators are
    favored.
    """
    return calc_mutator_fixprob_closed(N, K, x0=1.0 / N, **kwargs)
