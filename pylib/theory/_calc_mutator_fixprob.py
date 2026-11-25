import numpy as np

from ._calc_escape_prob import calc_escape_prob
from ._calc_hold_prob import calc_hold_prob


def calc_mutator_fixprob(
    N: float,
    K: float,
    x0: float = 0.5,
    sweep_slowdown: float = 1.0,
    **kwargs,
) -> float:
    """Fixation probability of mutators started at frequency x0 in a
    population of size N with K available beneficial mutations.

    P_fix = P_escape(N) * P_hold(N, K); normalized N P_fix is P_fix / x0.

    Under the simulated tournament selection scheme, an individual's expected
    offspring number is W = 1 + s (F_below - F_above), with F_below and
    F_above the population fractions with strictly lower and higher score.
    Selection is rank-based: any score advantage is worth s.

    The default `sweep_slowdown` of 1 describes a well-mixed population. For
    a structured population, pass the factor by which sweeps are slowed; see
    `calc_hold_prob`. Structure is taken to leave P_escape unchanged.
    """
    p_escape = calc_escape_prob(N, x0, **kwargs)
    if not np.isfinite(K):
        return p_escape
    p_hold, __ = calc_hold_prob(
        N, int(K), x0, sweep_slowdown=sweep_slowdown, **kwargs
    )
    return p_escape * p_hold
