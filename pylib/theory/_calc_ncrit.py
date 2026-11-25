import typing

import numpy as np
from scipy import optimize as sp_optimize

from ._calc_mutator_fixprob import calc_mutator_fixprob


def calc_ncrit(
    K: float,
    which: typing.Literal["lower", "upper"],
    x0: float = 0.5,
    **kwargs,
) -> float:
    """Population size where mutator fixation probability crosses its neutral
    expectation x0 (i.e., normalized N P_fix = 1).

    "lower": mutators become favored (sign inversion of Raynes et al.);
    "upper": mutators become disfavored again, for want of adaptive potential.
    Returns nan if there is no such crossing.
    """

    def objective(log_N: float) -> float:
        return calc_mutator_fixprob(10**log_N, K, x0, **kwargs) - x0

    log_Ns = np.arange(1.5, 14.01, 0.25)
    vals = np.array([objective(log_N) for log_N in log_Ns])
    if which == "lower":
        (candidates,) = np.nonzero((vals[:-1] < 0) & (vals[1:] >= 0))
    else:
        (candidates,) = np.nonzero((vals[:-1] >= 0) & (vals[1:] < 0))
    if not len(candidates):
        return np.nan
    i = candidates[0] if which == "lower" else candidates[-1]
    return float(10 ** sp_optimize.brentq(objective, log_Ns[i], log_Ns[i + 1]))
