import typing

import numpy as np

from ._calc_hold_prob_closed import calc_hold_prob_closed
from ._calc_load_params import calc_load_params


def calc_ncrit_closed(
    K: float,
    which: typing.Literal["lower", "upper"],
    x0: float = 0.5,
    deme_size: typing.Optional[int] = None,
    **kwargs,
) -> float:
    """Population size where mutator fixation probability crosses its
    neutral expectation x0, in closed form.

    "lower": above N = delta / supply_mut, mutators hitchhike before their
    load removes them (the sign inversion of Raynes et al. 2018). This holds
    for any x0, any structure, and any adaptive potential.

    "upper": above N = ln(1 / x0) / (Ub C_K), nonmutators catch up at the
    fitness peak before they are swept out; see `calc_hold_prob_closed`.
    Each further available beneficial mutation raises this threshold by the
    factor supply_mut / supply_non, about the mutator strength. With
    unlimited adaptive potential there is no upper threshold.
    """
    if which == "lower":
        return float(calc_load_params(**kwargs)["N_hitch"])
    if not np.isfinite(K):
        return np.nan
    Ub = kwargs.get("Ub", 1e-6)
    # C_K does not depend on N; use one large enough to span many demes
    __, c = calc_hold_prob_closed(1e12, int(K), x0, deme_size, **kwargs)
    return float(np.log(1.0 / x0) / (Ub * c))
