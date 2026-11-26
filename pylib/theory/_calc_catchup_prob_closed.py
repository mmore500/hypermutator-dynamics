import typing

import numpy as np

from ._calc_hitchhike_prob_closed import calc_hitchhike_prob_closed
from ._calc_load_params import calc_load_params
from ._calc_sweep_time import calc_sweep_time


def calc_catchup_prob_closed(
    N: float,
    K: int,
    x0: float = 0.5,
    deme_size: typing.Optional[int] = None,
    **kwargs,
) -> typing.Tuple[float, float]:
    """P(some nonmutator lineage establishes at the fitness peak K before
    nonmutators are swept out by adapting mutators), given that mutators
    escape their load, for any start frequency x0, in closed form.

    Once mutators reach the peak they stall, still carrying their load, so
    any nonmutator lineage that catches up there takes over. Mutators fix
    with probability P_escape * (1 - P_catchup). Three steps give the
    expected number of such lineages.

    1. Share law. Equal-fitness mutators and nonmutators compete on equal
       terms, so each new fitness class is divided between them in
       proportion to the rates at which they found it. Each class founded
       multiplies nonmutators' odds against mutators by the supply ratio,
       and conditioning on mutators' escape discounts them by P_hitch:
           o_k = (1 - x0) / x0 * P_hitch * (supply_non / supply_mut)^k.
       Nonmutators' share of class k is a_k = o_k / (1 + o_k).

    2. Exposure. Nonmutators one step below the peak can reach it until
       mutators' final sweep arrives, a time T from `calc_sweep_time` at
       mutators' supply rate supply_mut. One sweep time is used for every
       K; allowing for the smaller founding population at the first step
       changes the upper critical size by about 10%.

    3. Count. The expected number of nonmutator lineages established at the
       peak is N supply_non a_{K-1} T, so
           P_catchup = 1 - exp(-N supply_non a_{K-1} T)
                     = 1 - exp(-N Ub C_K).

    For a start at appreciable frequency in a large population P_hitch is
    one; for a single invader, x0 = 1 / N, conditioning on escape matters at
    every N. Population structure enters only through the sweep time T. Pass
    `deme_size` for a large 2D lattice of demes; the default is well-mixed.

    Returns (P_catchup, C_K).
    """
    if deme_size is not None and N <= deme_size:  # a single deme
        deme_size = None
    Ub = kwargs.get("Ub", 1e-6)
    p = calc_load_params(**kwargs)
    hitch = calc_hitchhike_prob_closed(N, x0, **kwargs)
    ratio = p["supply_non"] / p["supply_mut"]
    odds = (1.0 - x0) / x0 * hitch * ratio ** (K - 1)
    share_non = odds / (1.0 + odds)
    sweep_time = calc_sweep_time(
        p["supply_mut"], p["growth_mut"], p["pi_mut"], deme_size
    )
    num_established = N * p["supply_non"] * share_non * sweep_time
    return float(-np.expm1(-num_established)), float(
        num_established / (N * Ub)
    )
