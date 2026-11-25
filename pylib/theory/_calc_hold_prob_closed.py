import typing

import numpy as np

from ._calc_load_params import calc_load_params
from ._calc_sweep_time import calc_sweep_time


def calc_hold_prob_closed(
    N: float,
    K: int,
    x0: float = 0.5,
    deme_size: typing.Optional[int] = None,
    **kwargs,
) -> typing.Tuple[float, float]:
    """P(no nonmutator lineage establishes at the fitness peak K before
    nonmutators are swept out by adapting mutators), in closed form.

    Once mutators reach the peak they stall, still carrying their load, so
    any nonmutator lineage also established there takes over. Three steps
    give the expected number of such lineages.

    1. Share law. Equal-fitness mutators and nonmutators compete on equal
       terms, so each new fitness class is divided between them in
       proportion to the rates at which they found it. Nonmutators' share
       of class k is thus

           a_k = (1 - x0) / x0 * (supply_non / supply_mut)^k,

       beginning from a_0 = 1 - x0.
    2. Exposure. Nonmutators one step below the peak can reach it until
       mutators' final sweep arrives, a time T from `calc_sweep_time`.
    3. Count. The expected number of nonmutator lineages established at the
       peak is N supply_non a_{K-1} T, so

           P_hold = exp(-N supply_non a_{K-1} T) = exp(-N Ub C_K).

    Population structure enters only through the sweep time T. Pass
    `deme_size` for a large 2D lattice of demes; the default is well-mixed.

    Returns (P_hold, C_K).
    """
    if deme_size is not None and N <= deme_size:  # a single deme
        deme_size = None
    Ub = kwargs.get("Ub", 1e-6)
    p = calc_load_params(**kwargs)
    if K == 1:  # mutators' first sweep draws on only their share x0
        share, supply = 1.0 - x0, x0 * p["supply_mut"]
    else:
        ratio = p["supply_non"] / p["supply_mut"]
        share, supply = (1.0 - x0) / x0 * ratio ** (K - 1), p["supply_mut"]
    sweep_time = calc_sweep_time(
        supply, p["growth_mut"], p["pi_mut"], deme_size
    )
    num_established = N * p["supply_non"] * share * sweep_time
    return float(np.exp(-num_established)), float(num_established / (N * Ub))
