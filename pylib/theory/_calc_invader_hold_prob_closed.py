import numpy as np

from ._calc_load_params import calc_load_params
from ._calc_sweep_time import calc_sweep_time


def calc_invader_hold_prob_closed(N: float, K: float, **kwargs) -> float:
    """P(no nonmutator lineage establishes at the fitness peak K before
    nonmutators are swept out), for a single invading mutator lineage that
    has hitchhiked on its first beneficial mutation, in closed form.

    Counterpart to `calc_hold_prob_closed` for a start at x0 = 1/N, as
    considered by Raynes et al. (2018).

    K = 1. Nonmutators can establish the one available mutation while the
    mutator clone is being founded, which takes about 1 / delta generations
    given that it happens before load removes the lineage, and while it
    sweeps, a further ln(pi_mut N) / growth_mut generations, so
        P_hold = exp(-N supply_non [1 / delta + ln(pi_mut N) / growth_mut]).

    K >= 2. Nonmutator clones founded alongside the mutator clone share
    fitness class 1 with it. Counting clones founded in the 1 / delta
    generations before it in full and those founded t generations after it
    in proportion exp(-growth_mut t), their number is
        m = N supply_non (1 / delta + 1 / growth_mut),
    and nonmutators' share of class 1 is m / (1 + m). Each further class
    cuts that share by supply_non / supply_mut, and nonmutators one step
    below the peak can reach it for a sweep time T (`calc_sweep_time`), so
        P_hold = exp(-N supply_non a_{K-1} T),
        a_k = m / (1 + m) * (supply_non / supply_mut)^(k - 1).
    """
    if not np.isfinite(K):
        return 1.0
    p = calc_load_params(**kwargs)
    growth = p["growth_mut"]
    if K == 1:
        exposure = (
            1.0 / p["delta"] + np.log(max(p["pi_mut"] * N, 1.0)) / growth
        )
        return float(np.exp(-N * p["supply_non"] * exposure))
    m = N * p["supply_non"] * (1.0 / p["delta"] + 1.0 / growth)
    ratio = p["supply_non"] / p["supply_mut"]
    share = m / (1.0 + m) * ratio ** (int(K) - 2)
    sweep_time = calc_sweep_time(p["supply_mut"], growth, p["pi_mut"])
    return float(np.exp(-N * p["supply_non"] * share * sweep_time))
