import typing

import numpy as np

from ._calc_load_params import calc_load_params
from ._calc_sweep_time import calc_sweep_time
from ._calc_wave_speed import calc_wave_speed


def calc_denovo_critical_potential_closed(
    N: float,
    deme_size: typing.Optional[int] = None,
    mu: float = 1e-6,
    **kwargs,
) -> float:
    """Number of available beneficial mutations K* at which mutators
    arising de novo at rate `mu` fix half the time, in closed form, for
    populations large enough that sweeps overlap.

    Well-mixed (`deme_size` None). Founders of each sweep arise within about
    1 / s generations, when mutators number mu / s of their class. By the
    share law of `calc_hold_prob_closed`, mutators' odds in each new class
    then multiply by supply_mut / supply_non until nonmutators' share of the
    class below the peak, a, is too small to catch up,
    N supply_non a T = ln 2:

        K* = 1 + ln[N supply_non T s / (mu ln 2)]
                 / ln(supply_mut / supply_non).

    2D lattice of demes. A mutator lineage that gets ahead adapts faster
    than its surroundings but spreads no faster, so mutators fix only if
    their domains, founded at rate J per deme per generation and spreading
    at speed v, cover all N / deme_size demes before nonmutators climb K
    steps at one sweep time T_non each:

        K* = {3 ln[N / (deme_size ln 2)] / (pi J v^2)}^(1/3) / T_non.

    J is deme_size (mu / delta) supply_mut, discounted for mutators being
    purged by each nonmutator sweep and rebuilding toward mu / delta.

    K* depends only logarithmically on N and mu, and is far larger in the
    structured case. In small populations sweeps do not overlap and mutators
    instead fix with probability 1 - (1 - rho)^K; see
    `calc_denovo_step_prob`.
    """
    p = calc_load_params(**kwargs)
    if deme_size is None:
        sweep_time = calc_sweep_time(
            p["supply_mut"], p["growth_mut"], p["pi_mut"]
        )
        return float(
            1.0
            + np.log(
                N
                * p["supply_non"]
                * sweep_time
                * p["growth_non"]
                / (mu * np.log(2.0)),
            )
            / np.log(p["supply_mut"] / p["supply_non"]),
        )
    sweep_time_non = calc_sweep_time(
        p["supply_non"], p["growth_non"], p["pi_non"], deme_size
    )
    turnover = p["delta"] * sweep_time_non
    rate_lead = (
        deme_size
        * (mu / p["delta"])
        * p["supply_mut"]
        * (1.0 - (1.0 - np.exp(-turnover)) / turnover)
    )
    speed = calc_wave_speed(deme_size, p["growth_mut"])
    cover_time = (
        3.0
        * np.log(N / (deme_size * np.log(2.0)))
        / (np.pi * rate_lead * speed**2)
    ) ** (1.0 / 3.0)
    return float(cover_time / sweep_time_non)
