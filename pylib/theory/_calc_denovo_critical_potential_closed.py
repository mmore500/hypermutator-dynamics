import typing

import numpy as np

from ._calc_denovo_mutator_frequency import calc_denovo_mutator_frequency
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

    Mutators are treated as present at their typical frequency x in an
    adapting nonmutator population (`calc_denovo_mutator_frequency`).

    Well-mixed (`deme_size` None). By the share law of
    `calc_hold_prob_closed` at start frequency x, nonmutators' odds in class
    k are (supply_non / supply_mut)^k / x, and mutators hold their lead half
    the time once nonmutators' share of the class below the peak satisfies
    N supply_non a T = ln 2:
        K* = 1 + ln[N supply_non T / (x ln 2)]
                 / ln(supply_mut / supply_non).

    2D lattice of demes. A mutator lineage that gets ahead adapts faster
    than its surroundings but spreads no faster, so mutators fix only if
    their domains, founded at per-capita rate x supply_mut, cover all
    N / deme_size demes before nonmutators climb K steps at one sweep time
    T_non each. With S from `calc_unswept_prob`, K* solves
        (N / deme_size) S(K* T_non; x supply_mut) = ln 2.

    K* depends only logarithmically on N and mu, and is far larger in the
    structured case. In small populations sweeps do not overlap and mutators
    instead fix with probability 1 - (1 - rho)^K; see
    `calc_denovo_step_prob`.
    """
    p = calc_load_params(**kwargs)
    sweep_time_non = calc_sweep_time(
        p["supply_non"], p["growth_non"], p["pi_non"], deme_size
    )
    freq = calc_denovo_mutator_frequency(sweep_time_non, mu, **kwargs)
    if deme_size is None:
        sweep_time = calc_sweep_time(
            p["supply_mut"], p["growth_mut"], p["pi_mut"]
        )
        return float(
            1.0
            + np.log(N * p["supply_non"] * sweep_time / (freq * np.log(2.0)))
            / np.log(p["supply_mut"] / p["supply_non"]),
        )
    # invert S(t) = deme_size ln 2 / N for t
    speed = calc_wave_speed(deme_size, p["growth_mut"])
    cover_time = np.log(deme_size * p["pi_mut"]) / p["growth_mut"] + (
        3.0
        * np.log(N / (deme_size * np.log(2.0)))
        / (np.pi * freq * p["supply_mut"] * deme_size * speed**2)
    ) ** (1.0 / 3.0)
    return float(cover_time / sweep_time_non)
