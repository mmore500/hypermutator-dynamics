import numpy as np

from ._calc_load_params import calc_load_params
from ._calc_sweep_time import calc_sweep_time
from ._calc_wave_speed import calc_wave_speed


def calc_denovo_peak_prevalence_closed(
    K: int,
    deme_size: int = 256,
    mu: float = 1e-6,
    **kwargs,
) -> float:
    """Peak prevalence reached by mutators arising de novo on a large 2D
    lattice of demes with K available beneficial mutations, in closed form.

    Mutator domains are founded at rate J per deme per generation and spread
    at speed v, so by generation t they have reached a fraction
    1 - exp(-pi J v^2 t^3 / 3) of the lattice. Nonmutators reach the fitness
    peak, halting mutators' advance, after K sweep times T_non:

        prevalence = 1 - exp[-pi J v^2 (K T_non)^3 / 3].

    See `calc_denovo_critical_potential_closed` for J.
    """
    p = calc_load_params(**kwargs)
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
    return float(
        -np.expm1(
            -np.pi * rate_lead * speed**2 * (K * sweep_time_non) ** 3 / 3.0,
        ),
    )
