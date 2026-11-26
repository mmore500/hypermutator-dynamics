from ._calc_denovo_mutator_frequency import calc_denovo_mutator_frequency
from ._calc_load_params import calc_load_params
from ._calc_sweep_time import calc_sweep_time
from ._calc_unswept_prob import calc_unswept_prob


def calc_denovo_peak_prevalence_closed(
    K: int,
    deme_size: int = 256,
    mu: float = 1e-6,
    **kwargs,
) -> float:
    """Peak prevalence reached by mutators arising de novo on a large 2D
    lattice of demes with K available beneficial mutations, in closed form.

    Mutators, present at typical frequency x
    (`calc_denovo_mutator_frequency`), found domains at per-capita rate
    x supply_mut. Nonmutators reach the fitness peak, halting mutators'
    advance, after K sweep times T_non, by when mutator domains have reached
    all but a fraction S (`calc_unswept_prob`) of the lattice:
        prevalence = 1 - S(K T_non; x supply_mut).
    """
    p = calc_load_params(**kwargs)
    sweep_time_non = calc_sweep_time(
        p["supply_non"], p["growth_non"], p["pi_non"], deme_size
    )
    freq = calc_denovo_mutator_frequency(sweep_time_non, mu, **kwargs)
    return 1.0 - calc_unswept_prob(
        K * sweep_time_non,
        freq * p["supply_mut"],
        p["growth_mut"],
        p["pi_mut"],
        deme_size,
    )
