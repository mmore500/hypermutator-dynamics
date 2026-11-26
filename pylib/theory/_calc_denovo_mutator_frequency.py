from ._calc_load_params import calc_load_params


def calc_denovo_mutator_frequency(
    sweep_interval: float, mu: float = 1e-6, **kwargs
) -> float:
    """Typical frequency of mutators arising de novo at rate `mu` in an
    adapting nonmutator population.

    Mutators are supplied at rate mu and lost in two ways: to their own
    deleterious load, at rate delta, and to nonmutator sweeps, which pass
    about once every `sweep_interval` generations T. Supply over loss gives
        x = mu / (delta + 1 / T),
    the mutation-selection balance mu / delta when sweeps are rare and
    mu T when they are frequent. Treats sweeps as arriving at random.

    De novo results follow from treating mutators as present at this
    frequency; see `calc_denovo_critical_potential_closed`.
    """
    delta = calc_load_params(**kwargs)["delta"]
    return float(mu / (delta + 1.0 / sweep_interval))
