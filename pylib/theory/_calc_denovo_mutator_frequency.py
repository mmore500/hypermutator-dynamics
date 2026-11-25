import numpy as np

from ._calc_load_params import calc_load_params


def calc_denovo_mutator_frequency(
    sweep_interval: float, mu: float = 1e-6, **kwargs
) -> float:
    """Typical frequency of mutators arising de novo at rate `mu` in an
    adapting nonmutator population.

    Mutators approach their mutation-selection balance frequency mu / delta
    as 1 - exp(-delta t), and each nonmutator sweep purges them. Averaged
    over the interval T between sweeps,
        x = (mu / delta) * [1 - (1 - exp(-delta T)) / (delta T)].

    De novo results follow from treating mutators as present at this
    frequency; see `calc_denovo_critical_potential_closed`.
    """
    delta = calc_load_params(**kwargs)["delta"]
    turnover = delta * sweep_interval
    return float(mu / delta * (1.0 - -np.expm1(-turnover) / turnover))
