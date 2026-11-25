import numpy as np

from ._calc_denovo_kinetics_demes2d import calc_denovo_kinetics_demes2d


def calc_denovo_peak_prevalence_demes2d(
    N: float,
    K: int,
    deme_size: int = 256,
    **kwargs,
) -> float:
    """Expected peak prevalence reached by mutators arising de novo on a
    square grid of demes, with K available beneficial mutations.

    A location is taken by mutators if a mutator domain reaches it before
    nonmutators' peak fitness class does; thereafter nonmutators at the peak
    slowly reclaim ground. Peak mutator prevalence is thus the probability
    that mutators arrive first, per `calc_denovo_kinetics_demes2d`.
    """
    kinetics = calc_denovo_kinetics_demes2d(N, K, deme_size, **kwargs)
    return float(
        np.sum(
            np.gradient(kinetics["arrived"]) * (1.0 - kinetics["uncovered"]),
        ),
    )
