import numpy as np

from ._calc_establishment_prob import calc_establishment_prob


def calc_sweep_lag(
    N: float,
    frequency: float,
    s: float = 0.1,
    Ud: float = 1e-4,
    M: float = 100.0,
) -> float:
    """Generations for a mutator lineage that has just taken the lead to
    rise to the given frequency in a population of size N.

    An established lineage numbers about 1 / pi individuals and grows
    logistically at rate s - M Ud, its advantage net of deleterious load.
    Model fixation probabilities count a lineage from when it takes the
    lead, whereas experiments detect it only once it becomes common.
    Returns inf if the mutator's load cancels its advantage.
    """
    rate = s - M * Ud
    if rate <= 0:
        return np.inf
    pi = calc_establishment_prob((1.0 + s) * np.exp(-M * Ud))
    return float(np.log(frequency / (1.0 - frequency) * N * pi) / rate)
