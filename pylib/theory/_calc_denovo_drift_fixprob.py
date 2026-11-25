import numpy as np


def calc_denovo_drift_fixprob(
    N: float,
    Ud: float = 1e-4,
    M: float = 100.0,
    mu: float = 1e-6,
    n_gen: int = 500000,
) -> float:
    """P(some de novo mutator lineage fixes by drift alone in n_gen
    generations), per Kimura with selection -delta from a single copy."""
    delta = np.exp(-Ud) - np.exp(-M * Ud)
    if 2.0 * N * delta > 700:
        return 0.0
    p_fix = np.expm1(2.0 * delta) / np.expm1(2.0 * N * delta)
    return float(-np.expm1(-N * mu * p_fix * n_gen))
