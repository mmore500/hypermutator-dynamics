import numpy as np
from scipy import optimize as sp_optimize


def calc_establishment_prob(W: float) -> float:
    """Survival probability of a branching process with Poisson(W) offspring.

    Solves 1 - p = exp(-W p); approximately 2 (W - 1) for W near 1.
    """
    if W <= 1.0 + 1e-12:
        return 0.0
    return float(
        sp_optimize.brentq(lambda p: 1.0 - p - np.exp(-W * p), 1e-12, 1.0)
    )
