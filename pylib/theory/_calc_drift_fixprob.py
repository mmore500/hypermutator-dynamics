import numpy as np

from ._calc_load_params import calc_load_params


def calc_drift_fixprob(N: float, x0: float = 0.5, **kwargs) -> float:
    """Kimura fixation probability of loaded mutators absent beneficials.

    Generalizes Raynes et al. eq. 1b from x0 = 1/N to arbitrary x0.
    """
    delta = calc_load_params(**kwargs)["delta"]
    a = 2.0 * N * delta
    # (e^{a x0} - 1) / (e^a - 1), arranged to avoid overflow at large a
    return float(np.exp(a * (x0 - 1.0)) * np.expm1(-a * x0) / np.expm1(-a))
