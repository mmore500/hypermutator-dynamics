import numpy as np

from ._calc_load_params import calc_load_params


def calc_hitchhike_prob_closed(N: float, x0: float = 0.5, **kwargs) -> float:
    """Probability mutators establish a beneficial mutation before the
    deleterious load deterministically removes them.

    P = 1 - (1 - x0)^(N / N_hitch) exp(-N x0 t_lag Ub' q0 pi)

    Reduces to Raynes et al. eq. 1c, (Uben / Udel) Pfix(sben), for x0 = 1/N.
    """
    p = calc_load_params(**kwargs)
    M, Ub = kwargs.get("M", 100.0), kwargs.get("Ub", 1e-6)
    rate = N * M * Ub * p["q0"] * p["pi_mut"]
    return float(
        -np.expm1(
            rate * np.log1p(-x0) / p["delta"] - rate * x0 * p["t_lag"],
        )
    )
