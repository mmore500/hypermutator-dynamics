import numpy as np

from ._calc_load_params import calc_load_params


def calc_hitchhike_prob_closed(N: float, x0: float = 0.5, **kwargs) -> float:
    """Probability mutators establish a beneficial mutation before their
    deleterious load removes them.

    Loaded mutators decline logistically at rate delta, so the chance that
    none founds an established beneficial clone first is

        P = 1 - (1 - x0)^(N / N_hitch),    N_hitch = delta / supply_mut.

    Reduces to Raynes et al. (2018) eq. 1c, (Uben / Udel) Pfix(sben), for
    x0 = 1/N, and equals x0 at N = N_hitch whatever the value of x0.
    """
    p = calc_load_params(**kwargs)
    return float(-np.expm1(N / p["N_hitch"] * np.log1p(-x0)))
