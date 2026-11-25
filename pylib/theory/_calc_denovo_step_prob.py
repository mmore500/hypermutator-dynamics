import numpy as np

from ._calc_establishment_prob import calc_establishment_prob


def calc_denovo_step_prob(
    s: float = 0.1,
    Ub: float = 1e-6,
    Ud: float = 1e-4,
    M: float = 100.0,
    mu: float = 1e-6,
) -> float:
    """Closed-form small-N limit: probability that a given beneficial
    mutation is swept by a mutator rather than a nonmutator.

    rho = mu Phh / (Ub pi + mu Phh), with Phh the single-mutant hitchhiking
    probability (Raynes et al. eq. 1c). Then Pfix = 1 - (1 - rho)^K.
    """
    Udm = M * Ud
    delta = np.exp(-Ud) - np.exp(-Udm)
    q0 = 1.0 - (1.0 - np.exp(-Udm)) / s
    pi_mut = calc_establishment_prob((1.0 + s) * np.exp(-Udm))
    pi_non = calc_establishment_prob((1.0 + s) * np.exp(-Ud))
    t_lag = 1.0 / (s - Udm)
    p_hitch = M * Ub * q0 * pi_mut * (1.0 / delta + t_lag)
    return float(mu * p_hitch / (Ub * pi_non + mu * p_hitch))
