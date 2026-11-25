import numpy as np

from ._calc_establishment_prob import calc_establishment_prob


def calc_load_params(
    s: float = 0.1,
    Ub: float = 1e-6,
    Ud: float = 1e-4,
    M: float = 100.0,
) -> dict:
    """Derived quantities shared by the closed-form and numeric results."""
    Udm, Ubm = M * Ud, M * Ub
    # indirect selection against mutator due to excess deleterious load
    delta = np.exp(-Ud) - np.exp(-Udm)
    # fraction of mutators that are unloaded at mutation-selection balance
    q0 = 1.0 - (1.0 - np.exp(-Udm)) / s
    # establishment probability of a beneficial mutation, by background
    pi_mut = calc_establishment_prob((1.0 + s) * np.exp(-Udm))
    pi_non = calc_establishment_prob((1.0 + s) * np.exp(-Ud))
    # generations before load-driven decline of mutators sets in
    t_lag = 1.0 / (s - Udm)
    return {
        "delta": delta,
        "q0": q0,
        "pi_mut": pi_mut,
        "pi_non": pi_non,
        "t_lag": t_lag,
        # lower critical population size; cf. Raynes et al. N_crit
        "N_hitch": delta / (Ubm * q0 * pi_mut),
    }
