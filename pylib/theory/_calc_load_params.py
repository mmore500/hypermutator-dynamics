import numpy as np


def calc_load_params(
    s: float = 0.1,
    Ub: float = 1e-6,
    Ud: float = 1e-4,
    M: float = 100.0,
) -> dict:
    """Derived quantities from which all results are built.

    - delta: indirect selection against mutators, their excess rate of
      deleterious mutation.
    - q0: fraction of mutators free of deleterious mutations at
      mutation-selection balance.
    - pi_non, pi_mut: probability that a beneficial mutation establishes,
      1 - exp(-2 s), the large-population limit of Kimura's fixation
      probability as used by Raynes et al. (2018). The same for both
      types: mutators' load enters their supply through q0 instead.
    - growth_non, growth_mut: per-generation growth rate of an established
      beneficial clone; mutators' is net of their load.
    - supply_non, supply_mut: rate per individual per generation at which
      nonmutators and mutators found established beneficial clones.
    - N_hitch: lower critical population size, delta / supply_mut; cf.
      N_crit of Raynes et al. (2018).
    """
    Udm, Ubm = M * Ud, M * Ub
    delta = np.exp(-Ud) - np.exp(-Udm)
    q0 = 1.0 - (1.0 - np.exp(-Udm)) / s
    pi_mut = pi_non = -np.expm1(-2.0 * s)
    supply_mut = Ubm * q0 * pi_mut
    return {
        "delta": delta,
        "q0": q0,
        "pi_mut": pi_mut,
        "pi_non": pi_non,
        "growth_non": s,
        "growth_mut": s - delta,
        "supply_non": Ub * pi_non,
        "supply_mut": supply_mut,
        "N_hitch": delta / supply_mut,
    }
