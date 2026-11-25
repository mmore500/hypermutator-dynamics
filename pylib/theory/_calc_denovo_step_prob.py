from ._calc_load_params import calc_load_params


def calc_denovo_step_prob(
    s: float = 0.1,
    Ub: float = 1e-6,
    Ud: float = 1e-4,
    M: float = 100.0,
    mu: float = 1e-6,
) -> float:
    """Closed-form small-N limit: probability that a given beneficial
    mutation is swept by a mutator rather than a nonmutator.

    Each mutator hitchhikes with probability supply_mut / delta before its
    load removes it (Raynes et al. 2018, eq. 1c), so mutators found sweeps at
    rate mu supply_mut / delta per individual, against supply_non for
    nonmutators:

        rho = mu supply_mut / (delta supply_non + mu supply_mut).

    Then Pfix = 1 - (1 - rho)^K, independent of population size.
    """
    p = calc_load_params(s=s, Ub=Ub, Ud=Ud, M=M)
    hitch = mu * p["supply_mut"] / p["delta"]
    return float(hitch / (p["supply_non"] + hitch))
