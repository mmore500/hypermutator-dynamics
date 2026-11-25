from scipy import optimize as sp_optimize

from ._calc_num_sweeps import calc_num_sweeps


def calc_selection_from_num_sweeps(
    N: float,
    n_gen: int,
    num_sweeps: float,
    Ub: float = 1e-6,
    Ud: float = 1e-4,
) -> float:
    """Selection strength s at which nonmutators in a population of size N
    complete `num_sweeps` selective sweeps in n_gen generations.

    Calibrates the model's clock to an evolution experiment from the number
    of mutations observed to fix in populations that stayed nonmutator.
    """
    return float(
        sp_optimize.brentq(
            lambda s: calc_num_sweeps(N, n_gen, s, Ub, Ud) - num_sweeps,
            0.005,
            0.5,
            xtol=1e-4,
        ),
    )
