import numpy as np

from ._denovo_constants import MAX_DEL
from ._run_meanfield import run_meanfield


def calc_num_sweeps(
    N: float,
    n_gen: int,
    s: float = 0.1,
    Ub: float = 1e-6,
    Ud: float = 1e-4,
    max_ben: int = 60,
) -> float:
    """Mean number of beneficial mutations carried by nonmutators after
    n_gen generations of adaptation in a population of size N, i.e., the
    number of selective sweeps completed.

    Observed counts of mutations fixed over an evolution experiment can be
    compared against this quantity to calibrate the model's clock; see
    `calc_selection_from_num_sweeps`.
    """
    f = np.zeros((2, max_ben + 1, MAX_DEL + 1))
    f[0, 0, 0] = 1.0
    __, __, z, __ = run_meanfield(
        f, N, max_ben, s, Ub, Ud, 1.0, False, lambda *args: False, n_gen
    )
    return float(np.dot(z[-1], np.arange(max_ben + 1)) / z[-1].sum())
