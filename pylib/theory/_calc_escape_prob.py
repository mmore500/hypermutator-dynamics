import numpy as np
from scipy import linalg as sp_linalg

from ._calc_load_params import calc_load_params


def calc_escape_prob(
    N: float,
    x0: float = 0.5,
    s: float = 0.1,
    Ub: float = 1e-6,
    Ud: float = 1e-4,
    M: float = 100.0,
) -> float:
    """P(mutators escape their deleterious load), treating drift and
    hitchhiking jointly rather than additively.

    Mutator frequency x is a Wright-Fisher diffusion with selection -delta,
    "killed" in favor of mutators at rate a x, when a mutator establishes a
    beneficial mutation. The win probability u(x) solves

        x(1-x)/(2N) u'' - delta x(1-x) u' + a x (1-u) = 0,

    with u(0) = 0 and u(1) = 1.
    """
    p = calc_load_params(s=s, Ub=Ub, Ud=Ud, M=M)
    delta = p["delta"]
    a = N * M * Ub * p["q0"] * p["pi_mut"]
    if -a * np.log1p(-x0) / delta > 40.0:  # deterministic limit, certain
        return 1.0
    n = int(min(max(2000, 40 * N * delta), 400000))
    x = np.linspace(0.0, 1.0, n + 1)[1:-1]
    h = 1.0 / n
    diffusion = x * (1.0 - x) / (2.0 * N)
    drift = delta * x * (1.0 - x)  # toward x = 0; upwinded
    lower = diffusion / h**2 + drift / h
    upper = diffusion / h**2
    diag = -(lower + upper) - a * x
    rhs = -a * x
    rhs[-1] -= upper[-1]  # boundary condition u(1) = 1
    banded = np.zeros((3, n - 1))
    banded[0, 1:] = upper[:-1]
    banded[1] = diag
    banded[2, :-1] = lower[1:]
    u = sp_linalg.solve_banded((1, 1), banded, rhs)
    return float(np.interp(x0, x, u))
