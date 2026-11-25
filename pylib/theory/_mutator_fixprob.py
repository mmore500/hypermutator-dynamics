"""Semi-analytic fixation probability of a mutator allele competed against a
nonmutator from initial frequency x0, with finite adaptive potential.

Extends Raynes et al. (2018) eq. 1 to (a) arbitrary initial mutator frequency
and (b) a cap K on the number of available beneficial mutations, under the
tournament-selection model used in this repository's simulations.

Model recap (well-mixed): each generation every genome mutates (Poisson
beneficial and deleterious counts, mutators at M-fold rates; beneficial count
capped at K) then each of N slots is filled by a size-2 tournament with
probability s, else by a uniformly random parent. An individual's expected
offspring number is therefore

    W = 1 + s * (F_below - F_above),

with F_below/F_above the population fractions with strictly lower/higher
score. Selection is rank-based: any score advantage is worth s.
"""

import typing

import numpy as np
from scipy import linalg as sp_linalg
from scipy import optimize as sp_optimize


def calc_establishment_prob(W: float) -> float:
    """Survival probability of a branching process with Poisson(W) offspring.

    Solves 1 - p = exp(-W p); approximately 2 (W - 1) for W near 1.
    """
    if W <= 1.0 + 1e-12:
        return 0.0
    return float(
        sp_optimize.brentq(lambda p: 1.0 - p - np.exp(-W * p), 1e-12, 1.0)
    )


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


def calc_drift_fixprob(N: float, x0: float = 0.5, **kwargs) -> float:
    """Kimura fixation probability of loaded mutators absent beneficials.

    Generalizes Raynes et al. eq. 1b from x0 = 1/N to arbitrary x0.
    """
    delta = calc_load_params(**kwargs)["delta"]
    a = 2.0 * N * delta
    if a * (1.0 - x0) > 700:
        return 0.0
    return float(np.expm1(a * x0) / np.expm1(a))


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


def calc_escape_prob_closed(N: float, x0: float = 0.5, **kwargs) -> float:
    """Closed-form analog of Raynes et al. eq. 1a for a start at x0."""
    p_drift = calc_drift_fixprob(N, x0, **kwargs)
    p_hitch = calc_hitchhike_prob_closed(N, x0, **kwargs)
    return p_drift + (1.0 - p_drift) * p_hitch


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
    "killed" in favor of mutators at rate a x (a mutator establishes a
    beneficial mutation) and in favor of nonmutators at rate b(x) (1 - x) (a
    nonmutator establishes a beneficial mutation and mutators fail to answer
    within one selective timescale). The win probability u(x) solves

        x(1-x)/(2N) u'' - delta x(1-x) u' + a x (1-u) - b(x)(1-x) u = 0,

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
    b = N * Ub * p["pi_non"] * np.exp(-a * x / s)
    diffusion = x * (1.0 - x) / (2.0 * N)
    drift = delta * x * (1.0 - x)  # toward x = 0; upwinded
    lower = diffusion / h**2 + drift / h
    upper = diffusion / h**2
    diag = -(lower + upper) - (a * x + b * (1.0 - x))
    rhs = -a * x
    rhs[-1] -= upper[-1]  # boundary condition u(1) = 1
    banded = np.zeros((3, n - 1))
    banded[0, 1:] = upper[:-1]
    banded[1] = diag
    banded[2, :-1] = lower[1:]
    u = sp_linalg.solve_banded((1, 1), banded, rhs)
    u_x0 = float(np.interp(x0, x, u))
    # load takes ~t_lag generations to build, extending mutators' window
    return 1.0 - (1.0 - u_x0) * float(np.exp(-a * x0 * p["t_lag"]))


def calc_mutator_wave(
    K: int,
    N: float = np.inf,
    x0: float = 0.5,
    s: float = 0.1,
    Ub: float = 1e-6,
    Ud: float = 1e-4,
    M: float = 100.0,
    max_del: int = 6,
    tol: float = 1e-13,
) -> np.ndarray:
    """Mean-field adaptation of mutators toward the fitness peak K, in the
    presence of nonmutators that (counterfactually) never gain a beneficial
    mutation.

    Returns array W[t, j], the tournament growth factor experienced in
    generation t by a genome with j beneficial and no deleterious mutations.

    For finite N, successive mutator fitness classes are seeded
    semi-deterministically: class j opens, with 1 / pi individuals, once the
    cumulative expected number of established founders reaches one. The clock
    starts at establishment of the first mutator beneficial mutation.
    """
    pbm, pdm = -np.expm1(-M * Ub), -np.expm1(-M * Ud)
    pdn = -np.expm1(-Ud)
    f = np.zeros((2, K + 1, max_del + 1))  # [is mutator, n ben, n del]
    f[0, 0, 0], f[1, 0, 0] = 1.0 - x0, x0
    score = (
        np.arange(K + 1)[:, None] - np.arange(max_del + 1)[None, :] + max_del
    )
    n_score = K + max_del + 1
    n_established = np.zeros(K + 1)
    is_open = np.full(K + 1, not np.isfinite(N))
    is_open[0] = True
    if np.isfinite(N) and K >= 1:
        is_open[1] = True
        seed = 1.0 / (N * calc_establishment_prob((1.0 + s) * (1.0 - pdm)))
        seed = min(seed, x0)
        f[1, 0, 0] -= seed
        f[1, 1, 0] += seed

    W_score = np.ones(n_score)
    res = []
    while len(res) < 200000:
        mut = f[1]
        for b in reversed(range(K)):
            flux = mut[b] * pbm
            if is_open[b + 1]:
                mut[b] -= flux
                mut[b + 1] += flux
                continue
            pi = calc_establishment_prob(
                W_score[b + 1 + max_del] * (1.0 - pdm)
            )
            n_established[b + 1] += N * flux[0] * pi
            if n_established[b + 1] >= 1.0:
                is_open[b + 1] = True
                seed = min(1.0 / (N * pi), mut[b, 0])
                mut[b, 0] -= seed
                mut[b + 1, 0] += seed
        for genomes, p_del in (f[0], pdn), (f[1], pdm):
            for d in reversed(range(max_del)):
                flux = genomes[:, d] * p_del
                genomes[:, d] -= flux
                genomes[:, d + 1] += flux

        totals = np.bincount(
            score.ravel(), weights=f.sum(axis=0).ravel(), minlength=n_score
        )
        below = np.concatenate([[0.0], np.cumsum(totals)[:-1]])
        W_score = 1.0 + s * (2.0 * below + totals - 1.0)
        res.append(W_score[max_del:].copy())
        f *= W_score[score][None]
        f /= f.sum()

        if (
            len(res) > 50
            and f[0].sum() < tol
            and np.abs(res[-1] - res[-2]).max() < 1e-12
        ):
            break

    return np.array(res)


def calc_hold_prob(
    N: float,
    K: int,
    x0: float = 0.5,
    s: float = 0.1,
    Ub: float = 1e-6,
    Ud: float = 1e-4,
    M: float = 100.0,
    finite_wave: bool = True,
) -> typing.Tuple[float, float]:
    """P(no nonmutator lineage establishes at the fitness peak K before
    nonmutators are swept out by adapting mutators).

    Nonmutators carrying beneficial mutations are rare until they win, so
    they are treated as a multitype branching process riding the mutators'
    mean-field wave (`calc_mutator_wave`). With Q_j(t) the probability that a
    nonmutator with j beneficial mutations at generation t leaves no lineage
    established at the peak,

        Q_j(t) = (1 - ub - ud) G_j(t) + ub G_{j+1}(t) + ud,
        G_j(t) = exp(-W_j(t) (1 - Q_j(t + 1))),

    solved backward from the post-sweep stationary state. Then

        P_hold = Q_0(0)^(N (1 - x0)) = exp(-N Ub C_K).

    Returns (P_hold, C_K).
    """
    W = calc_mutator_wave(
        K, N if finite_wave else np.inf, x0=x0, s=s, Ub=Ub, Ud=Ud, M=M
    )
    pb, pd = -np.expm1(-Ub), -np.expm1(-Ud)

    def step(Q: np.ndarray, W_t: np.ndarray) -> np.ndarray:
        G = np.exp(-W_t * (1.0 - Q))
        res = (1.0 - pb - pd) * G + pd
        res[:-1] += pb * G[1:]
        res[-1] += pb * G[-1]  # capped: beneficial mutation has no effect
        return res

    Q = np.ones(K + 1)
    Q[K] = 1.0 - calc_establishment_prob(W[-1, K] * (1.0 - pd))
    for __ in range(5000):  # stationary post-sweep environment
        Q_ = step(Q, W[-1])
        if np.abs(Q_ - Q).max() < 1e-16:
            break
        Q = Q_
    for W_t in W[::-1]:
        Q = step(Q, W_t)

    p_success = 1.0 - Q[0]  # per founding nonmutator
    log_p_hold = N * (1.0 - x0) * np.log1p(-p_success)
    return float(np.exp(log_p_hold)), float((1.0 - x0) * p_success / Ub)


def calc_mutator_fixprob(
    N: float,
    K: float,
    x0: float = 0.5,
    **kwargs,
) -> float:
    """Fixation probability of mutators started at frequency x0 in a
    well-mixed population of size N with K available beneficial mutations.

    P_fix = P_escape(N) * P_hold(N, K); normalized N P_fix is P_fix / x0.
    """
    p_escape = calc_escape_prob(N, x0, **kwargs)
    if not np.isfinite(K):
        return p_escape
    return p_escape * calc_hold_prob(N, int(K), x0, **kwargs)[0]


def calc_ncrit(
    K: float,
    which: typing.Literal["lower", "upper"],
    x0: float = 0.5,
    **kwargs,
) -> float:
    """Population size where mutator fixation probability crosses its neutral
    expectation x0 (i.e., normalized N P_fix = 1).

    "lower": mutators become favored (sign inversion of Raynes et al.);
    "upper": mutators become disfavored again, for want of adaptive potential.
    Returns nan if there is no such crossing.
    """

    def objective(log_N: float) -> float:
        return calc_mutator_fixprob(10**log_N, K, x0, **kwargs) - x0

    log_Ns = np.arange(1.5, 14.01, 0.25)
    vals = np.array([objective(log_N) for log_N in log_Ns])
    if which == "lower":
        (candidates,) = np.nonzero((vals[:-1] < 0) & (vals[1:] >= 0))
    else:
        (candidates,) = np.nonzero((vals[:-1] >= 0) & (vals[1:] < 0))
    if not len(candidates):
        return np.nan
    i = candidates[0] if which == "lower" else candidates[-1]
    return float(10 ** sp_optimize.brentq(objective, log_Ns[i], log_Ns[i + 1]))
