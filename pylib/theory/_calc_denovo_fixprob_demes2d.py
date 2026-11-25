import numpy as np

from ._calc_denovo_fixprob import calc_denovo_fixprob
from ._calc_denovo_kinetics_demes2d import calc_denovo_kinetics_demes2d


def calc_denovo_fixprob_demes2d(
    N: float,
    K: int,
    deme_size: int = 256,
    s: float = 0.1,
    Ub: float = 1e-6,
    Ud: float = 1e-4,
    M: float = 100.0,
    mu: float = 1e-6,
    n_gen: int = 500000,
) -> float:
    """Probability that mutators arising de novo at rate `mu` fix in a
    population of size N arranged as a square grid of demes exchanging one
    migrant per neighbor per generation, with K available beneficial
    mutations.

    Mutations establish locally and spread as waves, so the population is
    treated as a nucleation-and-growth process in space and time
    (Kolmogorov-Johnson-Mehl-Avrami kinetics; Ralph & Coop 2010; Martens &
    Hallatschek 2011), built from four ingredients.

    1. Nucleation. In each deme, nonmutators establish the next beneficial
       mutation at rate deme_size Ub pi. Mutators, held at frequency
       mu / delta by their load, establish a mutation that puts them ahead
       locally (a "lead") at rate deme_size (mu / delta) Ub' q0 pi'.
       Structure leaves establishment probabilities unchanged (Maruyama
       1970).
    2. Spread. A clone advances one deme each time a migrant establishes
       next door, so its front moves at v = r / ln(deme_size r) demes per
       generation for growth rate r. Selection is rank-based, so being
       further ahead does not make the front faster.
    3. Nonmutator adaptation. Successive nonmutator sweeps reach a location
       as a renewal process: after one class arrives, the next arrives once
       a nucleus forms within reach.
    4. The race. A mutator domain adapts M times faster than its
       surroundings and so stays ahead until it reaches the fitness peak.
       Mutators fix if their domains cover the whole grid before nonmutators
       establish at the peak anywhere still uncovered. Coverage is
       conditioned on the time of the first lead, because on small grids a
       single mutator domain decides the outcome.

    There are no free parameters. A single deme is well-mixed, and is
    delegated to `calc_denovo_fixprob`.
    """
    if N <= deme_size:
        return calc_denovo_fixprob(
            N, K, s=s, Ub=Ub, Ud=Ud, M=M, mu=mu, n_gen=n_gen
        )
    kinetics = calc_denovo_kinetics_demes2d(
        N, K, deme_size, s, Ub, Ud, M, mu, n_gen
    )
    t, rate_peak = kinetics["t"], kinetics["rate_peak"]
    reach_mut, uncovered = kinetics["reach_mut"], kinetics["uncovered"]
    rate_lead, num_demes = kinetics["rate_lead"], kinetics["num_demes"]
    num_steps, dt = len(t), t[1] - t[0]

    # (4) the race, integrating over quantiles of the first lead's timing
    p_lead = -np.expm1(-num_demes * rate_lead * t)
    num_peak = num_demes * np.cumsum(rate_peak) * dt
    num_quantiles = 200
    quantiles = (np.arange(num_quantiles) + 0.5) / num_quantiles * p_lead[-1]
    res = 0.0
    for i in np.minimum(np.searchsorted(p_lead, quantiles), num_steps - 1):
        n = num_steps - i
        num_uncovered = (num_demes - reach_mut[:n] * (reach_mut[:n] > 1)) * (
            uncovered[:n]
        )
        num_uncovered = np.where(
            reach_mut[:n] >= num_demes, 0.0, num_uncovered
        )
        num_fail = num_peak[i] + np.sum(rate_peak[i:] * num_uncovered) * dt
        res += np.exp(-num_fail) * p_lead[-1] / num_quantiles
    return float(res)
