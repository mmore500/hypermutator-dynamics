import numpy as np
from scipy import signal as sp_signal

from ._calc_denovo_fixprob import calc_denovo_fixprob
from ._calc_disc_fraction import calc_disc_fraction
from ._calc_establishment_prob import calc_establishment_prob


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
    num_demes = N / deme_size
    if num_demes <= 1:
        return calc_denovo_fixprob(
            N, K, s=s, Ub=Ub, Ud=Ud, M=M, mu=mu, n_gen=n_gen
        )
    grid_len = np.sqrt(num_demes)
    delta = np.exp(-Ud) - np.exp(-M * Ud)
    q0 = 1.0 - (1.0 - np.exp(-M * Ud)) / s
    pi_non = calc_establishment_prob((1.0 + s) * np.exp(-Ud))
    pi_mut = calc_establishment_prob((1.0 + s) * np.exp(-M * Ud))
    r_non, r_mut = s, s - delta

    # (1) nucleation rates, per deme per generation
    rate_non = deme_size * Ub * pi_non
    t_lag = 1.0 / (s - M * Ud)  # as in calc_hitchhike_prob_closed
    rate_lead = deme_size * mu * (1.0 / delta + t_lag) * M * Ub * q0 * pi_mut

    # (2) spread: demes reached by a clone founded tau generations ago
    def calc_reach(tau: np.ndarray, r: float, pi: float) -> np.ndarray:
        speed = r / np.log(deme_size * r)
        t_local = np.log(deme_size * pi) / r  # to take over founding deme
        fraction = calc_disc_fraction(
            speed * np.clip(tau - t_local, 0.0, None) / grid_len,
        )
        return np.where(
            tau >= t_local, np.maximum(num_demes * fraction, 1.0), 0.0
        )

    speed_non = r_non / np.log(deme_size * r_non)
    t_max = min(
        float(n_gen),
        1.5
        * (K + 4)
        * (
            1.0 / (num_demes * rate_non)
            + (3.0 / (np.pi * rate_non * speed_non**2)) ** (1.0 / 3.0)
            + grid_len / speed_non
        ),
    )
    num_steps = int(np.clip(t_max / 2.0, 4000, 150000))
    t = np.linspace(0.0, t_max, num_steps)
    dt = t[1] - t[0]
    reach_non = calc_reach(t, r_non, pi_non)
    reach_mut = calc_reach(t, r_mut, pi_mut)

    # (3) nonmutator adaptation: P(class k has arrived at a location by t)
    next_arrival = -np.expm1(-rate_non * np.cumsum(reach_non) * dt)
    period = np.sum(1.0 - next_arrival) * dt
    arrived_prev, arrived = np.ones(num_steps), next_arrival
    for __ in range(K - 1):
        arrived_prev = arrived
        arrived = np.clip(
            sp_signal.fftconvolve(np.gradient(arrived_prev, dt), next_arrival)[
                :num_steps
            ]
            * dt,
            0.0,
            1.0,
        )
    # nonmutator nucleations of the peak class, per deme per generation
    rate_peak = rate_non * arrived_prev

    # each nonmutator sweep resets local mutators, which then rebuild toward
    # mu / delta; average their abundance over the time between sweeps
    rate_lead *= 1.0 - (1.0 - np.exp(-delta * period)) / (delta * period)

    # (4) the race, integrating over quantiles of the first lead's timing
    p_lead = -np.expm1(-num_demes * rate_lead * t)
    num_peak = num_demes * np.cumsum(rate_peak) * dt
    num_lead_within_reach = rate_lead * np.cumsum(reach_mut) * dt
    num_quantiles = 200
    quantiles = (np.arange(num_quantiles) + 0.5) / num_quantiles * p_lead[-1]
    res = 0.0
    for i in np.minimum(np.searchsorted(p_lead, quantiles), num_steps - 1):
        n = num_steps - i
        num_uncovered = (num_demes - reach_mut[:n] * (reach_mut[:n] > 1)) * (
            np.exp(-num_lead_within_reach[:n])
        )
        num_uncovered = np.where(
            reach_mut[:n] >= num_demes, 0.0, num_uncovered
        )
        num_fail = num_peak[i] + np.sum(rate_peak[i:] * num_uncovered) * dt
        res += np.exp(-num_fail) * p_lead[-1] / num_quantiles
    return float(res)
