import typing

import numpy as np
from scipy import signal as sp_signal

from ._calc_disc_fraction import calc_disc_fraction
from ._calc_establishment_prob import calc_establishment_prob


def calc_denovo_kinetics_demes2d(
    N: float,
    K: int,
    deme_size: int = 256,
    s: float = 0.1,
    Ub: float = 1e-6,
    Ud: float = 1e-4,
    M: float = 100.0,
    mu: float = 1e-6,
    n_gen: int = 500000,
) -> typing.Dict[str, typing.Any]:
    """Nucleation-and-growth kinetics of nonmutator adaptation and of
    mutator takeover on a square grid of demes; see
    `calc_denovo_fixprob_demes2d` for the model.

    Returns time points `t` with, at each: `arrived`, the probability that
    nonmutators' peak class K has arrived at a location; `rate_peak`,
    nonmutator nucleations of the peak class per deme per generation;
    `reach_mut`, demes reached by a mutator domain of that age; and
    `uncovered`, the probability a location has not been reached by any
    mutator domain. Also returns scalars `rate_lead` and `num_demes`.
    """
    num_demes = N / deme_size
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

    uncovered = np.exp(-rate_lead * np.cumsum(reach_mut) * dt)
    return {
        "t": t,
        "arrived": arrived,
        "rate_peak": rate_peak,
        "reach_mut": reach_mut,
        "uncovered": uncovered,
        "rate_lead": rate_lead,
        "num_demes": num_demes,
    }
