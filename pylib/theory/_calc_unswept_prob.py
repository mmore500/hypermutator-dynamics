import typing

import numpy as np

from ._calc_wave_speed import calc_wave_speed


def calc_unswept_prob(
    t: float,
    supply: float,
    growth: float,
    establishment: float,
    deme_size: typing.Optional[int] = None,
) -> float:
    """Probability that a given location has not been reached, by generation
    t, by any clone of a fitness class whose clones are founded at rate
    `supply` per individual per generation, each establishing with
    probability `establishment` and then growing at per-generation rate
    `growth`.

    Founding events are Poisson in space and time, and a location stays
    unswept so long as no clone founded earlier has reached it (Kolmogorov
    1937; Johnson & Mehl 1939; Avrami 1939):
        S(t) = exp[-supply * integral_0^t V(tau) dtau],
    where V(tau) is the number of individuals a clone of age tau has
    reached. Population structure enters only through V.

    Well-mixed (`deme_size` None): V(tau) = exp(growth tau) / establishment.

    2D lattice of demes: a clone takes tau0 = ln(deme_size establishment) /
    growth generations to take over its founding deme, then spreads as a
    disc at speed v from `calc_wave_speed`, so
    V(tau) = deme_size pi v^2 (tau - tau0)^2 for tau > tau0. In d dimensions
    V grows as (v tau)^d (cf. Ralph & Coop 2010).
    """
    if deme_size is None:
        reach = np.expm1(growth * t) / (growth * establishment)
    else:
        speed = calc_wave_speed(deme_size, growth)
        spread = max(t - np.log(deme_size * establishment) / growth, 0.0)
        reach = deme_size * np.pi * speed**2 * spread**3 / 3.0
    return float(np.exp(-supply * reach))
