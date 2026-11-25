import typing

import numpy as np
from scipy import special as sp_special

from ._calc_wave_speed import calc_wave_speed


def calc_sweep_time(
    supply: float,
    growth: float,
    establishment: float,
    deme_size: typing.Optional[int] = None,
) -> float:
    """Mean generations until a given location is reached by a sweeping
    fitness class, when clones of that class are founded at rate `supply`
    per individual per generation, each establishing with probability
    `establishment` and then growing at per-generation rate `growth`.

    Founding events are Poisson in space and time, and a location stays
    unswept so long as no clone founded earlier has reached it (Kolmogorov
    1937; Johnson & Mehl 1939; Avrami 1939). Population structure enters
    only through how fast a clone's reach grows.

    Well-mixed (`deme_size` None): an established clone holds fraction
    exp(growth t) / (establishment N) of the population, giving

        T = [ln(establishment growth / supply) - gamma] / growth,

    with gamma the Euler-Mascheroni constant.

    2D lattice of demes: a clone takes ln(deme_size establishment) / growth
    generations to take over its founding deme, then spreads as a disc at
    speed v from `calc_wave_speed`, giving

        T = Gamma(4/3) [3 / (pi supply deme_size v^2)]^(1/3)
            + ln(deme_size establishment) / growth.

    In d dimensions the first term scales as (supply v^d)^(-1/(d+1)) (cf.
    Ralph & Coop 2010).
    """
    if deme_size is None:
        return (
            np.log(establishment * growth / supply) - np.euler_gamma
        ) / growth
    speed = calc_wave_speed(deme_size, growth)
    return float(
        sp_special.gamma(4.0 / 3.0)
        * (3.0 / (np.pi * supply * deme_size * speed**2)) ** (1.0 / 3.0)
        + np.log(deme_size * establishment) / growth
    )
