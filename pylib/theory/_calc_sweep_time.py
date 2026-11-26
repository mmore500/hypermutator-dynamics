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
    fitness class: the integral over time of `calc_unswept_prob`,
        T = integral_0^inf S(t) dt.

    Well-mixed (`deme_size` None):
        T = [ln(establishment growth / supply) - gamma] / growth,
    with gamma the Euler-Mascheroni constant, for supply << growth.

    2D lattice of demes:
        T = Gamma(4/3) [3 / (pi supply deme_size v^2)]^(1/3)
            + ln(deme_size establishment) / growth,
    with wave speed v from `calc_wave_speed`. In d dimensions the first term
    scales as (supply v^d)^(-1/(d+1)) (cf. Ralph & Coop 2010).
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
