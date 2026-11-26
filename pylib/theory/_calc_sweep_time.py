import typing

import numpy as np
from scipy import special as sp_special

from ._calc_fill_time import calc_fill_time


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
        T = t_fill + Gamma(4/3) [3 t_fill^2 / (pi supply deme_size)]^(1/3),
    with t_fill from `calc_fill_time`: a lineage fills its founding deme,
    then advances one deme per t_fill. In d dimensions the second term
    scales as (supply / t_fill^d)^(-1/(d+1)) (cf. Ralph & Coop 2010).
    """
    if deme_size is None:
        return (
            np.log(establishment * growth / supply) - np.euler_gamma
        ) / growth
    fill_time = calc_fill_time(deme_size, growth, establishment)
    return float(
        fill_time
        + sp_special.gamma(4.0 / 3.0)
        * (3.0 * fill_time**2 / (np.pi * supply * deme_size)) ** (1.0 / 3.0)
    )
