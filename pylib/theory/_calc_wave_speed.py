import numpy as np


def calc_wave_speed(deme_size: int, growth: float) -> float:
    """Speed, in demes per generation, at which a clone with per-generation
    growth rate `growth` spreads across a lattice of demes that exchange one
    migrant per neighbor per generation.

    A deme is invaded once a migrant from its neighbor establishes there.
    A clone growing as exp(growth t) / pi within the neighbor, sending
    migrants in proportion to its frequency, each establishing with
    probability pi, is expected to have done so after

        ln(deme_size growth) / growth

    generations, the time to advance one deme. Assumes strong selection
    within demes, deme_size growth >> 1.
    """
    return growth / np.log(deme_size * growth)
