import numpy as np


def calc_fill_time(
    deme_size: int, growth: float, establishment: float
) -> float:
    """Generations for an established lineage to take over one deme.

    An established lineage holds exp(growth t) / establishment individuals
    after t generations, so it fills a deme of `deme_size` after
        t_fill = ln(deme_size establishment) / growth.

    On a lattice of demes exchanging about one migrant per neighbor per
    generation, this is also taken as the time a lineage needs to advance
    one further deme: a lineage waits t_fill to take over its founding deme
    and then spreads at 1 / t_fill demes per generation. The time for a
    migrant to establish next door, ln(deme_size growth / migrants) /
    growth, is similar at that migration rate; with much weaker or stronger
    migration the advance per deme would need that expression instead.
    """
    return float(np.log(deme_size * establishment) / growth)
