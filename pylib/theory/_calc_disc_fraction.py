import functools
import typing

import numpy as np


@functools.lru_cache(maxsize=None)
def _tabulate_disc_fraction() -> typing.Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    a, b = rng.random((2000, 2)), rng.random((2000, 2))
    distances = np.sort(
        np.hypot(*(a[:, None, :] - b[None, :, :]).transpose(2, 0, 1)).ravel(),
    )
    radii = np.linspace(0.0, np.sqrt(2.0), 600)
    return radii, np.searchsorted(distances, radii) / distances.size


def calc_disc_fraction(radius: np.ndarray) -> np.ndarray:
    """Mean fraction of a unit square lying within `radius` of a uniformly
    random point in that square.

    Describes how much of a square grid a clone spreading as a circular wave
    from a random origin has reached, accounting for grid edges. Equals
    pi radius^2 for small radii and 1 beyond the square's diagonal.
    """
    radii, fractions = _tabulate_disc_fraction()
    return np.interp(radius, radii, fractions, right=1.0)
