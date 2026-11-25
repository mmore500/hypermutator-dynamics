import numpy as np

from pylib.theory._calc_lead_hold_table import calc_lead_hold_table
from pylib.theory._denovo_constants import HOLD_KMAX, HOLD_TAUS


def test_calc_lead_hold_table():
    res = calc_lead_hold_table(1679616)
    assert res.shape == (HOLD_KMAX + 1, len(HOLD_TAUS))
    assert ((0.0 <= res) & (res <= 1.0)).all()
    # holding is easier with more adaptive potential remaining ...
    assert (np.diff(res[1:], axis=0) >= -1e-9).all()
    # ... and when nonmutators are caught earlier in their own sweep
    assert (np.diff(res[1:], axis=1) <= 1e-9).all()
    assert res[HOLD_KMAX].min() > 0.99
