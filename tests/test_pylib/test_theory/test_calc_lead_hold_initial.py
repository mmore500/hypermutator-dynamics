from pylib.theory._calc_lead_hold_initial import calc_lead_hold_initial


def test_calc_lead_hold_initial():
    args = 0.1, 1e-6, 1e-4, 100.0
    res = [calc_lead_hold_initial(1679616, k, *args) for k in (1, 2, 3, 4)]
    assert res == sorted(res)
    assert res[0] < 0.01 and res[-1] > 0.9
    assert calc_lead_hold_initial(2304, 1, *args) > 0.95
