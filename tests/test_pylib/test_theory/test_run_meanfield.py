import numpy as np

from pylib.theory._run_meanfield import run_meanfield


def test_run_meanfield_nonmutators_climb_to_peak():
    K = 2
    f = np.zeros((2, K + 1, 5))
    f[0, 0, 0] = 1.0
    W, nose, z, states = run_meanfield(
        f,
        1e6,
        K,
        0.1,
        1e-6,
        1e-4,
        100.0,
        False,
        lambda f, top, Ws: top[0] == K and f[0, K, 0] > 0.99,
        max_gen=10000,
        record_states=True,
    )
    assert len(W) == len(nose) == len(z) == len(states) < 10000
    assert W.shape[1] == z.shape[1] == K + 1
    assert nose[0] <= 1 and nose[-1] == K
    assert (np.diff(nose) >= 0).all()
    assert states[0][0, 0, 0] == 1.0


def test_run_meanfield_block_peak():
    K = 2
    f = np.zeros((2, K + 1, 5))
    f[0, 0, 0] = 1.0
    __, nose, __, __ = run_meanfield(
        f, 1e6, K, 0.1, 1e-6, 1e-4, 100.0, True, lambda *_: False, 2000
    )
    assert nose.max() == K - 1
