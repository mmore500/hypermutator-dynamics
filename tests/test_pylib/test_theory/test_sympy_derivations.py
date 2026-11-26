"""Closed forms derived symbolically, from the model's defining differential
equations and integrals, agree with their Python implementations."""

import functools
import itertools as it

import numpy as np
import pytest
from scipy import integrate as sp_integrate
import sympy as sp

from pylib.theory._calc_catchup_prob_closed import calc_catchup_prob_closed
from pylib.theory._calc_drift_fixprob import calc_drift_fixprob
from pylib.theory._calc_escape_prob_closed import calc_escape_prob_closed
from pylib.theory._calc_fill_time import calc_fill_time
from pylib.theory._calc_hitchhike_prob_closed import calc_hitchhike_prob_closed
from pylib.theory._calc_load_params import calc_load_params
from pylib.theory._calc_mutator_fixprob_closed import (
    calc_mutator_fixprob_closed,
)
from pylib.theory._calc_ncrit_closed import calc_ncrit_closed
from pylib.theory._calc_sweep_time import calc_sweep_time
from pylib.theory._calc_unswept_prob import calc_unswept_prob

t, tau, x = sp.symbols("t tau x", positive=True)
N, K, M, n = sp.symbols("N K M n", positive=True)
s, lam_b, lam_d = sp.symbols("s lambda_b lambda_d", positive=True)
delta, r, p, u, u_n, u_m = sp.symbols("delta r p u u_n u_m", positive=True)
x_0, T = sp.symbols("x_0 T", positive=True)

param_sets = (
    {},
    {"s": 0.05, "Ub": 2e-6, "Ud": 3e-4, "M": 30.0},
    {"s": 0.2, "Ub": 1e-7, "Ud": 5e-5, "M": 200.0},
)


def _raw(kwargs: dict) -> dict:
    """Numeric values of the raw parameters."""
    full = {"s": 0.1, "Ub": 1e-6, "Ud": 1e-4, "M": 100.0, **kwargs}
    return {s: full["s"], lam_b: full["Ub"], lam_d: full["Ud"], M: full["M"]}


@functools.lru_cache(maxsize=None)
def _derive() -> dict:
    """Derive the model's closed forms from their defining equations."""
    res = {}

    # ingredients
    q = sp.symbols("q", positive=True)
    (q0,) = [
        sol
        for sol in sp.solve(s * q * (1 - q) - (1 - sp.exp(-M * lam_d)) * q, q)
        if sol != 0
    ]
    sigma = sp.symbols("sigma", real=True, nonzero=True)
    fix = sp.Function("U")
    kimura = sp.dsolve(
        sp.Eq(fix(x).diff(x, 2) / (2 * N) + sigma * fix(x).diff(x), 0),
        fix(x),
        ics={fix(0): 0, fix(1): 1},
    ).rhs
    res["delta"] = sp.exp(-lam_d) - sp.exp(-M * lam_d)
    res["q0"] = q0
    res["p"] = sp.limit(kimura.subs({sigma: s, x: 1 / N}), N, sp.oo)
    res["ingredients"] = {
        delta: res["delta"],
        p: res["p"],
        r: s - res["delta"],
        u_n: lam_b * res["p"],
        u_m: M * lam_b * q0 * res["p"],
    }

    # escape
    res["drift"] = kimura.subs({sigma: -delta, x: x_0})
    freq = sp.Function("x")
    decline = sp.dsolve(
        sp.Eq(freq(t).diff(t), -delta * freq(t) * (1 - freq(t))),
        freq(t),
        ics={freq(0): x_0},
    ).rhs
    exposure = sp.integrate(decline, (t, 0, sp.oo), conds="none")
    res["hitch"] = 1 - sp.exp(-N * u_m * exposure)
    res["escape"] = res["drift"] + (1 - res["drift"]) * res["hitch"]

    # sweeps: S = exp(-u int V), T = int S
    res["unswept_mixed"] = sp.exp(
        -u * sp.integrate(sp.exp(r * tau) / p, (tau, 0, t))
    )
    a = sp.symbols("a", positive=True)
    exact = sp.integrate(
        sp.exp(-a * (sp.exp(r * t) - 1)), (t, 0, sp.oo), conds="none"
    )
    res["sweep_mixed_exact"] = exact.subs(a, u / (p * r))
    res["sweep_mixed"] = (
        sp.series(r * exact, a, 0, 1).removeO().subs(a, u / (p * r)) / r
    )

    (fill,) = sp.solve(sp.Eq(sp.exp(r * t) / p, n), t)
    res["fill"] = fill
    reach = sp.integrate(
        n * sp.pi * (tau / fill - 1) ** 2, (tau, fill, t)
    )  # for t > fill
    res["unswept_lattice"] = sp.exp(-u * reach)
    c = sp.symbols("c", positive=True)
    tail = sp.integrate(sp.exp(-c * x**3), (x, 0, sp.oo))
    res["sweep_lattice"] = fill + tail.subs(c, sp.pi * u * n / (3 * fill**2))

    # catch-up
    odds = (1 - x_0) / x_0 * res["hitch"] * (u_n / u_m) ** (K - 1)
    res["catchup"] = 1 - sp.exp(-N * u_n * odds / (1 + odds) * T)
    res["fixprob"] = res["escape"] * (1 - res["catchup"])
    share = sp.symbols("a_peak", positive=True)
    (res["ncrit_upper"],) = sp.solve(
        sp.Eq(sp.exp(-N * u_n * share * T), x_0), N
    )
    res["share"] = share
    return res


def _lambdify(expr, kwargs: dict, lattice: bool, args=(N, K, x_0, n)):
    """Compile an expression in terms of raw parameters only."""
    res = _derive()
    sweep = res["sweep_lattice"] if lattice else res["sweep_mixed"]
    expr = expr.subs(T, sweep.subs(u, u_m))
    expr = expr.subs(res["ingredients"]).subs(_raw(kwargs))
    return sp.lambdify(args, expr, "mpmath")


@pytest.mark.parametrize("kwargs", param_sets)
def test_ingredients(kwargs: dict):
    res, params = _derive(), calc_load_params(**kwargs)
    raw = _raw(kwargs)
    for key, sym in (
        ("delta", delta),
        ("pi_mut", p),
        ("pi_non", p),
        ("growth_mut", r),
        ("supply_non", u_n),
        ("supply_mut", u_m),
    ):
        assert np.isclose(
            float(res["ingredients"][sym].subs(raw)), params[key], rtol=1e-9
        )
    assert np.isclose(float(res["q0"].subs(raw)), params["q0"], rtol=1e-9)


@pytest.mark.parametrize("kwargs", param_sets)
@pytest.mark.parametrize("x0", [0.5, 0.1, 0.01])
def test_escape(kwargs: dict, x0: float):
    res = _derive()
    drift = _lambdify(res["drift"], kwargs, False)
    hitch = _lambdify(res["hitch"], kwargs, False)
    escape = _lambdify(res["escape"], kwargs, False)
    for N_ in np.logspace(1, 9, 33):
        at = (N_, 1, x0, 1)
        for func, impl in (
            (drift, calc_drift_fixprob),
            (hitch, calc_hitchhike_prob_closed),
            (escape, calc_escape_prob_closed),
        ):
            assert np.isclose(
                float(func(*at)), impl(N_, x0, **kwargs), rtol=1e-6, atol=1e-12
            )


@pytest.mark.parametrize("kwargs", param_sets)
@pytest.mark.parametrize("supply", ["supply_non", "supply_mut"])
def test_sweep_wellmixed(kwargs: dict, supply: str):
    res, params = _derive(), calc_load_params(**kwargs)
    at = {u: params[supply], r: params["growth_mut"], p: params["pi_mut"]}
    impl = calc_sweep_time(*at.values())
    assert np.isclose(float(res["sweep_mixed"].subs(at)), impl, rtol=1e-9)
    # leading-order form against the exact integral, and against quadrature
    exact = float(res["sweep_mixed_exact"].subs(at).evalf())
    assert np.isclose(exact, impl, rtol=1e-2)
    unswept = sp.lambdify(t, res["unswept_mixed"].subs(at), "numpy")
    quad = sp_integrate.quad(unswept, 0, 10 * impl, limit=200)[0]
    assert np.isclose(quad, exact, rtol=1e-6)
    for t_ in 10.0, 50.0, 100.0:
        assert np.isclose(
            float(res["unswept_mixed"].subs({**at, t: t_})),
            calc_unswept_prob(t_, *at.values()),
            rtol=1e-9,
        )


@pytest.mark.parametrize("kwargs", param_sets)
@pytest.mark.parametrize("supply", ["supply_non", "supply_mut"])
@pytest.mark.parametrize("deme_size", [32, 256, 2048])
def test_sweep_lattice(kwargs: dict, supply: str, deme_size: int):
    res, params = _derive(), calc_load_params(**kwargs)
    at = {
        u: params[supply],
        r: params["growth_mut"],
        p: params["pi_mut"],
        n: deme_size,
    }
    fill = float(res["fill"].subs(at))
    assert np.isclose(fill, calc_fill_time(deme_size, at[r], at[p]))
    impl = calc_sweep_time(at[u], at[r], at[p], deme_size)
    assert np.isclose(float(res["sweep_lattice"].subs(at)), impl, rtol=1e-9)
    unswept = sp.lambdify(t, res["unswept_lattice"].subs(at), "numpy")
    quad = fill + sp_integrate.quad(unswept, fill, 10 * impl, limit=200)[0]
    assert np.isclose(quad, impl, rtol=1e-6)
    for mult in 1.5, 3.0, 10.0:
        assert np.isclose(
            float(res["unswept_lattice"].subs({**at, t: mult * fill})),
            calc_unswept_prob(mult * fill, at[u], at[r], at[p], deme_size),
            rtol=1e-9,
        )


@pytest.mark.parametrize("kwargs", param_sets)
@pytest.mark.parametrize("deme_size", [None, 256])
@pytest.mark.parametrize("x0", [0.5, 0.1, 0.01])
def test_fixprob(kwargs: dict, deme_size: int, x0: float):
    res = _derive()
    lattice = deme_size is not None
    catchup = _lambdify(res["catchup"], kwargs, lattice)
    fixprob = _lambdify(res["fixprob"], kwargs, lattice)
    # the implementation treats N <= deme_size as a single well-mixed deme
    sizes = np.logspace(3 if lattice else 1, 11, 33)
    for N_, K_ in it.product(sizes, (1, 2, 3, 5)):
        at = (N_, K_, x0, deme_size or 1)
        assert np.isclose(
            float(catchup(*at)),
            calc_catchup_prob_closed(
                N_, K_, x0, deme_size=deme_size, **kwargs
            )[0],
            rtol=1e-6,
            atol=1e-12,
        )
        assert np.isclose(
            float(fixprob(*at)),
            calc_mutator_fixprob_closed(
                N_, K_, x0, deme_size=deme_size, **kwargs
            ),
            rtol=1e-6,
            atol=1e-12,
        )


@pytest.mark.parametrize("kwargs", param_sets)
@pytest.mark.parametrize("deme_size", [None, 256])
@pytest.mark.parametrize("K_", [1, 2, 3, 4])
def test_ncrit_upper(kwargs: dict, deme_size: int, K_: int):
    res, params = _derive(), calc_load_params(**kwargs)
    odds = (params["supply_non"] / params["supply_mut"]) ** (K_ - 1)
    at = {
        u_n: params["supply_non"],
        res["share"]: odds / (1 + odds),  # x0 = 1/2, mutators escaped
        T: calc_sweep_time(
            params["supply_mut"],
            params["growth_mut"],
            params["pi_mut"],
            deme_size,
        ),
        x_0: 0.5,
    }
    assert np.isclose(
        float(res["ncrit_upper"].subs(at)),
        calc_ncrit_closed(K_, "upper", deme_size=deme_size, **kwargs),
        rtol=1e-6,
    )
