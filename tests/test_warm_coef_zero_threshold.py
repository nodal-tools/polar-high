"""Warm-path small-coefficient cutoff parity.

The initial-build small-coefficient cutoff (``Problem.coef_zero_threshold``)
floors any LP matrix coefficient or RHS row-bound with ``abs < threshold``
to exactly ``0.0`` at the two build chokepoints.  These tests pin the
*warm* in-place equivalents:

  * ``WarmProblem.update_param`` re-derives matrix coefficients and pushes
    them via ``h.changeCoeff`` — a coefficient that newly lands in
    ``(0, threshold)`` on a warm roll must be floored.
  * ``WarmProblem.update_rhs`` re-derives row bounds and pushes them via
    ``h.changeRowsBounds`` — an RHS that newly lands in ``(0, threshold)``
    must be floored, while ``±inf`` one-sided sentinels survive verbatim.

Default threshold ``0.0`` must leave warm updates byte-identical (no
flooring at all).
"""

from __future__ import annotations

import numpy as np
import polars as pl

import polar_high as fp


def _build_coef_problem(coef_value: float, demand_value: float) -> tuple[fp.Problem, fp.Param]:
    """One constraint ``coef * v_flow == demand`` over a 3-step index, with
    the LHS coefficient carried by a *named* Param so it can be tracked and
    warm-updated via ``update_param``.  Returns the Problem and the coef
    Param (so the test can re-issue it with a tiny value)."""
    n_t = 3
    p = fp.Problem()
    t_idx = pl.DataFrame({"t": np.arange(n_t, dtype=np.int64)})
    v_flow = p.add_var("v_flow", "t", t_idx, lower=0.0, upper=1.0e6)

    coef_p = fp.Param(
        ("t",),
        pl.DataFrame({"t": np.arange(n_t, dtype=np.int64), "value": np.full(n_t, coef_value)}),
        name="p_coef",
    )
    demand_p = fp.Param(
        ("t",),
        pl.DataFrame({"t": np.arange(n_t, dtype=np.int64), "value": np.full(n_t, demand_value)}),
    )
    p.add_cstr(
        "balance",
        over=t_idx,
        sense="==",
        lhs_terms={"coef_flow": coef_p * v_flow},
        rhs_terms={"demand": demand_p},
    )
    cost_p = fp.Param(
        ("t",),
        pl.DataFrame({"t": np.arange(n_t, dtype=np.int64), "value": np.ones(n_t)}),
    )
    p.set_objective(cost_p * v_flow, sense="min")
    return p, coef_p


class _CoeffSpy:
    """Wrap a HiGHS instance, recording every changeCoeff value."""

    def __init__(self, h):
        self._h = h
        self.coeffs: list[float] = []

    def changeCoeff(self, r, c, v):  # noqa: N802 (mirror HiGHS API)
        self.coeffs.append(float(v))
        return self._h.changeCoeff(r, c, v)

    def __getattr__(self, name):
        return getattr(self._h, name)


class _RowBoundsSpy:
    """Wrap a HiGHS instance, recording every changeRowsBounds lb/ub array."""

    def __init__(self, h):
        self._h = h
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def changeRowsBounds(self, n, idx, lb, ub):  # noqa: N802
        self.calls.append((np.asarray(lb, dtype=np.float64).copy(),
                            np.asarray(ub, dtype=np.float64).copy()))
        return self._h.changeRowsBounds(n, idx, lb, ub)

    def __getattr__(self, name):
        return getattr(self._h, name)


# --------------------------------------------------------------------------
# update_param (matrix coefficient) flooring
# --------------------------------------------------------------------------


def test_warm_update_param_floors_small_coef():
    p, _ = _build_coef_problem(coef_value=1.0, demand_value=10.0)
    p.coef_zero_threshold = 1e-4
    wp = fp.WarmProblem(p)
    wp.declare_mutable("p_coef")
    wp.solve()

    spy = _CoeffSpy(wp._h)
    wp._h = spy
    tiny = fp.Param(
        ("t",),
        pl.DataFrame({"t": np.arange(3, dtype=np.int64), "value": np.full(3, 1e-8)}),
    )
    wp.update_param("p_coef", tiny)

    assert spy.coeffs, "update_param pushed no coefficients"
    # Every pushed coef (factor==1 here, so value≈1e-8) must be floored.
    assert all(v == 0.0 for v in spy.coeffs), spy.coeffs


def test_warm_update_param_default_off_byte_identical():
    p, _ = _build_coef_problem(coef_value=1.0, demand_value=10.0)
    assert p.coef_zero_threshold == 0.0
    wp = fp.WarmProblem(p)
    wp.declare_mutable("p_coef")
    wp.solve()

    spy = _CoeffSpy(wp._h)
    wp._h = spy
    tiny = fp.Param(
        ("t",),
        pl.DataFrame({"t": np.arange(3, dtype=np.int64), "value": np.full(3, 1e-8)}),
    )
    wp.update_param("p_coef", tiny)

    assert spy.coeffs
    # No flooring: tiny values reach HiGHS unchanged.
    assert all(v == 1e-8 for v in spy.coeffs), spy.coeffs


# --------------------------------------------------------------------------
# update_rhs (row bound) flooring
# --------------------------------------------------------------------------


def test_warm_update_rhs_floors_small_bound():
    p, _ = _build_coef_problem(coef_value=1.0, demand_value=10.0)
    p.coef_zero_threshold = 1e-4
    wp = fp.WarmProblem(p)
    wp.solve()

    spy = _RowBoundsSpy(wp._h)
    wp._h = spy
    tiny_demand = fp.Param(
        ("t",),
        pl.DataFrame({"t": np.arange(3, dtype=np.int64), "value": np.full(3, 1e-8)}),
    )
    wp.update_rhs("balance", tiny_demand)

    assert spy.calls, "update_rhs pushed no row bounds"
    lb, ub = spy.calls[-1]
    # sense == "==" → lb == ub == rhs, both floored to 0.0.
    assert np.all(lb == 0.0), lb
    assert np.all(ub == 0.0), ub


def test_warm_update_rhs_preserves_inf_sentinel():
    # A one-sided (<=) constraint: lb is the -inf sentinel, ub is the rhs.
    n_t = 3
    p = fp.Problem()
    t_idx = pl.DataFrame({"t": np.arange(n_t, dtype=np.int64)})
    v_flow = p.add_var("v_flow", "t", t_idx, lower=0.0, upper=1.0e6)
    demand_p = fp.Param(
        ("t",),
        pl.DataFrame({"t": np.arange(n_t, dtype=np.int64), "value": np.full(n_t, 10.0)}),
    )
    p.add_cstr(
        "cap",
        over=t_idx,
        sense="<=",
        lhs_terms={"flow": v_flow.to_expr()},
        rhs_terms={"demand": demand_p},
    )
    p.set_objective(fp.Param(("t",), pl.DataFrame(
        {"t": np.arange(n_t, dtype=np.int64), "value": -np.ones(n_t)})) * v_flow, sense="min")
    p.coef_zero_threshold = 1e-4
    wp = fp.WarmProblem(p)
    wp.solve()

    spy = _RowBoundsSpy(wp._h)
    wp._h = spy
    tiny_demand = fp.Param(
        ("t",),
        pl.DataFrame({"t": np.arange(n_t, dtype=np.int64), "value": np.full(n_t, 1e-8)}),
    )
    wp.update_rhs("cap", tiny_demand)

    lb, ub = spy.calls[-1]
    # ub (the rhs side) floored to 0.0; lb (-inf sentinel) preserved verbatim.
    assert np.all(ub == 0.0), ub
    assert np.all(np.isneginf(lb)), lb


def test_warm_update_rhs_default_off_byte_identical():
    p, _ = _build_coef_problem(coef_value=1.0, demand_value=10.0)
    assert p.coef_zero_threshold == 0.0
    wp = fp.WarmProblem(p)
    wp.solve()

    spy = _RowBoundsSpy(wp._h)
    wp._h = spy
    tiny_demand = fp.Param(
        ("t",),
        pl.DataFrame({"t": np.arange(3, dtype=np.int64), "value": np.full(3, 1e-8)}),
    )
    wp.update_rhs("balance", tiny_demand)

    lb, ub = spy.calls[-1]
    assert np.all(lb == 1e-8), lb
    assert np.all(ub == 1e-8), ub
