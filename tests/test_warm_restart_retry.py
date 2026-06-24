"""Warm-restart retry for :meth:`WarmProblem.solve(retry_on_unknown=True)`.

The Benders master appends an optimality cut row each iteration and re-solves.
On a WELL-SCALED master the warm re-solve (retaining the prior basis so dual
simplex hot-starts after the new ``>=`` row) stays ``kOptimal`` and the cold
``clearSolver()`` re-presolve is never needed.  On a BADLY-SCALED master (cut
coefficients O(1e9), like the real region reduced-cost slopes) HiGHS leaves a
primal-infeasible point and returns the not-determined ``kUnknown`` status; the
warm objective is then WRONG, and only a ``clearSolver()`` cold fallback
recovers the certified optimum.

These hand-verified tiny LPs mirror the spec's "Master tractability" probe
(``specs/benders_option_c.md`` §1).  ``retry_on_unknown=True`` runs warm first
and falls back to cold ONLY on ``kUnknown``; ``retry_on_unknown=False`` (the
default) is byte-identical to the legacy ``solve()``.
"""

from __future__ import annotations

import polars as pl
import pytest

import polar_high as fp


def _build_master(*, cost: float, eta_floor: float) -> fp.Problem:
    """A tiny 3-var Benders master (mirrors the real master shape)::

        min  cost*C + eta
        s.t. C - f >= 0           (capacity row "cap")
             0 <= f <= 10
             0 <= C <= 100
             eta_floor <= eta <= 1e11

    ``cost`` scales the invest term; ``eta_floor`` sets the (signed) lower
    bound on the recourse var.  With ``cost`` and the appended cut slopes both
    O(1) this is well-conditioned; with ``cost`` and slopes O(1e9) (and a wide
    eta floor) the warm re-solve after a cut append flips to kUnknown.
    """
    p = fp.Problem()
    idx = pl.DataFrame({"i": [0]})
    f = p.add_var("f", "i", idx, lower=0.0, upper=10.0)
    cap = p.add_var("C", "i", idx, lower=0.0, upper=100.0)
    eta = p.add_var("eta", "i", idx, lower=eta_floor, upper=1.0e11)
    # Capacity: C - f >= 0.
    p.add_cstr(
        "cap",
        over=idx,
        sense=">=",
        lhs_terms={"C": cap, "negf": f * (-1.0)},
        rhs_terms={"zero": fp.Param((), pl.DataFrame({"value": [0.0]}))},
    )
    p.set_objective(cap.to_expr() * cost + eta.to_expr(), sense="min")
    return p


# -- Test 1: well-scaled — warm append stays optimal, no fallback ----------


def test_well_scaled_warm_append_stays_optimal() -> None:
    """Well-conditioned master: append a BINDING cut, re-solve with
    ``retry_on_unknown=True``.  The warm path itself returns kOptimal (so the
    cold fallback never fires) and the optimum is the hand-verified value."""
    import highspy

    wp = fp.WarmProblem(_build_master(cost=1.0, eta_floor=-1.0e2))
    sol0 = wp.solve()
    # iter-0: cost*C + eta with eta free down to -100, C>=f>=0 ⇒ C=f=0,
    # eta=-100, obj=-100.
    assert sol0.optimal
    assert sol0.obj == pytest.approx(-100.0, abs=1e-9)

    f_col = wp.col_id_of_var("f", (0,))
    eta_col = wp.col_id_of_var("eta", (0,))

    # Cut: eta >= 50 - 2*f  <=>  eta + 2*f >= 50.  At (f=0, eta=-100): binds.
    # Slope 2 > C's unit cost ⇒ pushing f up is STRICTLY beneficial (no tie),
    # so the optimum is unique and hand-verifiable.
    wp.add_cut_row([eta_col, f_col], [1.0, 2.0], 50.0)

    sol1 = wp.solve(retry_on_unknown=True)
    assert sol1.optimal
    # Warm path alone (NO cold fallback) must already be kOptimal — i.e. the
    # live handle is optimal, so the retry branch was a no-op.
    assert wp._h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    # New optimum: raising f by 1 costs 1 (on C, since C=f) but lets eta drop
    # by 2 → net -1, so f→10: f=C=10, eta = 50 - 2*10 = 30, obj = 10 + 30 = 40.
    assert sol1.col_value[f_col] == pytest.approx(10.0, abs=1e-9)
    assert sol1.col_value[eta_col] == pytest.approx(30.0, abs=1e-9)
    assert sol1.obj == pytest.approx(40.0, abs=1e-9)


# -- Test 2: badly-scaled — warm kUnknown ⇒ cold fallback recovers optimum --


def _badly_scaled_after_cut():
    """Build the badly-scaled master, append the O(1e9)-slope cut, and return
    ``(wp, f_col, eta_col)`` poised for the warm re-solve.  Asserts that the
    bare warm re-solve genuinely returns kUnknown (the failure this test
    locks)."""
    import highspy

    COST = 7.1e5  # invest cost like the real pipes
    SLOPE = 2.45e9  # cut slope like the region col_dual reduced costs
    wp = fp.WarmProblem(_build_master(cost=COST, eta_floor=-1.0e11))
    wp.solve()
    f_col = wp.col_id_of_var("f", (0,))
    eta_col = wp.col_id_of_var("eta", (0,))
    # Cut: eta - SLOPE*f >= -SLOPE*10  (binds, pulls eta up steeply with f).
    wp.add_cut_row([eta_col, f_col], [1.0, -SLOPE], -SLOPE * 10.0)
    return wp, f_col, eta_col, highspy


def test_badly_scaled_bare_warm_returns_kunknown() -> None:
    """Lock the precondition: a bare warm re-solve (no retry) on the
    badly-scaled master after the cut append returns kUnknown — the transient
    the retry exists to repair.  If HiGHS ever stops doing this the fallback
    test below would silently pass for the wrong reason."""
    wp, _f_col, _eta_col, highspy = _badly_scaled_after_cut()
    sol_warm = wp.solve()  # default: NO retry, bare warm
    assert wp._h.getModelStatus() == highspy.HighsModelStatus.kUnknown
    assert not sol_warm.optimal


def test_badly_scaled_retry_falls_back_to_cold_and_certifies() -> None:
    """``retry_on_unknown=True`` detects the kUnknown and calls clearSolver,
    returning the CERTIFIED optimum — not the wrong primal-infeasible value the
    bare warm path leaves."""
    wp, f_col, eta_col, highspy = _badly_scaled_after_cut()

    sol = wp.solve(retry_on_unknown=True)
    # After the cold fallback the live handle is determined-optimal.
    assert wp._h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    assert sol.optimal
    # Certified optimum (cross-checked against an independent cold solve below):
    # minimise COST*C + eta with C=f and eta >= SLOPE*(f-10).  eta's floor -1e11
    # lets eta drop to SLOPE*(f-10) (negative for f<10).  d obj/df ≈ COST + SLOPE
    # (>0) so the optimum drives f→0: f=C=0, eta = -SLOPE*10, obj = -SLOPE*10.
    expected_obj = -2.45e9 * 10.0
    assert sol.obj == pytest.approx(expected_obj, rel=1e-9)
    assert sol.col_value[f_col] == pytest.approx(0.0, abs=1e-3)

    # Independent certified cross-check: rebuild + cold-solve the SAME model
    # (cut included) from scratch and confirm the fallback matched it.
    cold = fp.WarmProblem(_build_master(cost=7.1e5, eta_floor=-1.0e11))
    cold.solve()
    f2 = cold.col_id_of_var("f", (0,))
    e2 = cold.col_id_of_var("eta", (0,))
    cold.add_cut_row([e2, f2], [1.0, -2.45e9], -2.45e9 * 10.0)
    cold._h.clearSolver()
    sol_cold = cold.solve()
    assert sol_cold.optimal
    assert sol_cold.obj == pytest.approx(sol.obj, rel=1e-9)


# -- Test 3: default (retry_on_unknown=False) is byte-identical -------------


def test_default_retry_false_is_byte_identical() -> None:
    """The default ``solve()`` and ``solve(retry_on_unknown=False)`` produce
    identical Solutions on a well-scaled master with an appended cut — proving
    the new keyword does not perturb the legacy path."""
    import numpy as np

    def _run(retry_kw: dict):
        wp = fp.WarmProblem(_build_master(cost=1.0, eta_floor=-1.0e2))
        wp.solve()
        f_col = wp.col_id_of_var("f", (0,))
        eta_col = wp.col_id_of_var("eta", (0,))
        wp.add_cut_row([eta_col, f_col], [1.0, 1.0], 50.0)
        return wp.solve(**retry_kw)

    legacy = _run({})  # bare solve() — exactly today's call
    explicit = _run({"retry_on_unknown": False})

    assert legacy.optimal == explicit.optimal
    assert legacy.obj == explicit.obj  # exact, not approx — same code path
    assert np.array_equal(legacy.col_value, explicit.col_value)
    assert np.array_equal(legacy.row_dual, explicit.row_dual)
    assert np.array_equal(legacy.col_dual, explicit.col_dual)
