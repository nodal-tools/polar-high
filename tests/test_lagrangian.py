"""Domain-free tests for the generic Lagrangian primitive.

These exercise :class:`polar_high_opt.LagrangianProblem` and the new
:class:`WarmProblem` array-form methods (``update_obj_coef_array``,
``fix_cols``) on synthetic LPs only — no flextool dependency.

Test inventory:
  1. ``test_two_subproblem_consensus_closed_form``
  2. ``test_single_subproblem_no_couplings_matches_problem_solve``
  3. ``test_couplingspec_validation_subproblem_idx_out_of_range``
  4. ``test_couplingspec_validation_unknown_var``
  5. ``test_couplingspec_validation_bad_dim_tuple``
  6. ``test_max_iters_without_convergence_returns_unconverged``
  7. ``test_warm_update_obj_coef_array_matches_unvectorised``
  8. ``test_warm_fix_cols_matches_unvectorised``
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polar_high_opt as fp
from polar_high_opt import CouplingEntry, CouplingSpec, LagrangianProblem, Problem, WarmProblem

# ---------------------------------------------------------------------------
# Helper: build a tiny 1-cell maximisation LP   max c · x   s.t. x ≤ ub
# ---------------------------------------------------------------------------


def _demand_problem(
    demand: float, cost: float = 1.0, upper: float = 100.0, var_name: str = "x"
) -> Problem:
    """A minimisation LP   min cost · x   s.t.  x >= demand,  0 ≤ x ≤ upper.

    Optimum is x = demand, obj = cost * demand.  Used to exercise
    Lagrangian decomposition in the natural LP sense (min).
    """
    p = Problem()
    idx = pl.DataFrame({"k": [0]})
    x = p.add_var(var_name, "k", idx, lower=0.0, upper=upper)
    cost_p = fp.Param(("k",), pl.DataFrame({"k": [0], "value": [float(cost)]}))
    p.set_objective(cost_p * x, sense="min")
    p.add_cstr(
        "demand",
        over=None,
        sense=">=",
        lhs_terms={"sum_x": fp.Sum(x.to_expr(), over=("k",))},
        rhs_terms={"d": float(demand)},
    )
    return p


# ---------------------------------------------------------------------------
# 1. Closed-form 2-subproblem LP
# ---------------------------------------------------------------------------


def test_two_subproblem_consensus_closed_form() -> None:
    """Two equal-cost minimisation LPs coupled by ``x_A == x_B``:

      min  x_A    s.t.   x_A >= 4,  0 ≤ x_A ≤ 100
      min  x_B    s.t.   x_B >= 2,  0 ≤ x_B ≤ 100
      x_A == x_B

    Without the coupling each LP picks its own demand floor (4 and 2
    respectively).  With the coupling x_A = x_B, the joint feasible
    floor is max(4, 2) = 4; cost = 1*4 + 1*4 = 8.

    Equal cost ⇒ no integrality gap on the dual: the dual optimum
    equals the LP's primal optimum.  Subgradient hits the right
    Lagrangian value at iter 2 and the best-dual tracker reports it
    to floating-point precision.  (The recovered primal can drift
    upwards on this LP because LP1's demand-floor x_a >= 4 makes the
    consensus value 0.5*(4+x_b) infeasible whenever x_b > 4 — the
    fix-and-resolve falls back to the last iter's Lagrangian value.
    For the *generic* dual-bound assertion we use best_dual.)
    """
    p_a = _demand_problem(demand=4.0, cost=1.0)
    p_b = _demand_problem(demand=2.0, cost=1.0)
    spec = CouplingSpec(
        entries=[
            CouplingEntry(0, "x", [(0,)], +1.0),
            CouplingEntry(1, "x", [(0,)], -1.0),
        ],
        rhs=0.0,
    )
    lp = LagrangianProblem([p_a, p_b], [spec])
    sol = lp.solve(max_iters=200, tol=1e-9, step=0.5, initial_lambda=0.0, min_iters=20)
    # Best dual is the tight LB on a min problem.  For this LP the
    # dual gap is zero (no integrality), so best_dual == LP optimum
    # to floating-point precision.
    rel_dual = abs(sol.best_dual_total - 8.0) / 8.0
    assert rel_dual < 1e-9, (
        f"best_dual {sol.best_dual_total} differs from closed-form 8.0 by rel {rel_dual}"
    )
    # And total_objective == best_dual (current report policy).
    assert sol.report_kind == "best_dual"
    assert sol.total_objective == pytest.approx(sol.best_dual_total, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Trivial 1-subproblem case
# ---------------------------------------------------------------------------


def test_single_subproblem_no_couplings_matches_problem_solve() -> None:
    """1 subproblem, 0 couplings → matches Problem.solve() exactly."""
    p = _demand_problem(demand=4.0, cost=2.5)
    direct_sol = p.solve()
    assert direct_sol.optimal

    p2 = _demand_problem(demand=4.0, cost=2.5)
    lp = LagrangianProblem([p2], [])
    sol = lp.solve(max_iters=10, tol=1e-9)
    assert sol.converged
    assert sol.iterations == 0
    assert sol.total_objective == pytest.approx(direct_sol.obj, rel=1e-12)


# ---------------------------------------------------------------------------
# 3-5. CouplingSpec validation
# ---------------------------------------------------------------------------


def test_couplingspec_validation_subproblem_idx_out_of_range() -> None:
    p_a = _demand_problem(demand=1.0)
    p_b = _demand_problem(demand=1.0)
    spec = CouplingSpec(
        entries=[
            CouplingEntry(0, "x", [(0,)], +1.0),
            CouplingEntry(99, "x", [(0,)], -1.0),  # bogus
        ]
    )
    lp = LagrangianProblem([p_a, p_b], [spec])
    with pytest.raises(ValueError, match="out of range"):
        lp.solve(max_iters=2, tol=1e-9)


def test_couplingspec_validation_unknown_var() -> None:
    p_a = _demand_problem(demand=1.0, var_name="x")
    p_b = _demand_problem(demand=1.0, var_name="x")
    spec = CouplingSpec(
        entries=[
            CouplingEntry(0, "y_does_not_exist", [(0,)], +1.0),
            CouplingEntry(1, "x", [(0,)], -1.0),
        ]
    )
    lp = LagrangianProblem([p_a, p_b], [spec])
    with pytest.raises(ValueError, match="not declared in subproblem"):
        lp.solve(max_iters=2, tol=1e-9)


def test_couplingspec_validation_bad_dim_tuple() -> None:
    p_a = _demand_problem(demand=1.0)
    p_b = _demand_problem(demand=1.0)
    spec = CouplingSpec(
        entries=[
            CouplingEntry(0, "x", [(0,)], +1.0),
            CouplingEntry(1, "x", [(99,)], -1.0),  # dim value doesn't exist
        ]
    )
    lp = LagrangianProblem([p_a, p_b], [spec])
    with pytest.raises(KeyError, match="does not resolve"):
        lp.solve(max_iters=2, tol=1e-9)


# ---------------------------------------------------------------------------
# 6. max_iters without convergence
# ---------------------------------------------------------------------------


def test_max_iters_without_convergence_returns_unconverged() -> None:
    """Tight tolerance + few iters with mismatched demands → can't
    converge in 2 iters; returns converged=False cleanly."""
    # Mismatched demands force a non-zero residual at iteration 1
    # (x_a == 5, x_b == 3, residual = 2).  With tiny step the dual
    # can't move enough in 2 iters to bring the residual under 1e-12.
    p_a = _demand_problem(demand=5.0, cost=1.0)
    p_b = _demand_problem(demand=3.0, cost=1.0)
    spec = CouplingSpec(
        entries=[
            CouplingEntry(0, "x", [(0,)], +1.0),
            CouplingEntry(1, "x", [(0,)], -1.0),
        ]
    )
    lp = LagrangianProblem([p_a, p_b], [spec])
    sol = lp.solve(max_iters=2, tol=1e-12, step=0.001, min_iters=2)
    assert sol.converged is False
    # iterations == 2 (it ran the full max_iters loop)
    assert sol.iterations == 2


# ---------------------------------------------------------------------------
# 7-8. WarmProblem.update_obj_coef_array / fix_cols vs un-vectorised
# ---------------------------------------------------------------------------


def _build_three_cell_lp() -> Problem:
    """3-cell LP:  min Σ_k c_k · x_k   s.t.  x_k ∈ [0, 10]."""
    p = Problem()
    idx = pl.DataFrame({"k": [0, 1, 2]})
    x = p.add_var("x", "k", idx, lower=0.0, upper=10.0)
    # Objective: c · x with all coefs = 1 initially
    coef = fp.Param(("k",), pl.DataFrame({"k": [0, 1, 2], "value": [1.0, 1.0, 1.0]}), name="cost")
    p.set_objective(coef * x, sense="min")
    # Add a dummy constraint to force a feasible solve
    p.add_cstr(
        "ub_total",
        over=None,
        sense="<=",
        lhs_terms={"sum_x": fp.Sum(x.to_expr(), over=("k",))},
        rhs_terms={"limit": 30.0},
    )
    return p


def test_warm_update_obj_coef_array_matches_unvectorised() -> None:
    """Pushing an array of new objective coefs via the new vectorised
    method must yield the same solve result as updating via the old
    Param-based ``update_obj_coef`` path."""
    # Path A: vectorised update_obj_coef_array
    p_a = _build_three_cell_lp()
    wp_a = WarmProblem(p_a)
    wp_a.solve()
    new_coefs = np.array([2.0, -3.0, 1.5], dtype=np.float64)
    wp_a.update_obj_coef_array("x", [(0,), (1,), (2,)], new_coefs)
    sol_a = wp_a.solve()

    # Path B: equivalent via update_obj_coef with a Param
    p_b = _build_three_cell_lp()
    wp_b = WarmProblem(p_b)
    wp_b.solve()
    new_param = fp.Param(("k",), pl.DataFrame({"k": [0, 1, 2], "value": new_coefs.tolist()}))
    wp_b.update_obj_coef("x", new_param)
    sol_b = wp_b.solve()

    assert sol_a.optimal and sol_b.optimal
    assert sol_a.obj == pytest.approx(sol_b.obj, rel=1e-12)
    np.testing.assert_allclose(sol_a.col_value, sol_b.col_value, rtol=1e-12)


def test_warm_fix_cols_matches_unvectorised() -> None:
    """Vectorised fix_cols with a list of dim_tuples must match the
    effect of one-cell-at-a-time changeColsBounds calls."""
    # Path A: vectorised fix_cols
    p_a = _build_three_cell_lp()
    wp_a = WarmProblem(p_a)
    wp_a.solve()
    fix_vals = np.array([4.0, 2.5, 1.0], dtype=np.float64)
    wp_a.fix_cols("x", [(0,), (1,), (2,)], fix_vals)
    sol_a = wp_a.solve()

    # Path B: per-cell via changeColsBounds
    p_b = _build_three_cell_lp()
    wp_b = WarmProblem(p_b)
    wp_b.solve()
    h = wp_b._h
    for k, val in zip([0, 1, 2], fix_vals):
        col_id = wp_b.col_id_of_var("x", (k,))
        h.changeColsBounds(
            1,
            np.array([col_id], dtype=np.int32),
            np.array([val], dtype=np.float64),
            np.array([val], dtype=np.float64),
        )
    sol_b = wp_b.solve()

    assert sol_a.optimal and sol_b.optimal
    assert sol_a.obj == pytest.approx(sol_b.obj, rel=1e-12)
    np.testing.assert_allclose(sol_a.col_value, sol_b.col_value, rtol=1e-12)
