"""Domain-free tests for the generic Lagrangian primitive.

These exercise :class:`polar_high.LagrangianProblem` and the new
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

import threading
import time

import numpy as np
import polars as pl
import pytest

import polar_high as fp
from polar_high import CouplingEntry, CouplingSpec, LagrangianProblem, Problem, WarmProblem
from polar_high.lagrangian import _prewarm_global_scheduler

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


def _two_region_consensus_problem() -> LagrangianProblem:
    """Build the standard 2-region ``x_A == x_B`` consensus problem."""
    p_a = _demand_problem(demand=4.0, cost=1.0)
    p_b = _demand_problem(demand=2.0, cost=1.0)
    spec = CouplingSpec(
        entries=[
            CouplingEntry(0, "x", [(0,)], +1.0),
            CouplingEntry(1, "x", [(0,)], -1.0),
        ],
        rhs=0.0,
    )
    return LagrangianProblem([p_a, p_b], [spec])


def test_progress_callback_invoked_per_iteration() -> None:
    """``progress_callback`` fires once per outer iteration with that
    iteration's log dict, plus once for the final-summary marker
    (``iter == -1``)."""
    lp = _two_region_consensus_problem()
    calls: list[dict] = []
    sol = lp.solve(
        max_iters=5,
        tol=1e-12,
        step=0.5,
        min_iters=5,
        progress_callback=calls.append,
    )
    iter_entries = [c for c in calls if c["iter"] != -1]
    final_entries = [c for c in calls if c["iter"] == -1]
    # One per outer iteration actually run.
    assert len(iter_entries) == sol.iterations
    assert iter_entries[0].keys() >= {
        "iter",
        "alpha_k",
        "max_abs_residual",
        "total_obj",
    }
    # Exactly one final-summary marker carrying the dual / recovered totals.
    assert len(final_entries) == 1
    assert final_entries[0]["best_dual_total"] == sol.best_dual_total
    assert final_entries[0]["recovered_total"] == sol.recovered_total


def test_progress_callback_default_none_is_silent() -> None:
    """No callback (default) preserves the original behaviour."""
    lp = _two_region_consensus_problem()
    sol = lp.solve(max_iters=5, tol=1e-12, step=0.5, min_iters=5)
    assert sol.iterations >= 1


def test_progress_callback_exception_never_breaks_solve() -> None:
    """A faulty observer cannot abort the solve."""
    lp = _two_region_consensus_problem()

    def _boom(_entry: dict) -> None:
        raise RuntimeError("observer blew up")

    sol = lp.solve(
        max_iters=3,
        tol=1e-12,
        step=0.5,
        min_iters=3,
        progress_callback=_boom,
    )
    assert sol.iterations >= 1


# ---------------------------------------------------------------------------
# subproblem_col_values retention
# ---------------------------------------------------------------------------


def test_subproblem_col_values_populated_and_aligned() -> None:
    """The retained per-region recovered-primal ``col_value`` arrays are
    full-length, 1-D float arrays, and index-aligned with each region's
    own ``_vars[name].frame['col_id']`` — so a caller can reconstruct the
    region's ``x`` value by indexing into them."""
    lp = _two_region_consensus_problem()
    sol = lp.solve(max_iters=200, tol=1e-9, step=0.5, initial_lambda=0.0, min_iters=20)

    n_sp = len(lp.subproblems)
    assert len(sol.subproblem_col_values) == n_sp
    for cv in sol.subproblem_col_values:
        assert isinstance(cv, np.ndarray)
        assert cv.ndim == 1
        assert cv.dtype == np.float64

    # For each region, reconstruct x via that region's own col_id.  The
    # retained array is that region's FINAL recovered-primal solve in its
    # own col layout, so the reconstructed x must equal that region's
    # reported recovery objective (cost == 1 ⇒ obj == x).
    recon_x = []
    for i, p in enumerate(lp.subproblems):
        col_ids = p._vars["x"].frame["col_id"].to_numpy()
        recon = sol.subproblem_col_values[i][col_ids]
        assert recon.shape == (1,)
        assert recon[0] == pytest.approx(sol.subproblem_objectives[i], rel=1e-9)
        recon_x.append(recon[0])
    # Consensus coupling x_A == x_B holds in the recovered primal: the two
    # regions' reconstructed x values agree.
    assert recon_x[0] == pytest.approx(recon_x[1], rel=1e-9)


def test_subproblem_col_values_trivial_path_full_length() -> None:
    """The trivial/no-coupling early-return path (single subproblem,
    empty couplings) still returns a full-length ``subproblem_col_values``
    populated from the initial solve."""
    p = _demand_problem(demand=4.0, cost=2.5)
    lp = LagrangianProblem([p], [])
    sol = lp.solve(max_iters=10, tol=1e-9)
    assert sol.iterations == 0
    assert len(sol.subproblem_col_values) == 1
    cv = sol.subproblem_col_values[0]
    assert isinstance(cv, np.ndarray)
    assert cv.ndim == 1
    assert cv.dtype == np.float64
    # The region's own col_id reconstructs x = demand = 4.
    col_ids = lp.subproblems[0]._vars["x"].frame["col_id"].to_numpy()
    assert cv[col_ids][0] == pytest.approx(4.0, abs=1e-6)


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


# ---------------------------------------------------------------------------
# Thread-parallel subsolves + per-subsolve callback + HiGHS-log silencing
# ---------------------------------------------------------------------------


def _three_region_consensus_problem() -> LagrangianProblem:
    """Three demand LPs coupled by a chain ``x_0 == x_1`` and
    ``x_1 == x_2`` — real cross-subproblem coupling over 3 regions."""
    p_a = _demand_problem(demand=4.0, cost=1.0)
    p_b = _demand_problem(demand=2.0, cost=1.0)
    p_c = _demand_problem(demand=3.0, cost=1.0)
    spec_ab = CouplingSpec(
        entries=[
            CouplingEntry(0, "x", [(0,)], +1.0),
            CouplingEntry(1, "x", [(0,)], -1.0),
        ],
        rhs=0.0,
    )
    spec_bc = CouplingSpec(
        entries=[
            CouplingEntry(1, "x", [(0,)], +1.0),
            CouplingEntry(2, "x", [(0,)], -1.0),
        ],
        rhs=0.0,
    )
    return LagrangianProblem([p_a, p_b, p_c], [spec_ab, spec_bc])


_SOLVE_KW = dict(max_iters=30, tol=1e-9, step=0.5, initial_lambda=0.0, min_iters=10)


def test_parallel_determinism_2_vs_4_workers() -> None:
    """Both sides parallel (threads=1) ⇒ byte-identical results, so 2 and 4
    workers must agree exactly on every numeric field."""
    sol2 = _three_region_consensus_problem().solve(max_workers=2, **_SOLVE_KW)
    sol4 = _three_region_consensus_problem().solve(max_workers=4, **_SOLVE_KW)

    assert len(sol2.final_lambdas) == len(sol4.final_lambdas)
    for a, b in zip(sol2.final_lambdas, sol4.final_lambdas):
        assert np.array_equal(a, b)
    assert len(sol2.primal_recovery) == len(sol4.primal_recovery)
    for a, b in zip(sol2.primal_recovery, sol4.primal_recovery):
        assert np.array_equal(a, b)
    assert len(sol2.subproblem_col_values) == len(sol4.subproblem_col_values)
    for a, b in zip(sol2.subproblem_col_values, sol4.subproblem_col_values):
        assert np.array_equal(a, b)
    assert sol2.subproblem_objectives == sol4.subproblem_objectives
    assert sol2.total_objective == sol4.total_objective
    assert sol2.best_dual_total == sol4.best_dual_total
    assert sol2.recovered_total == sol4.recovered_total
    assert sol2.iterations == sol4.iterations
    assert sol2.converged == sol4.converged


def test_sequential_vs_parallel_correctness() -> None:
    """The default sequential path (max_workers=1) and parallel (max_workers=4)
    reach the same solution to high tolerance.  HiGHS thread counts differ
    (sequential keeps today's default; parallel forces threads=1), so use
    ``allclose`` rather than exact equality."""
    sol1 = _three_region_consensus_problem().solve(max_workers=1, **_SOLVE_KW)
    sol4 = _three_region_consensus_problem().solve(max_workers=4, **_SOLVE_KW)

    for a, b in zip(sol1.final_lambdas, sol4.final_lambdas):
        assert np.allclose(a, b, rtol=1e-9, atol=1e-9)
    for a, b in zip(sol1.primal_recovery, sol4.primal_recovery):
        assert np.allclose(a, b, rtol=1e-9, atol=1e-9)
    for a, b in zip(sol1.subproblem_col_values, sol4.subproblem_col_values):
        assert np.allclose(a, b, rtol=1e-9, atol=1e-9)
    assert np.allclose(sol1.subproblem_objectives, sol4.subproblem_objectives, rtol=1e-9, atol=1e-9)
    assert sol1.total_objective == pytest.approx(sol4.total_objective, rel=1e-9, abs=1e-9)
    assert sol1.best_dual_total == pytest.approx(sol4.best_dual_total, rel=1e-9, abs=1e-9)


def test_backward_compat_no_new_kwargs() -> None:
    """No new kwargs ⇒ default sequential behaviour matches max_workers=1
    exactly; the solve converges and stays correct."""
    sol_default = _three_region_consensus_problem().solve(**_SOLVE_KW)
    sol_w1 = _three_region_consensus_problem().solve(max_workers=1, **_SOLVE_KW)

    assert sol_default.converged == sol_w1.converged
    assert sol_default.iterations == sol_w1.iterations
    for a, b in zip(sol_default.final_lambdas, sol_w1.final_lambdas):
        assert np.array_equal(a, b)
    assert sol_default.total_objective == sol_w1.total_objective
    assert sol_default.best_dual_total == sol_w1.best_dual_total


def test_backward_compat_default_log_not_silenced(capsys, monkeypatch) -> None:
    """Silencing is opt-in: with no new kwargs the native HiGHS log IS present
    (output_flag is left at HiGHS' default), proving plain existing callers
    keep today's verbose native log.  The silence-on-opt-in contract is pinned
    in :func:`test_silencing_parallel_suppresses_log`."""
    monkeypatch.delenv("POLAR_HIGH_LAGRANGIAN_VERBOSE", raising=False)
    capsys.readouterr()
    _three_region_consensus_problem().solve(**_SOLVE_KW)
    out = capsys.readouterr()
    assert _has_highs_banner(out.out + out.err)


def test_callback_events_parallel() -> None:
    _run_callback_assertions(max_workers=3)


def test_callback_events_sequential() -> None:
    _run_callback_assertions(max_workers=1)


def _run_callback_assertions(*, max_workers: int) -> None:
    """The subsolve_callback fires for every individual subsolve regardless of
    parallelism, with the pinned start/finish schema."""
    lp = _three_region_consensus_problem()
    n = len(lp.subproblems)
    lock = threading.Lock()
    events: list[dict] = []

    def _collect(entry: dict) -> None:
        with lock:
            events.append(dict(entry))

    sol = lp.solve(
        max_iters=3,
        tol=1e-12,
        step=0.5,
        min_iters=3,
        max_workers=max_workers,
        subsolve_callback=_collect,
    )

    for e in events:
        assert e["event"] in {"start", "finish"}
        assert e["phase"] in {"initial", "iterate", "recovery"}
        assert isinstance(e["iter"], int)
        assert isinstance(e["subproblem"], int)

    starts = [e for e in events if e["event"] == "start"]
    finishes = [e for e in events if e["event"] == "finish"]
    assert len(starts) == len(finishes)

    def by_phase(phase, ev):
        return [e for e in events if e["phase"] == phase and e["event"] == ev]

    # initial: exactly 2*n (one start + one finish per region).
    assert len(by_phase("initial", "start")) == n
    assert len(by_phase("initial", "finish")) == n
    # iterate: 2*n per iteration actually run.
    assert len(by_phase("iterate", "start")) == n * sol.iterations
    assert len(by_phase("iterate", "finish")) == n * sol.iterations
    # recovery: 2*n when recovery runs (it does here — primal_tail > 0).
    rec_starts = by_phase("recovery", "start")
    rec_finishes = by_phase("recovery", "finish")
    assert len(rec_starts) == len(rec_finishes)
    assert len(rec_starts) in (0, n)

    # Every optimal finish carries a float "obj"; no start carries "obj".
    for e in starts:
        assert "obj" not in e
    for e in finishes:
        if "obj" in e:
            assert isinstance(e["obj"], float)


def _has_highs_banner(text: str) -> bool:
    return "HiGHS" in text


def test_silencing_parallel_suppresses_log(capsys, monkeypatch) -> None:
    """max_workers>1 with POLAR_HIGH_LAGRANGIAN_VERBOSE unset ⇒ no HiGHS
    banner reaches stdout."""
    monkeypatch.delenv("POLAR_HIGH_LAGRANGIAN_VERBOSE", raising=False)
    capsys.readouterr()
    _three_region_consensus_problem().solve(max_workers=4, **_SOLVE_KW)
    out = capsys.readouterr()
    assert not _has_highs_banner(out.out)
    assert not _has_highs_banner(out.err)


def test_silencing_verbose_env_reenables_log(capsys, monkeypatch) -> None:
    """POLAR_HIGH_LAGRANGIAN_VERBOSE=1 forces the native log back on even
    under parallel/hooked solves."""
    monkeypatch.setenv("POLAR_HIGH_LAGRANGIAN_VERBOSE", "1")
    capsys.readouterr()
    _three_region_consensus_problem().solve(max_workers=4, **_SOLVE_KW)
    out = capsys.readouterr()
    combined = out.out + out.err
    assert _has_highs_banner(combined)


def _infeasible_demand_problem() -> Problem:
    """A demand LP whose feasible region is empty: ``x >= 200`` but
    ``x <= 100`` ⇒ the initial solve is non-optimal (infeasible)."""
    return _demand_problem(demand=200.0, cost=1.0, upper=100.0)


def test_exception_names_subproblem_and_pool_shuts_down() -> None:
    """An infeasible region at index 1 makes its initial solve non-optimal;
    the RuntimeError names that index, and the ThreadPoolExecutor shuts down
    (no leaked worker threads) afterwards."""
    p_a = _demand_problem(demand=4.0, cost=1.0)
    p_b = _infeasible_demand_problem()
    p_c = _demand_problem(demand=3.0, cost=1.0)
    spec_ab = CouplingSpec(
        entries=[
            CouplingEntry(0, "x", [(0,)], +1.0),
            CouplingEntry(1, "x", [(0,)], -1.0),
        ],
        rhs=0.0,
    )
    spec_bc = CouplingSpec(
        entries=[
            CouplingEntry(1, "x", [(0,)], +1.0),
            CouplingEntry(2, "x", [(0,)], -1.0),
        ],
        rhs=0.0,
    )
    lp = LagrangianProblem([p_a, p_b, p_c], [spec_ab, spec_bc])

    baseline = threading.active_count()
    with pytest.raises(RuntimeError, match="subproblem 1"):
        lp.solve(max_workers=4, **_SOLVE_KW)
    # Pool shut down: active thread count settles back to baseline.
    deadline = time.time() + 5.0
    while threading.active_count() > baseline and time.time() < deadline:
        time.sleep(0.02)
    assert threading.active_count() <= baseline


def test_exception_on_iteration_solve_names_subproblem() -> None:
    """A *per-iteration* (not initial) non-optimal subsolve must surface as a
    RuntimeError naming the offending subproblem via the iterate-loop's
    ``iter {it}: subproblem {i}`` message, and the pool must shut down.

    The LP feasible set is fixed across iterations (only costs change in the
    iterate barrier), so we force a mid-iteration non-optimal Solution by
    making subproblem 1's WarmProblem.solve return a non-optimal result on its
    SECOND call (iteration 1) — the initial build (first call) stays optimal so
    the barrier reaches the iterate loop.
    """
    lp = _three_region_consensus_problem()
    target_wp = lp.warm_problems[1]
    orig_solve = WarmProblem.solve
    call_counts: dict[int, int] = {}
    lock = threading.Lock()

    def _flaky_solve(self, *, options=None):
        sol = orig_solve(self, options=options)
        if self is target_wp:
            with lock:
                call_counts[id(self)] = call_counts.get(id(self), 0) + 1
                n = call_counts[id(self)]
            if n >= 2:  # 1st call = initial build (stay optimal); 2nd = iterate
                sol.optimal = False
        return sol

    WarmProblem.solve = _flaky_solve
    baseline = threading.active_count()
    try:
        with pytest.raises(RuntimeError, match="iter 1: subproblem 1"):
            lp.solve(max_workers=4, **_SOLVE_KW)
    finally:
        WarmProblem.solve = orig_solve

    deadline = time.time() + 5.0
    while threading.active_count() > baseline and time.time() < deadline:
        time.sleep(0.02)
    assert threading.active_count() <= baseline


def _record_first_solve_threads(monkeypatch) -> dict[int, int]:
    """Monkeypatch ``WarmProblem.solve`` to record, per instance, the thread
    ident of its FIRST solve (the cold build).  Returns the recording dict."""
    first_solve_threads: dict[int, int] = {}
    lock = threading.Lock()
    orig_solve = WarmProblem.solve

    def _recording_solve(self, *, options=None):
        with lock:
            key = id(self)
            if key not in first_solve_threads:
                first_solve_threads[key] = threading.get_ident()
        return orig_solve(self, options=options)

    monkeypatch.setattr(WarmProblem, "solve", _recording_solve)
    return first_solve_threads


def test_initial_build_parallelizes_across_regions_when_prewarmed(monkeypatch) -> None:
    """With ``max_workers >= 2`` and a successful one-time global-scheduler
    prewarm, the COLD initial build fans out ACROSS regions: at least one
    region's FIRST solve runs on a WORKER thread (NOT all on the calling
    thread).  Parallelism is across regions only — each solve stays
    single-threaded (pinned pool).  Pins the cold-parallel build path."""
    # Prewarm must succeed on this box for the parallel cold-build path to run.
    assert _prewarm_global_scheduler(1) is True

    main_ident = threading.get_ident()
    first_solve_threads = _record_first_solve_threads(monkeypatch)

    lp = _three_region_consensus_problem()
    lp.solve(max_workers=4, **_SOLVE_KW)

    assert first_solve_threads, "no solves were recorded"
    idents = set(first_solve_threads.values())
    # At least one cold build ran OFF the calling thread (across-region fan-out).
    assert idents != {main_ident}, (
        "cold initial build did not parallelize: all first solves ran on the main thread"
    )


def test_initial_build_sequential_on_main_thread_when_prewarm_fails(monkeypatch) -> None:
    """When the one-time prewarm FAILS (monkeypatched to False) the COLD initial
    build falls back to a sequential loop on the calling thread — every
    WarmProblem's FIRST solve runs on the main thread — while the WARM
    iterations still parallelize (worker threads appear for non-first solves)
    and the result is still correct."""
    monkeypatch.setattr("polar_high.lagrangian._prewarm_global_scheduler", lambda threads=1: False)

    main_ident = threading.get_ident()
    first_solve_threads: dict[int, int] = {}
    all_solve_threads: set[int] = set()
    lock = threading.Lock()
    orig_solve = WarmProblem.solve

    def _recording_solve(self, *, options=None):
        with lock:
            ident = threading.get_ident()
            all_solve_threads.add(ident)
            key = id(self)
            if key not in first_solve_threads:
                first_solve_threads[key] = ident
        return orig_solve(self, options=options)

    monkeypatch.setattr(WarmProblem, "solve", _recording_solve)

    lp = _three_region_consensus_problem()
    sol = lp.solve(max_workers=4, **_SOLVE_KW)

    # Every cold build (first solve) ran on the calling thread.
    assert first_solve_threads, "no solves were recorded"
    for ident in first_solve_threads.values():
        assert ident == main_ident
    # Warm iterations still parallelized: some solve ran off the main thread.
    assert all_solve_threads != {main_ident}, (
        "warm iterations did not parallelize in the sequential-fallback path"
    )

    # And the fallback path is still correct vs. the parallel cold-build path.
    ref = _three_region_consensus_problem().solve(max_workers=4, **_SOLVE_KW)
    assert sol.total_objective == pytest.approx(ref.total_objective, rel=1e-9, abs=1e-9)


def test_cold_parallel_vs_cold_sequential_bit_identical(monkeypatch) -> None:
    """The parallel cold build (prewarm succeeds) and the sequential cold build
    (prewarm forced False) must be BIT-IDENTICAL — pins "parallel cold build ==
    sequential cold build" on objectives, lambdas, recovery and col_values."""
    sol_parallel = _three_region_consensus_problem().solve(max_workers=4, **_SOLVE_KW)

    monkeypatch.setattr("polar_high.lagrangian._prewarm_global_scheduler", lambda threads=1: False)
    sol_seq = _three_region_consensus_problem().solve(max_workers=4, **_SOLVE_KW)

    assert len(sol_parallel.final_lambdas) == len(sol_seq.final_lambdas)
    for a, b in zip(sol_parallel.final_lambdas, sol_seq.final_lambdas):
        assert np.array_equal(a, b)
    assert len(sol_parallel.primal_recovery) == len(sol_seq.primal_recovery)
    for a, b in zip(sol_parallel.primal_recovery, sol_seq.primal_recovery):
        assert np.array_equal(a, b)
    assert len(sol_parallel.subproblem_col_values) == len(sol_seq.subproblem_col_values)
    for a, b in zip(sol_parallel.subproblem_col_values, sol_seq.subproblem_col_values):
        assert np.array_equal(a, b)
    assert sol_parallel.subproblem_objectives == sol_seq.subproblem_objectives
    assert sol_parallel.total_objective == sol_seq.total_objective
    assert sol_parallel.best_dual_total == sol_seq.best_dual_total
    assert sol_parallel.recovered_total == sol_seq.recovered_total
    assert sol_parallel.iterations == sol_seq.iterations
    assert sol_parallel.converged == sol_seq.converged


def test_prewarm_global_scheduler_returns_true_and_run_works() -> None:
    """:func:`_prewarm_global_scheduler` returns True on this box, and a fresh
    Highs ``run()`` with NO ``threads`` option works afterward (smoke) — i.e.
    the run inherits the pinned single-thread pool."""
    import highspy

    assert _prewarm_global_scheduler(1) is True

    # Fresh Highs instance, no threads option ⇒ must still solve optimally
    # (inherits the pre-pinned single-thread scheduler).
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    lp = highspy.HighsLp()
    lp.num_col_ = 1
    lp.num_row_ = 0
    lp.col_cost_ = np.array([1.0])
    lp.col_lower_ = np.array([0.0])
    lp.col_upper_ = np.array([5.0])
    lp.row_lower_ = np.array([])
    lp.row_upper_ = np.array([])
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_ = 1
    lp.a_matrix_.num_row_ = 0
    lp.a_matrix_.start_ = np.array([0, 0])
    lp.a_matrix_.index_ = np.array([], dtype=np.int32)
    lp.a_matrix_.value_ = np.array([])
    h.passModel(lp)
    h.run()
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
