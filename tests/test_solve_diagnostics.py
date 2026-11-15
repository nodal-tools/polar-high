""":meth:`Solution.solve_diagnostics` — normalised HiGHS status snapshot.

The accessor reads ``getModelStatus()`` + ``getInfo()`` off the live handle
into a policy-free :class:`SolveDiagnostics`, and returns ``None`` when no
queryable handle is attached (a synthesised Solution or a read-only shim
without ``getInfo``).  Downstream (flextool) uses this to accept a solve
that HiGHS could not *certify* as optimal but whose primal is feasible.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import polar_high as fp
from polar_high import SolveDiagnostics
from polar_high.engine import Solution


def _build_problem(solver_options: dict) -> fp.Problem:
    # min Σ cost·x, x >= demand ⇒ obj = 3 + 8 + 15 = 26.
    p = fp.Problem()
    idx = pl.DataFrame({"i": [0, 1, 2]})
    x = p.add_var("x", "i", idx, lower=0.0, upper=10.0)
    demand = fp.Param(("i",), pl.DataFrame({"i": [0, 1, 2], "value": [3.0, 4.0, 5.0]}))
    p.add_cstr("meet", over=idx, sense=">=", lhs_terms={"x": x}, rhs_terms={"d": demand})
    cost = fp.Param(("i",), pl.DataFrame({"i": [0, 1, 2], "value": [1.0, 2.0, 3.0]}))
    p.set_objective(cost * x, sense="min")
    p.set_solver_options({"output_flag": False, **solver_options})
    return p


# Non-optimal primary: IPM, zero iterations, presolve off so it cannot solve
# the LP outright, crossover off so nothing cleans up.
_BAD_PRIMARY = {
    "solver": "ipm",
    "ipm_iteration_limit": 0,
    "presolve": "off",
    "run_crossover": "off",
}


def test_diagnostics_on_optimal_solve() -> None:
    sol = _build_problem({}).solve(keep_solver=True)
    assert sol.optimal
    diag = sol.solve_diagnostics()
    assert diag is not None
    assert isinstance(diag, SolveDiagnostics)
    assert diag.model_status_name == "kOptimal"
    assert diag.primal_feasible
    assert diag.num_primal_infeasibilities == 0
    assert diag.primal_dual_objective_error < 1e-6
    assert diag.objective_value == pytest.approx(26.0, abs=1e-9)
    # An LP has no MIP gap: HiGHS reports inf.
    assert diag.mip_gap == math.inf


def test_diagnostics_on_nonoptimal_solve() -> None:
    sol = _build_problem(_BAD_PRIMARY).solve(keep_solver=True)
    assert not sol.optimal
    diag = sol.solve_diagnostics()
    assert diag is not None
    # A truncated IPM certifies nothing — status is not kOptimal.
    assert diag.model_status_name != "kOptimal"
    assert diag.model_status_name == diag.model_status.name


def test_diagnostics_none_when_no_solver_kept() -> None:
    """keep_solver defaults False → no live handle → None."""
    sol = _build_problem({}).solve()
    assert sol.highs is None
    assert sol.solve_diagnostics() is None


def test_diagnostics_none_for_synthesised_solution() -> None:
    syn = Solution(
        optimal=True,
        obj=0.0,
        col_value=np.zeros(1),
        row_dual=np.zeros(1),
        col_names=["x"],
        row_names=["c"],
        vars={},
        highs=None,
    )
    assert syn.solve_diagnostics() is None


def test_diagnostics_none_for_shim_without_getinfo() -> None:
    """A read-only shim (subprocess/commercial path) lacks getInfo/
    getModelStatus → the accessor must return None, not raise."""

    class _Shim:  # mimics _SolHighsShim: no getInfo / getModelStatus
        def getSolution(self):  # noqa: N802 - highspy naming
            return None

    syn = Solution(
        optimal=True,
        obj=0.0,
        col_value=np.zeros(1),
        row_dual=np.zeros(1),
        col_names=["x"],
        row_names=["c"],
        vars={},
        highs=_Shim(),
    )
    assert syn.solve_diagnostics() is None


def test_mip_gap_of_an_optimal_mip_is_zero() -> None:
    # max 3x + 2y, x + y <= 4, x + 3y <= 6, x, y integer in [0, 10]
    # → x = 4, y = 0, obj 12, proven optimal with no gap.
    p = fp.Problem()
    idx = pl.DataFrame({"i": [0]})
    x = p.add_var("x", "i", idx, lower=0.0, upper=10.0, integer=True)
    y = p.add_var("y", "i", idx, lower=0.0, upper=10.0, integer=True)
    p.add_cstr("c1", over=idx, sense="<=", lhs_terms={"x": x, "y": y}, rhs_terms={"r": 4.0})
    p.add_cstr("c2", over=idx, sense="<=", lhs_terms={"x": x, "y": 3.0 * y}, rhs_terms={"r": 6.0})
    p.set_objective(3.0 * x + 2.0 * y, sense="max")
    p.set_solver_options({"output_flag": False, "threads": 1})
    sol = p.solve(keep_solver=True)
    assert sol.optimal
    diag = sol.solve_diagnostics()
    assert diag is not None
    assert diag.objective_value == pytest.approx(12.0, abs=1e-9)
    assert diag.mip_gap == 0.0


def _knapsack(solver_options: dict) -> fp.Problem:
    # 5-row 0/1 knapsack, 400 items, seeded: weights first, then values.
    rng = np.random.default_rng(0)
    w = rng.integers(10, 100, (5, 400))
    v = rng.integers(10, 100, 400)
    cap = w.sum(axis=1) / 2

    p = fp.Problem()
    items = pl.DataFrame({"i": list(range(400))})
    rows = pl.DataFrame({"k": list(range(5))})
    x = p.add_var("x", "i", items, lower=0.0, upper=1.0, integer=True)
    weight = fp.Param(
        ("k", "i"),
        pl.DataFrame(
            {
                "k": np.repeat(np.arange(5), 400).tolist(),
                "i": np.tile(np.arange(400), 5).tolist(),
                "value": w.ravel().astype(float).tolist(),
            }
        ),
    )
    capacity = fp.Param(("k",), pl.DataFrame({"k": list(range(5)), "value": cap.tolist()}))
    p.add_cstr(
        "cap",
        over=rows,
        sense="<=",
        lhs_terms={"w": fp.Sum(weight * x, over="i")},
        rhs_terms={"cap": capacity},
    )
    value = fp.Param(
        ("i",), pl.DataFrame({"i": list(range(400)), "value": v.astype(float).tolist()})
    )
    p.set_objective(fp.Sum(value * x, over="i"), sense="max")
    p.set_solver_options({"output_flag": False, "threads": 1, **solver_options})
    return p


def test_mip_gap_reports_the_gap_a_mip_stopped_at() -> None:
    """The knapsack stopped by ``mip_rel_gap=0.01``: HiGHS ends ``kOptimal``
    with a positive gap (incumbent 15809 against dual bound 15952, i.e.
    143/15809 = 0.9045 %).  The stop rule guarantees ``gap <= 0.01``;
    positivity holds for this seeded instance."""
    sol = _knapsack({"mip_rel_gap": 0.01}).solve(keep_solver=True)
    assert sol.optimal
    diag = sol.solve_diagnostics()
    assert diag is not None
    assert diag.model_status_name == "kOptimal"
    assert 0.0 < diag.mip_gap <= 0.01


def test_mip_gap_without_a_feasible_solution_is_inf() -> None:
    """A maximising MIP stopped before any incumbent: HiGHS reports a
    ``NaN`` gap; the snapshot normalises it to ``inf``."""
    sol = _knapsack({"time_limit": 0.0}).solve(keep_solver=True)
    diag = sol.solve_diagnostics()
    assert diag is not None
    assert not diag.primal_feasible
    assert diag.mip_gap == math.inf


def test_solve_diagnostics_mip_gap_defaults_to_inf() -> None:
    """Code that builds a SolveDiagnostics without ``mip_gap`` keeps
    working and gets ``inf`` ("no gap recorded")."""
    import highspy

    diag = SolveDiagnostics(
        model_status=highspy.HighsModelStatus.kOptimal,
        model_status_name="kOptimal",
        primal_feasible=True,
        dual_feasible=True,
        num_primal_infeasibilities=0,
        num_dual_infeasibilities=0,
        max_primal_infeasibility=0.0,
        max_relative_primal_infeasibility=0.0,
        max_dual_infeasibility=0.0,
        primal_dual_objective_error=0.0,
        objective_value=0.0,
    )
    assert diag.mip_gap == math.inf
    # No user-scaled counters recorded either.
    assert diag.user_bound_scale == 0
    assert diag.user_objective_scale == 0
    assert diag.user_scaled_num_primal_infeasibilities is None
    assert diag.user_scaled_max_primal_infeasibility is None
    assert diag.user_scaled_num_dual_infeasibilities is None
    assert diag.user_scaled_max_dual_infeasibility is None


# --- user-scaled counters ----------------------------------------------------


def _user_scaled_case(**overrides):
    """Hand-built unscaled model + solution (all numbers exact in binary).

    Columns (lower, upper, value, continuous, dual):
      c0 [0, 10]      10.5        cont   0.5
      c1 [0, inf)     -3e-8       int   -0.25
      c2 free         5           cont   1e-8
      c3 [0, 1]       1 + 2e-7    cont   0
    Row r0 fixed at 5, activity 5.25, dual 3.
    """
    case = dict(
        col_lower=np.array([0.0, 0.0, -np.inf, 0.0]),
        col_upper=np.array([10.0, np.inf, np.inf, 1.0]),
        col_value=np.array([10.5, -3e-8, 5.0, 1.0 + 2e-7]),
        row_lower=np.array([5.0]),
        row_upper=np.array([5.0]),
        row_value=np.array([5.25]),
        col_continuous=np.array([True, False, True, True]),
        user_bound_scale=-2,
        user_objective_scale=1,
        primal_feasibility_tolerance=1e-7,
        dual_feasibility_tolerance=1e-7,
        col_dual=np.array([0.5, -0.25, 1e-8, 0.0]),
        row_dual=np.array([3.0]),
        sense=1,
    )
    case.update(overrides)
    return case


def test_user_scaled_primal_counters_by_hand() -> None:
    """``user_bound_scale = -2`` multiplies continuous columns and rows by
    1/4.  c0: 10.5 vs 10 → 2.625 vs 2.5, residual 0.125 (counted).  c1 is
    integer, so unscaled: -3e-8 is within 1e-7 of 0, residual 3e-8 (not
    counted).  c2 is free.  c3: residual 2e-7 unscaled (would count) →
    5e-8 scaled (does not).  r0: 5.25 vs 5 → 1.3125 vs 1.25, residual
    0.0625 (counted).  So num 2, max 0.125."""
    num_p, max_p, _num_d, _max_d = fp.engine.user_scaled_kkt_failures(**_user_scaled_case())
    assert num_p == 2
    assert max_p == 0.125


def test_user_scaled_dual_counters_by_hand() -> None:
    """``user_objective_scale = 1`` doubles the duals; minimising.
    c0 (2.625, above the midpoint 1.25 of [0, 2.5]) needs dual <= 0:
    1.0 → infeasibility 1.0.  c1 ([0, inf), below the infinite midpoint)
    needs dual >= 0: -0.5 → 0.5.  c2 free: |2e-8| (not counted).  c3 dual
    0.  r0 is fixed → 0.  So num 2, max 1.0."""
    _num_p, _max_p, num_d, max_d = fp.engine.user_scaled_kkt_failures(**_user_scaled_case())
    assert num_d == 2
    assert max_d == 1.0


def test_user_scaled_dual_counters_maximising() -> None:
    """Maximising flips the dual signs: c0 -1.0 and c1 0.5 now satisfy
    their conditions; only the free c2 has |2e-8| left: num 0, max 2e-8."""
    _num_p, _max_p, num_d, max_d = fp.engine.user_scaled_kkt_failures(**_user_scaled_case(sense=-1))
    assert num_d == 0
    assert max_d == 2e-8


def test_user_scaled_without_dual_solution() -> None:
    num_p, max_p, num_d, max_d = fp.engine.user_scaled_kkt_failures(
        **_user_scaled_case(col_dual=None, row_dual=None)
    )
    assert (num_p, max_p) == (2, 0.125)
    assert num_d is None and max_d is None


# An LP with equality rows that a truncated interior-point solve leaves
# primal and dual infeasible: random integer A (8 x 20), b = A @ x* for an
# interior x*, 0 <= x <= 10.
_RNG = np.random.default_rng(1)
_A = _RNG.integers(1, 9, (8, 20)).astype(float)
_RHS = _A @ _RNG.uniform(1, 5, 20)
_COST = _RNG.uniform(1, 9, 20)
_TRUNCATED_IPM = {
    "solver": "ipm",
    "ipm_iteration_limit": 1,
    "presolve": "off",
    "run_crossover": "off",
}


def _equality_lp(bound_factor: float, cost_factor: float, solver_options: dict) -> fp.Problem:
    p = fp.Problem()
    cols = pl.DataFrame({"j": list(range(20))})
    rows = pl.DataFrame({"k": list(range(8))})
    x = p.add_var("x", "j", cols, lower=0.0, upper=10.0 * bound_factor)
    a = fp.Param(
        ("k", "j"),
        pl.DataFrame(
            {
                "k": np.repeat(np.arange(8), 20).tolist(),
                "j": np.tile(np.arange(20), 8).tolist(),
                "value": _A.ravel().tolist(),
            }
        ),
    )
    b = fp.Param(
        ("k",), pl.DataFrame({"k": list(range(8)), "value": (_RHS * bound_factor).tolist()})
    )
    p.add_cstr(
        "bal", over=rows, sense="==", lhs_terms={"ax": fp.Sum(a * x, over="j")}, rhs_terms={"b": b}
    )
    c = fp.Param(
        ("j",), pl.DataFrame({"j": list(range(20)), "value": (_COST * cost_factor).tolist()})
    )
    p.set_objective(fp.Sum(c * x, over="j"), sense="min")
    p.set_solver_options({"output_flag": False, **solver_options})
    return p


def test_user_scaled_counters_match_highs_on_the_prescaled_model() -> None:
    """With ``user_bound_scale = -4`` / ``user_objective_scale = 3`` HiGHS
    solves the LP whose bounds are 1/16 and costs 8 times the given ones.
    Solving that LP directly, without user scaling, gives HiGHS' own
    counters in that space: the user-scaled counters must equal them
    exactly, while the plain (unscaled) counters differ."""
    scaled = _equality_lp(
        1.0, 1.0, {**_TRUNCATED_IPM, "user_bound_scale": -4, "user_objective_scale": 3}
    )
    diag = scaled.solve(keep_solver=True).solve_diagnostics()
    oracle = (
        _equality_lp(2.0**-4, 2.0**3, _TRUNCATED_IPM).solve(keep_solver=True).solve_diagnostics()
    )
    assert diag is not None and oracle is not None
    assert (diag.user_bound_scale, diag.user_objective_scale) == (-4, 3)
    assert diag.user_scaled_num_primal_infeasibilities == oracle.num_primal_infeasibilities
    assert diag.user_scaled_max_primal_infeasibility == oracle.max_primal_infeasibility
    assert diag.user_scaled_num_dual_infeasibilities == oracle.num_dual_infeasibilities
    assert diag.user_scaled_max_dual_infeasibility == oracle.max_dual_infeasibility
    # The truncated solve is primal infeasible, and the unscaled residual
    # reads 2**4 times the scaled one.
    assert oracle.max_primal_infeasibility > 0.0
    assert diag.max_primal_infeasibility == 16.0 * diag.user_scaled_max_primal_infeasibility


def test_user_scaled_counters_equal_plain_ones_without_user_scaling() -> None:
    diag = _equality_lp(1.0, 1.0, _TRUNCATED_IPM).solve(keep_solver=True).solve_diagnostics()
    assert diag is not None
    assert (diag.user_bound_scale, diag.user_objective_scale) == (0, 0)
    assert diag.user_scaled_num_primal_infeasibilities == diag.num_primal_infeasibilities
    assert diag.user_scaled_max_primal_infeasibility == diag.max_primal_infeasibility
    assert diag.user_scaled_num_dual_infeasibilities == diag.num_dual_infeasibilities
    assert diag.user_scaled_max_dual_infeasibility == diag.max_dual_infeasibility


def test_user_scaled_counters_not_recorded_for_an_optimal_user_scaled_solve() -> None:
    diag = (
        _equality_lp(1.0, 1.0, {"user_bound_scale": -4}).solve(keep_solver=True).solve_diagnostics()
    )
    assert diag is not None and diag.model_status_name == "kOptimal"
    assert diag.user_bound_scale == -4
    assert diag.user_scaled_num_primal_infeasibilities is None
    assert diag.user_scaled_max_primal_infeasibility is None
    assert diag.user_scaled_num_dual_infeasibilities is None
    assert diag.user_scaled_max_dual_infeasibility is None


def test_mip_user_scaled_rebuild_takes_integrality_from_the_variables(monkeypatch) -> None:
    """A MIP stopped at its first incumbent (kSolutionLimit) with
    ``user_bound_scale = -2`` rebuilds the user-scaled counters.  The
    continuous mask comes from the variable families: the 400 knapsack
    columns are integer, the appended column ``s`` is continuous; HiGHS'
    LP is never copied for it."""
    import highspy

    p = _knapsack({"mip_max_improving_sols": 1, "user_bound_scale": -2, "presolve": "off"})
    one = pl.DataFrame({"m": [0]})
    s = p.add_var("s", "m", one, lower=0.0, upper=3.0)
    p.add_cstr("s_cap", over=None, sense="<=", lhs_terms={"s": fp.Sum(s)}, rhs_terms={"r": 2.0})
    sol = p.solve(keep_solver=True)
    assert sol.highs.getModelStatus() == highspy.HighsModelStatus.kSolutionLimit

    seen = {}
    real = fp.engine.user_scaled_kkt_failures

    def spy(**kw):
        seen["mask"] = kw["col_continuous"].copy()
        return real(**kw)

    def no_copy(*a, **k):
        raise AssertionError("getLp() copies the whole LP")

    monkeypatch.setattr(fp.engine, "user_scaled_kkt_failures", spy)
    monkeypatch.setattr(type(sol.highs), "getLp", no_copy)
    diag = sol.solve_diagnostics()
    assert diag.user_scaled_num_primal_infeasibilities is not None
    expected = np.zeros(401, dtype=bool)
    expected[s.frame["col_id"].to_numpy()] = True
    assert np.array_equal(seen["mask"], expected)


def test_solve_diagnostics_is_cached_on_the_solution(monkeypatch) -> None:
    sol = _build_problem({}).solve(keep_solver=True)
    first = sol.solve_diagnostics()

    def no_read(*a, **k):
        raise AssertionError("read HiGHS again")

    monkeypatch.setattr(type(sol.highs), "getInfo", no_read)
    assert sol.solve_diagnostics() is first
