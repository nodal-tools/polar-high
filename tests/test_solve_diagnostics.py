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
