""":meth:`Solution.solve_diagnostics` — normalised HiGHS status snapshot.

The accessor reads ``getModelStatus()`` + ``getInfo()`` off the live handle
into a policy-free :class:`SolveDiagnostics`, and returns ``None`` when no
queryable handle is attached (a synthesised Solution or a read-only shim
without ``getInfo``).  Downstream (flextool) uses this to accept a solve
that HiGHS could not *certify* as optimal but whose primal is feasible.
"""

from __future__ import annotations

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
