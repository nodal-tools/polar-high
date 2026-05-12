"""Phase 1 scaffold tests for ``polar_high.solvers``.

These verify the registry + dispatch shell only — no real solving happens
here. Real adapter behaviour is covered by later phases.
"""

from __future__ import annotations

import pytest

from polar_high import solvers
from polar_high.solvers import (
    IOMode,
    SolverNotAvailableError,
    available_solvers,
    solve,
)


def test_available_solvers_lists_highs() -> None:
    """CI always installs highspy, so 'highs' must appear in the registry."""
    assert "highs" in available_solvers


def test_unknown_solver_raises_SolverNotAvailableError() -> None:
    with pytest.raises(SolverNotAvailableError) as excinfo:
        solve(object(), solver_name="not_a_real_solver")
    msg = str(excinfo.value)
    assert "not_a_real_solver" in msg
    assert "Installed" in msg


def test_empty_registry_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no solvers installed, dispatch must raise the install-help error."""
    monkeypatch.setattr(solvers, "available_solvers", [])
    with pytest.raises(SolverNotAvailableError) as excinfo:
        solve(object())
    msg = str(excinfo.value)
    assert "No solver Python wrapper found" in msg
    assert "highspy" in msg


def test_default_solver_is_first_available() -> None:
    """With ``solver_name=None``, dispatch picks ``available_solvers[0]``.

    In Phase 1 every branch raises ``NotImplementedError``; the adapter name
    is embedded in the error message so we can confirm which branch ran.
    """
    assert available_solvers, "test precondition: at least one solver installed"
    expected = available_solvers[0]
    branch_keywords = {
        "gurobi": "Gurobi",
        "cplex": "CPLEX",
        "xpress": "Xpress",
        "copt": "COPT",
        "highs": "HiGHS",
    }
    with pytest.raises(NotImplementedError) as excinfo:
        solve(object(), io_api=IOMode.DIRECT)
    assert branch_keywords[expected] in str(excinfo.value)
