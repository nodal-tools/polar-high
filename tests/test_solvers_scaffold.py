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


def test_default_solver_is_first_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``solver_name=None``, dispatch picks ``available_solvers[0]``.

    The HiGHS branch is wired in Phase 2, so to keep this test independent
    of which solvers are installed we monkey-patch ``available_solvers``
    with a non-HiGHS entry and confirm dispatch routes into its branch by
    inspecting the (still-unwired) ``NotImplementedError``.
    """
    # gurobi (Phase 5) and copt (Phase 6) are now wired and raise
    # SolverNotAvailableError when their Python wrapper is not installed
    # (rather than NotImplementedError).  The remaining direct branches
    # still raise NotImplementedError until Phases 7-8.
    branch_keywords = {
        "cplex": "CPLEX",
        "xpress": "Xpress",
    }
    for name, kw in branch_keywords.items():
        monkeypatch.setattr(solvers, "available_solvers", [name])
        with pytest.raises(NotImplementedError) as excinfo:
            solve(object(), io_api=IOMode.DIRECT)
        assert kw in str(excinfo.value), (
            f"dispatch routed {name!r} to the wrong branch: {excinfo.value}"
        )
