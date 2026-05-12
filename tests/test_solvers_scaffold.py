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

    After Phase 8 every direct branch (gurobi, cplex, xpress, copt,
    highs) is wired to a real adapter — there is no longer a
    ``NotImplementedError`` branch left to assert against.  We
    therefore confirm the *registry-default* behaviour: monkey-patching
    ``available_solvers`` to a single-entry list ``[name]`` makes
    ``solve(..., solver_name=None)`` route to ``name``.  We use the
    HiGHS branch as the routing target (always installed in CI) and
    just verify the call returns a HiGHS-tagged result.
    """
    # We can only meaningfully exercise the "first-available" routing
    # when that first solver actually has its Python wrapper installed.
    # HiGHS is the safe choice — always installed in CI.
    monkeypatch.setattr(solvers, "available_solvers", ["highs"])
    from toy_data import make_toy_data
    from toy_model import build_dispatch

    from polar_high import Problem

    pb = Problem()
    build_dispatch(pb, make_toy_data())  # type: ignore[no-untyped-call]
    result = solve(pb, io_api=IOMode.DIRECT)
    assert result.solver_name == "highs"
