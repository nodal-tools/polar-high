"""Multi-solver dispatch for polar-high.

Phase 1 scaffold: this module exposes the registry of installed solver Python
wrappers (``available_solvers``) and a :func:`solve` dispatch entry point.
Every solver branch currently raises :class:`NotImplementedError` — real
adapters are wired in later phases of
``specs/polar-high-multi-solver-implementation-plan.md``.

The bring-your-own-license stance from the handoff doc means: the registry
detects which wrappers are *installed*, not which ones are *usable*. License
checks happen later when an env or model is constructed inside an adapter.
"""

from __future__ import annotations

from ._base import (
    IOMode,
    LicenseError,
    SolverError,
    SolverNotAvailableError,
    SolverResult,
    SolverStatus,
)

# Lazy try/except imports at module load. Critical: the import succeeds
# without a license — license checks only fire when we construct an env or a
# Model. So this list tells us "wrapper is installed," not "wrapper is
# usable."
available_solvers: list[str] = []

for _name, _import in [
    ("gurobi", "gurobipy"),
    ("cplex", "cplex"),
    ("xpress", "xpress"),
    ("copt", "coptpy"),
    ("highs", "highspy"),
]:
    try:
        __import__(_import)
        available_solvers.append(_name)
    except ImportError:
        pass


def solve(
    model,
    solver_name: str | None = None,
    io_api: IOMode = IOMode.DIRECT,
    env=None,
    **solver_options,
) -> SolverResult:
    """Dispatch ``model`` to the chosen solver.

    Parameters
    ----------
    model
        A polar-high model object. The concrete type accepted by each adapter
        will be defined when adapters are wired (Phase 2+).
    solver_name
        Which solver to use. Defaults to the first entry in
        :data:`available_solvers`.
    io_api
        :data:`IOMode.DIRECT` (Python API) or :data:`IOMode.MPS` /
        :data:`IOMode.LP` (file-based fallback).
    env
        Optional pre-constructed solver env (e.g. ``gurobipy.Env`` with WLS
        creds). Pass-through only — polar-high does not inspect or construct
        this.
    **solver_options
        Forwarded as-is to the underlying solver.

    Returns
    -------
    SolverResult
        Normalised result from the solver.

    Raises
    ------
    SolverNotAvailableError
        If no solver wrappers are installed, or if ``solver_name`` is not in
        :data:`available_solvers`.
    NotImplementedError
        In Phase 1 every adapter branch raises this. Later phases replace
        the branches with real adapter calls.
    """
    # Re-read the module-level list at call time so monkey-patching in tests
    # (and runtime mutation by callers) is honoured.
    from . import available_solvers as _available

    if not _available:
        raise SolverNotAvailableError(
            "No solver Python wrapper found. Install at least one of: "
            "highspy, gurobipy, cplex, xpress, coptpy."
        )
    if solver_name is None:
        solver_name = _available[0]
    if solver_name not in _available:
        raise SolverNotAvailableError(
            f"Solver '{solver_name}' not available. Installed: {_available}"
        )

    if io_api in (IOMode.MPS, IOMode.LP):
        raise NotImplementedError(
            f"File-based dispatch (io_api={io_api.value!r}) is not yet "
            "implemented; wired in Phase 4 of the multi-solver plan."
        )

    if solver_name == "gurobi":
        raise NotImplementedError("Gurobi adapter is not yet implemented (Phase 5).")
    elif solver_name == "cplex":
        raise NotImplementedError("CPLEX adapter is not yet implemented (Phase 7).")
    elif solver_name == "xpress":
        raise NotImplementedError("Xpress adapter is not yet implemented (Phase 8).")
    elif solver_name == "copt":
        raise NotImplementedError("COPT adapter is not yet implemented (Phase 6).")
    elif solver_name == "highs":
        raise NotImplementedError(
            "HiGHS adapter is not yet wired through solvers.solve (Phase 2). "
            "Use polar_high.Problem.solve() for now."
        )
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


__all__ = [
    "solve",
    "available_solvers",
    "IOMode",
    "SolverStatus",
    "SolverResult",
    "SolverNotAvailableError",
    "LicenseError",
    "SolverError",
]
