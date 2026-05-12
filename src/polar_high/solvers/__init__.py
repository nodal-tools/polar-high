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
    from ._mps_fallback import _BINARY_NAMES as _MPS_SOLVERS

    # The MPS-file path does not need the solver's Python wrapper — only
    # its CLI binary.  So if the caller asked for io_api=MPS and the
    # target is one of the four commercial solvers, we let it through
    # the registry check.  The dispatch into ``_mps_fallback`` will then
    # surface a clean ``SolverError`` if the binary is also missing.
    _mps_eligible = io_api == IOMode.MPS and solver_name is not None and solver_name in _MPS_SOLVERS

    if not _available and not _mps_eligible:
        raise SolverNotAvailableError(
            "No solver Python wrapper found. Install at least one of: "
            "highspy, gurobipy, cplex, xpress, coptpy."
        )
    if solver_name is None:
        solver_name = _available[0]
    if solver_name not in _available and not _mps_eligible:
        # Surface the optional-extra install hint when the user asked for
        # a solver whose adapter we ship but whose Python wrapper is not
        # installed.  This keeps the error UX consistent with the
        # adapter-internal SolverNotAvailableError that would otherwise
        # fire one frame deeper.
        _extra_hint = {
            "gurobi": "  Install via:  pip install 'polar-high[gurobi]'",
            "cplex": "  Install via:  pip install 'polar-high[cplex]'",
            "xpress": "",
            "copt": "  Install via:  pip install 'polar-high[copt]'",
        }.get(solver_name, "")
        raise SolverNotAvailableError(
            f"Solver '{solver_name}' not available. Installed: {_available}." + _extra_hint
        )

    if io_api == IOMode.LP:
        # The LP file format is reserved by the enum but the Phase 4
        # plan ("Out of scope") leaves implementation to a follow-up.
        raise NotImplementedError(
            "io_api='lp' is reserved but not implemented. Use io_api='mps' "
            "for the file-based fallback, or io_api='direct' for the "
            "in-memory adapter."
        )

    if io_api == IOMode.MPS:
        # HiGHS: refuse loudly rather than silently round-trip through a
        # temp MPS file (would lose names and incur a write/read cost
        # for no benefit — the in-memory HiGHS path is always strictly
        # better).  See ``_mps_fallback.run_via_file``'s docstring.
        if solver_name == "highs":
            raise ValueError(
                "io_api='mps' is not supported for solver_name='highs'. "
                "HiGHS always uses the in-memory direct path; use "
                "io_api=IOMode.DIRECT (the default) for HiGHS."
            )
        from ._lp_view import LpView
        from ._mps_fallback import run_via_file

        view = LpView.from_problem(model)
        return run_via_file(view, solver_name, io_api, **solver_options)

    if solver_name == "gurobi":
        from ._gurobi import run as _gurobi_run
        from ._lp_view import LpView

        view = LpView.from_problem(model)
        return _gurobi_run(view, env=env, **solver_options)
    elif solver_name == "cplex":
        from ._cplex import run as _cplex_run
        from ._lp_view import LpView

        view = LpView.from_problem(model)
        return _cplex_run(view, env=env, **solver_options)
    elif solver_name == "xpress":
        raise NotImplementedError("Xpress adapter is not yet implemented (Phase 8).")
    elif solver_name == "copt":
        from ._copt import run as _copt_run
        from ._lp_view import LpView

        view = LpView.from_problem(model)
        return _copt_run(view, env=env, **solver_options)
    elif solver_name == "highs":
        from ._highs import run
        from ._lp_view import LpView

        # Build the LP view once; ``_highs.run`` consumes it directly
        # without reaching back into engine internals.  The
        # ``set_solver_options`` precedence is preserved here so callers
        # of ``solvers.solve`` see the same option resolution as
        # ``Problem.solve``: per-call options win over what was stored
        # on the Problem via ``set_solver_options``.
        view = LpView.from_problem(model)
        opts = solver_options.pop("options", None)
        if opts is None:
            opts = getattr(model, "_solver_options", None)
        keep_solver = solver_options.pop("keep_solver", False)
        return run(view, env=env, options=opts, keep_solver=keep_solver, **solver_options)
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
