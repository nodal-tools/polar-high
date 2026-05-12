"""Gurobi direct adapter for ``polar_high.solvers.solve``.

Phase 5 of ``specs/polar-high-multi-solver-implementation-plan.md``.

This adapter consumes a fully-extracted
:class:`~polar_high.solvers._lp_view.LpView` and pushes it into a fresh
``gurobipy.Model`` in memory.  Vectorized load via
``scipy.sparse.csc_matrix`` + ``Model.addMVar`` + ``Model.addMConstr``,
following the linopy reference pattern and handoff Step 5.

License model — bring-your-own
------------------------------
``polar_high`` never inspects, constructs, or validates a Gurobi
license.  The caller may pass a pre-constructed ``gurobipy.Env`` via the
``env=`` kwarg (e.g. configured with WLS credentials or a Cluster
Manager endpoint).  When ``env is None`` the default ``gp.Model()``
constructor is used, which picks up ``GRB_LICENSE_FILE`` / ``~/gurobi.lic``
via Gurobi's own discovery.

Any :class:`gurobipy.GurobiError` is caught and re-raised as either
:class:`~polar_high.solvers._base.LicenseError` (when ``errno`` is in
the documented license range ``10009..10015``) or
:class:`~polar_high.solvers._base.SolverError` (everything else).  The
raw vendor exception never reaches the caller.

Dependency note
---------------
``gurobipy`` and ``scipy`` are *optional* — they are pulled in by the
``polar-high[gurobi]`` extra.  Both imports happen inside :func:`run`,
not at module load, so this module is always importable.  A missing
wrapper raises :class:`~polar_high.solvers._base.SolverNotAvailableError`
with an install pointer.

Out of scope (matching the plan)
--------------------------------
- Callbacks, lazy constraints, MIP starts, multi-objective
- Quadratic / SOCP / nonlinear models
- Solver-switch warm starts (every call is a fresh ``gp.Model``)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._base import (
    LicenseError,
    SolverError,
    SolverNotAvailableError,
    SolverResult,
    SolverStatus,
)
from ._lp_view import LpView

# Gurobi license-error codes.  Documented values in the
# ``GRB.ERROR_*`` family that signal "the solver imported fine but the
# license check failed."  See handoff Step 4.
_LICENSE_ERROR_CODES = frozenset({10009, 10010, 10011, 10012, 10013, 10014, 10015})

_INSTALL_HINT = (
    "Install the optional extra:  pip install 'polar-high[gurobi]'  "
    "(this pulls gurobipy and scipy together)."
)


def run(
    view: LpView,
    *,
    env: Any = None,
    **options: Any,
) -> SolverResult:
    """Solve ``view`` with Gurobi via the in-memory direct API.

    Parameters
    ----------
    view
        A fully-extracted :class:`LpView`.  Built once by
        :func:`polar_high.solvers.solve`.
    env
        Optional pre-constructed ``gurobipy.Env``.  Pass-through only —
        ``polar_high`` does not inspect or construct it.  When ``None``,
        ``gurobipy`` discovers the license via its usual ``gurobi.lic``
        / ``GRB_LICENSE_FILE`` lookup.
    **options
        Forwarded one-by-one to ``Model.setParam`` (e.g. ``TimeLimit=60``,
        ``MIPGap=0.01``).  Unknown keys raise ``GurobiError`` which is
        wrapped into :class:`SolverError`.

    Returns
    -------
    SolverResult
        ``status`` is mapped from ``Model.status``.  ``primal`` is keyed
        by ``VarName`` whenever a solution is available
        (``SolCount > 0``).  ``dual`` is populated for LP solves only
        (``IsMIP == 0`` and ``SolCount > 0``); ``None`` otherwise.
        ``raw_status`` carries the integer Gurobi status code for
        debugging.

    Raises
    ------
    SolverNotAvailableError
        ``gurobipy`` or ``scipy`` is not importable.
    LicenseError
        ``GurobiError`` with ``errno`` in the license range.
    SolverError
        Any other ``GurobiError`` from model construction or solve.
    """
    # ------------------------------------------------------------------
    # Lazy imports — keep the module importable without [gurobi] extra.
    # ------------------------------------------------------------------
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:  # pragma: no cover — covered manually
        raise SolverNotAvailableError("gurobipy is not installed.  " + _INSTALL_HINT) from exc

    try:
        import scipy.sparse as sp
    except ImportError as exc:  # pragma: no cover — covered manually
        raise SolverNotAvailableError(
            "scipy is not installed (required by the Gurobi adapter for "
            "vectorized matrix load).  " + _INSTALL_HINT
        ) from exc

    n_cols = int(view.n_cols)

    try:
        # --------------------------------------------------------------
        # Model construction.  We pass-through ``env`` verbatim; we do
        # NOT inspect, construct, or validate it.  When ``env is None``
        # gurobipy uses its default env (license discovery is gurobipy's
        # responsibility, not ours).
        # --------------------------------------------------------------
        m = gp.Model(env=env) if env is not None else gp.Model()

        # --------------------------------------------------------------
        # Variable types.
        #
        # LpView.integrality follows the HiGHS convention used in
        # `_lp_view.from_problem`:
        #   None  -> pure LP, every column continuous
        #   array -> int8, 1 = integer column, 0 = continuous
        #
        # polar-high's engine has no notion of a "binary" column today
        # (an integer var with bounds [0, 1] stays typed integer).  We
        # therefore map int8 1 -> GRB.INTEGER; binary detection is left
        # to Gurobi's own presolve via the [0, 1] bounds.  Adding a
        # GRB.BINARY shortcut is a no-op for the solver and would only
        # complicate the round-trip with HiGHS.
        # --------------------------------------------------------------
        if view.integrality is None:
            vtype: Any = GRB.CONTINUOUS
        else:
            vtype = np.where(view.integrality.astype(bool), GRB.INTEGER, GRB.CONTINUOUS)

        x = m.addMVar(
            n_cols,
            lb=view.col_lb,
            ub=view.col_ub,
            obj=view.col_obj,
            vtype=vtype,
        )

        # --------------------------------------------------------------
        # Variable names.  ``MVar.VarName`` is settable per-element via
        # the underlying ``Var`` objects.  Gurobi tolerates duplicate /
        # empty names but they hurt LP-file readability; emit only the
        # names polar-high actually populated.
        # --------------------------------------------------------------
        for i, nm in enumerate(view.col_names):
            if nm is not None:
                x[i].VarName = nm

        # --------------------------------------------------------------
        # Constraint matrix and RHS.
        #
        # We feed Gurobi a CSC ``scipy.sparse`` matrix built directly
        # from the view's arrays — vectorized through the C API, fast
        # enough that solver runtime dominates even at ~10^6 rows.
        #
        # Ranged-row handling: rather than call ``Model.addRange`` per
        # ranged row (which would need a Python-side loop over the
        # ranged subset and a fresh expression each iteration), we use
        # ``LpView.split_ranged_rows()`` to expand each ranged row into
        # a >=/<= pair *upstream* of the Gurobi load.  Trade-off:
        #
        #   + One vectorized ``addMConstr`` call regardless of range
        #     count; no Python-level per-row loop.
        #   + Constraint duals stay accessible by name (the lo/hi halves
        #     each get a name suffix from ``split_ranged_rows``).
        #   - Each ranged row's nonzeros are duplicated in the matrix
        #     (memory cost is bounded by 2x nnz in the ranged subset).
        #
        # Models with no ranged rows pay zero overhead — the splitter
        # short-circuits to a shallow copy.
        # --------------------------------------------------------------
        load_view = view.split_ranged_rows()
        load_n_rows = int(load_view.n_rows)

        a = sp.csc_matrix(
            (load_view.a_value, load_view.a_index, load_view.a_start),
            shape=(load_n_rows, n_cols),
        )

        senses_arr, rhs_arr, _range_arr = load_view.row_sense_rhs()
        # row_sense_rhs after split returns only "E"/"L"/"G".  Map them
        # to Gurobi's char constants for addMConstr.
        sense_map = {"E": GRB.EQUAL, "L": GRB.LESS_EQUAL, "G": GRB.GREATER_EQUAL}
        sense_chars = np.array([sense_map[s] for s in senses_arr.tolist()], dtype=object)

        if load_n_rows > 0:
            constrs = m.addMConstr(a, x, sense_chars, rhs_arr)
            # addMConstr returns an MConstr; underlying Constr objects
            # carry the per-row name.
            for i, nm in enumerate(load_view.row_names):
                if nm:
                    constrs[i].ConstrName = nm

        # --------------------------------------------------------------
        # Objective.  ``addMVar(..., obj=...)`` already set the linear
        # coefficients; we just need the sense and any offset.
        # --------------------------------------------------------------
        m.ModelSense = GRB.MAXIMIZE if view.sense == "max" else GRB.MINIMIZE
        if view.obj_offset:
            m.ObjCon = float(view.obj_offset)

        # --------------------------------------------------------------
        # Per-call options.  Pass-through; unknown keys surface as
        # ``GurobiError`` and are wrapped below.
        # --------------------------------------------------------------
        for k, v in options.items():
            m.setParam(k, v)

        m.optimize()

        # --------------------------------------------------------------
        # Result assembly.
        # --------------------------------------------------------------
        status_map = {
            GRB.OPTIMAL: SolverStatus.OPTIMAL,
            GRB.INFEASIBLE: SolverStatus.INFEASIBLE,
            GRB.INF_OR_UNBD: SolverStatus.UNBOUNDED,
            GRB.UNBOUNDED: SolverStatus.UNBOUNDED,
            GRB.TIME_LIMIT: SolverStatus.TIME_LIMIT,
            GRB.INTERRUPTED: SolverStatus.INTERRUPTED,
        }
        status = status_map.get(m.Status, SolverStatus.OTHER)

        has_solution = m.SolCount > 0
        is_mip = m.IsMIP == 1

        primal: dict[str, float] | None
        dual: dict[str, float] | None
        objective: float | None

        if has_solution:
            primal = {v.VarName: float(v.X) for v in m.getVars() if v.VarName}
            objective = float(m.ObjVal)
            if is_mip:
                dual = None
            else:
                # Constraint names may be empty strings on the lo/hi
                # halves if the original row was anonymous; preserve
                # whatever Gurobi returned.
                dual = {c.ConstrName: float(c.Pi) for c in m.getConstrs() if c.ConstrName}
        else:
            primal = None
            dual = None
            objective = None

        return SolverResult(
            status=status,
            objective=objective,
            primal=primal,
            dual=dual,
            solver_name="gurobi",
            raw_status=m.Status,
        )

    except gp.GurobiError as exc:
        errno = getattr(exc, "errno", None)
        if errno in _LICENSE_ERROR_CODES:
            raise LicenseError(
                f"Gurobi license check failed (code {errno}): {exc}.  "
                "Place gurobi.lic at $HOME or /opt/gurobi/, set "
                "GRB_LICENSE_FILE, or pass a configured gurobipy.Env "
                "via env=..."
            ) from exc
        raise SolverError(f"Gurobi error (code {errno}): {exc}") from exc


__all__ = ["run"]
