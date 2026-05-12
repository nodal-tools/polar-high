"""HiGHS adapter for ``polar_high.solvers.solve``.

Phase 3 refactor: this adapter now consumes a fully-extracted
:class:`~polar_high.solvers._lp_view.LpView` (see ``_lp_view.py``) and
no longer reaches into :class:`~polar_high.engine.Problem` internals.
The view is built once by the dispatch in ``solvers/__init__.py`` (and
by ``Problem.solve(streaming=False)``) and handed to :func:`run`, which
loads it into a fresh ``highspy.Highs`` via ``passModel`` and solves.

Streaming-path split (locked in pre-implementation insights)
------------------------------------------------------------
``Problem.solve()`` exposes two ways to feed HiGHS:

* ``streaming=True`` (the default) — columns added once via ``addCols`` and
  each constraint family is pushed via ``addRows`` immediately after its
  COO triples are built. Peak memory is bounded by one family. This path
  is **HiGHS-only** and lives entirely inside :mod:`polar_high.engine`. It
  does **not** flow through this module or through
  :func:`polar_high.solvers.solve`.

* ``streaming=False`` — the full LP is assembled into a single
  ``highspy.HighsLp`` and loaded via ``passModel``. This is the path that
  every commercial solver also wants (whole matrix at once), so this is
  the path that lives behind :func:`polar_high.solvers.solve` for every
  backend. :meth:`Problem.solve(streaming=False)` is now a thin wrapper
  that delegates here and converts the returned ``SolverResult`` back
  into the legacy ``Solution`` shape.

``WarmProblem`` continues to drive HiGHS directly (also HiGHS-only) and
does **not** flow through this adapter — incremental warm-starts are out
of scope for the cross-solver dispatch.
"""

from __future__ import annotations

import warnings
from typing import Any

import highspy
import numpy as np

from ._base import SolverResult, SolverStatus
from ._lp_view import LpView


def _map_status(model_status: Any) -> SolverStatus:
    """Translate ``highspy.HighsModelStatus`` to :class:`SolverStatus`."""
    M = highspy.HighsModelStatus
    if model_status == M.kOptimal:
        return SolverStatus.OPTIMAL
    if model_status == M.kInfeasible:
        return SolverStatus.INFEASIBLE
    if model_status == getattr(M, "kUnbounded", object()):
        return SolverStatus.UNBOUNDED
    # HiGHS uses kUnboundedOrInfeasible for the ambiguous case
    if model_status == getattr(M, "kUnboundedOrInfeasible", object()):
        return SolverStatus.UNBOUNDED
    if model_status == getattr(M, "kTimeLimit", object()):
        return SolverStatus.TIME_LIMIT
    if model_status == getattr(M, "kInterrupt", object()):
        return SolverStatus.INTERRUPTED
    return SolverStatus.OTHER


def run(
    view: LpView,
    *,
    env: Any = None,
    options: dict | None = None,
    keep_solver: bool = False,
    **kwargs: Any,
) -> SolverResult:
    """Solve ``view`` with HiGHS via the non-streaming ``passModel`` path.

    Parameters
    ----------
    view
        A fully-extracted :class:`LpView`.  Build via
        :meth:`LpView.from_problem` (the dispatch entry in
        :mod:`polar_high.solvers` does this for you).
    env
        Accepted for API symmetry with the commercial adapters; HiGHS has
        no concept of a pre-built env, so a non-None value is silently
        ignored (no license to validate).
    options
        Per-call HiGHS options dict.  Caller is responsible for resolving
        precedence between per-call and per-``Problem`` options before
        invoking ``run``.
    keep_solver
        When ``True``, the live ``highspy.Highs`` handle is stashed on
        the returned ``SolverResult`` as the private attribute
        ``_highs_instance``. Used by ``Problem.solve(keep_solver=True)``
        to preserve the old ``Solution.highs`` field. Other callers
        should leave this ``False``.
    **kwargs
        Reserved for forward compatibility with the cross-solver
        dispatch; ignored here.

    Returns
    -------
    SolverResult
        Solver-agnostic result. ``primal`` is keyed by column name,
        ``dual`` by row name (LP duals only). When ``keep_solver=True``,
        the returned object also carries private attributes
        ``_highs_instance`` (the live ``Highs``), ``_col_value``,
        ``_row_dual``, ``_col_dual`` (raw numpy arrays), and
        ``_col_names`` / ``_row_names`` for zero-copy round-trip back
        into the legacy :class:`~polar_high.engine.Solution` shape.
    """
    del env, kwargs  # unused — see docstring

    n_cols = int(view.n_cols)
    n_rows = int(view.n_rows)
    col_obj = view.col_obj
    col_names = view.col_names
    row_names = view.row_names

    # ------------------------------------------------------------------
    # HiGHS uses kHighsInf (a finite sentinel) in place of ±inf in its LP
    # arrays.  The view stores ±inf for cross-solver portability, so we
    # convert here.  ``np.where`` is O(n) but n is the column / row
    # count, not nnz, so this is well below the solve cost.
    # ------------------------------------------------------------------
    inf = highspy.kHighsInf
    col_lb_h = np.where(view.col_lb == -np.inf, -inf, view.col_lb).astype(np.float64)
    col_ub_h = np.where(view.col_ub == np.inf, inf, view.col_ub).astype(np.float64)
    row_lb_h = np.where(view.row_lb == -np.inf, -inf, view.row_lb).astype(np.float64)
    row_ub_h = np.where(view.row_ub == np.inf, inf, view.row_ub).astype(np.float64)

    lp = highspy.HighsLp()
    lp.num_col_ = n_cols
    lp.num_row_ = n_rows
    lp.col_cost_ = col_obj.astype(np.float64)
    lp.col_lower_ = col_lb_h
    lp.col_upper_ = col_ub_h
    lp.row_lower_ = row_lb_h
    lp.row_upper_ = row_ub_h
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_ = n_cols
    lp.a_matrix_.num_row_ = n_rows
    lp.a_matrix_.start_ = view.a_start
    lp.a_matrix_.index_ = view.a_index
    lp.a_matrix_.value_ = view.a_value
    lp.sense_ = highspy.ObjSense.kMaximize if view.sense == "max" else highspy.ObjSense.kMinimize
    if view.obj_offset:
        lp.offset_ = float(view.obj_offset)
    if view.integrality is not None:
        kCont = highspy.HighsVarType.kContinuous
        kInt = highspy.HighsVarType.kInteger
        integ_arr = np.where(view.integrality.astype(bool), kInt, kCont)
        lp.integrality_ = integ_arr.tolist()

    h = highspy.Highs()

    # Apply solver options BEFORE passModel — some HiGHS options
    # (notably ``presolve``) must be set before the model is loaded to
    # take effect on the first ``run()``.
    if options:
        ok_status = getattr(highspy.HighsStatus, "kOk", None)
        for key, val in options.items():
            try:
                status = h.setOptionValue(key, val)
            except Exception as exc:  # belt-and-braces — highspy may raise
                warnings.warn(
                    f"HiGHS rejected option {key}={val!r}: {exc}",
                    stacklevel=2,
                )
                continue
            if ok_status is not None and status != ok_status:
                warnings.warn(
                    f"HiGHS rejected option {key}={val!r} (status={status!r})",
                    stacklevel=2,
                )

    h.passModel(lp)

    # names — passColName / passRowName per item (cheap)
    for i, n in enumerate(col_names):
        if n is not None:
            h.passColName(i, n)
    for i, n in enumerate(row_names):
        h.passRowName(i, n)

    h.run()
    sol = h.getSolution()
    model_status = h.getModelStatus()
    status = _map_status(model_status)
    obj_val_raw = h.getObjectiveValue()
    col_value = np.asarray(sol.col_value, dtype=np.float64)
    row_dual = (
        np.asarray(sol.row_dual, dtype=np.float64)
        if sol.row_dual
        else np.zeros(n_rows, dtype=np.float64)
    )
    col_dual = (
        np.asarray(sol.col_dual, dtype=np.float64)
        if sol.col_dual
        else np.zeros(n_cols, dtype=np.float64)
    )

    # Build the cross-solver dict views. Skip None column names (gaps
    # are possible if a Var was registered but its name slot stayed
    # unset — defensive).
    is_mip = view.integrality is not None
    primal: dict[str, float] | None
    dual: dict[str, float] | None
    objective: float | None
    if status == SolverStatus.OPTIMAL:
        primal = {nm: float(col_value[i]) for i, nm in enumerate(col_names) if nm is not None}
        # Only LP solves carry meaningful duals; for a MIP HiGHS may
        # still return zeros, but exposing them would be misleading.
        if is_mip:
            dual = None
        else:
            dual = {nm: float(row_dual[i]) for i, nm in enumerate(row_names)}
        objective = float(obj_val_raw)
    else:
        primal = None
        dual = None
        objective = None

    result = SolverResult(
        status=status,
        objective=objective,
        primal=primal,
        dual=dual,
        solver_name="highs",
        raw_status=model_status,
    )

    # Stash the raw arrays so ``Problem.solve(streaming=False)`` can
    # rebuild the legacy ``Solution`` shape without re-walking the dicts.
    # These are private to the HiGHS adapter; cross-solver callers must
    # rely on the public ``primal`` / ``dual`` fields instead.
    result._col_value = col_value
    result._row_dual = row_dual
    result._col_dual = col_dual
    result._col_names = col_names
    result._row_names = row_names
    # Raw HiGHS objective (carried even for non-optimal solves) so the
    # legacy ``Solution.obj`` round-trip stays bit-identical.
    result._objective_raw = float(obj_val_raw)
    if keep_solver:
        result._highs_instance = h
    else:
        result._highs_instance = None
        del h, lp

    return result


__all__ = ["run"]
