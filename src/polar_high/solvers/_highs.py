"""HiGHS adapter for ``polar_high.solvers.solve``.

This is the Phase 2 extraction of the non-streaming (``passModel``) HiGHS
solve path out of :meth:`polar_high.engine.Problem.solve`. The function
:func:`run` takes a fully-populated :class:`~polar_high.engine.Problem`,
builds the LP arrays via :meth:`Problem._build_lp_arrays`, hands them to a
fresh ``highspy.Highs`` instance via ``passModel``, runs the solver, and
returns a :class:`~polar_high.solvers._base.SolverResult`.

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
from typing import TYPE_CHECKING, Any

import highspy
import numpy as np
import polars as pl

from ._base import SolverResult, SolverStatus

if TYPE_CHECKING:
    from ..engine import Problem


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
    problem: Problem,
    *,
    env: Any = None,
    options: dict | None = None,
    keep_solver: bool = False,
    **kwargs: Any,
) -> SolverResult:
    """Solve ``problem`` with HiGHS via the non-streaming ``passModel`` path.

    Parameters
    ----------
    problem
        A fully-populated :class:`polar_high.engine.Problem`.
    env
        Accepted for API symmetry with the commercial adapters; HiGHS has
        no concept of a pre-built env, so a non-None value is silently
        ignored (no license to validate).
    options
        Per-call HiGHS options dict; overrides whatever was set via
        ``Problem.set_solver_options``. Same semantics as the existing
        ``Problem.solve(options=...)`` kwarg.
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

    # ------------------------------------------------------------------
    # Column extraction — copied verbatim from the pre-Phase-2
    # ``Problem.solve`` non-streaming branch so behaviour is bit-identical.
    # ------------------------------------------------------------------
    n_cols = problem._next_col
    col_lb = np.zeros(n_cols, dtype=np.float64)
    col_ub = np.full(n_cols, np.inf, dtype=np.float64)
    col_obj = np.zeros(n_cols, dtype=np.float64)
    col_int = np.zeros(n_cols, dtype=np.int8)  # 1 = integer column
    col_names: list[str] = [None] * n_cols  # type: ignore[list-item]

    for v in problem._vars.values():
        ids = v.frame["col_id"].to_numpy()
        col_lb[ids] = float(v.lower)
        col_ub[ids] = float(v.upper)
        if v.integer:
            col_int[ids] = 1
        if v.dims:
            tagged = (
                v.frame.select(
                    pl.format(
                        "{}[{}]",
                        pl.lit(v.name),
                        pl.concat_str([pl.col(d).cast(pl.String) for d in v.dims], separator=","),
                    ).alias("__name")
                )
            )["__name"].to_list()
            ids_list = ids.tolist()
            for cid, nm in zip(ids_list, tagged):
                col_names[cid] = nm
        else:
            cid0 = int(ids[0])
            col_names[cid0] = v.name

    # Objective scatter — see comment on the original code path.
    for t in problem._obj_terms:
        f = t.lazy.collect()
        np.add.at(col_obj, f["col_id"].to_numpy(), f["coef"].to_numpy())
        del f

    # ------------------------------------------------------------------
    # LP-array build via the shared helper on Problem.
    # ------------------------------------------------------------------
    (
        col_lb_h,
        col_ub_h,
        row_lb_h,
        row_ub_h,
        sorted_v,
        sorted_r,
        starts,
        row_names,
        n_rows,
    ) = problem._build_lp_arrays(
        n_cols=n_cols,
        col_lb=col_lb,
        col_ub=col_ub,
    )

    lp = highspy.HighsLp()
    lp.num_col_ = int(n_cols)
    lp.num_row_ = int(n_rows)
    lp.col_cost_ = col_obj.astype(np.float64)
    lp.col_lower_ = col_lb_h
    lp.col_upper_ = col_ub_h
    lp.row_lower_ = row_lb_h
    lp.row_upper_ = row_ub_h
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_ = int(n_cols)
    lp.a_matrix_.num_row_ = int(n_rows)
    lp.a_matrix_.start_ = starts
    lp.a_matrix_.index_ = sorted_r
    lp.a_matrix_.value_ = sorted_v
    lp.sense_ = (
        highspy.ObjSense.kMaximize if problem._obj_sense == "max" else highspy.ObjSense.kMinimize
    )
    if problem._obj_offset:
        lp.offset_ = float(problem._obj_offset)
    if col_int.any():
        kCont = highspy.HighsVarType.kContinuous
        kInt = highspy.HighsVarType.kInteger
        integ_arr = np.where(col_int, kInt, kCont)
        lp.integrality_ = integ_arr.tolist()

    h = highspy.Highs()

    # Apply solver options BEFORE passModel — some HiGHS options
    # (notably ``presolve``) must be set before the model is loaded to
    # take effect on the first ``run()``. Per-call ``options`` wins over
    # whatever was stored on the Problem.
    opts = options if options is not None else problem._solver_options
    if opts:
        ok_status = getattr(highspy.HighsStatus, "kOk", None)
        for key, val in opts.items():
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
    primal: dict[str, float] | None
    dual: dict[str, float] | None
    objective: float | None
    if status == SolverStatus.OPTIMAL:
        primal = {nm: float(col_value[i]) for i, nm in enumerate(col_names) if nm is not None}
        # Only LP solves carry meaningful duals; for a MIP HiGHS may
        # still return zeros, but exposing them would be misleading.
        if col_int.any():
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
