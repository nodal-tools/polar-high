"""HiGHS' KKT measures and its tolerance-violation factor, in the space HiGHS solved.

HiGHS 1.15 judges an LP solution before it undoes ``user_bound_scale`` /
``user_objective_scale`` (``Highs.cpp:1990`` vs ``:1123``), then recomputes
its info on the unscaled numbers without residuals
(``HighsInterface.cpp:3161-3200``).  :func:`kkt_report` rebuilds what it
judged.  User scaling multiplies the continuous columns' and the rows' bounds
and values by ``2**user_bound_scale`` and the costs and duals by
``2**user_objective_scale``; the matrix and ``offset_`` stay
(``HighsLpUtils.cpp:781-905``).  Powers of two make the rebuild exact.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import highspy
import numpy as np

from .engine import (
    _HIGHS_DEFAULT_KKT_TOLERANCE,
    _dual_infeasibilities,
    _highs_float_option,
    _highs_int_option,
    _primal_infeasibilities,
)

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KktTolerances:
    """The five tolerances HiGHS' KKT check uses: the options, or all five
    equal to ``kkt_tolerance`` when that is not its default 1e-7
    (``HighsSolution.cpp:83-89``, ``:1055-1062``)."""

    primal_feasibility: float
    dual_feasibility: float
    primal_residual: float
    dual_residual: float
    optimality: float


def kkt_tolerances(h) -> KktTolerances:
    """The effective :class:`KktTolerances` of handle *h*."""
    kkt = _highs_float_option(h, "kkt_tolerance", _HIGHS_DEFAULT_KKT_TOLERANCE)
    if kkt != _HIGHS_DEFAULT_KKT_TOLERANCE:
        return KktTolerances(kkt, kkt, kkt, kkt, kkt)
    return KktTolerances(
        _highs_float_option(h, "primal_feasibility_tolerance", 1e-7),
        _highs_float_option(h, "dual_feasibility_tolerance", 1e-7),
        _highs_float_option(h, "primal_residual_tolerance", 1e-7),
        _highs_float_option(h, "dual_residual_tolerance", 1e-7),
        _highs_float_option(h, "optimality_tolerance", 1e-7),
    )


def _get(values: Any, name: str) -> float:
    return values[name] if isinstance(values, Mapping) else getattr(values, name)


def tolerance_violation(
    values, *, basis_valid: bool, tolerances: KktTolerances
) -> tuple[float, tuple[tuple[str, float], ...]]:
    """HiGHS' factor X of ``lpKktCheck`` (``HighsSolution.cpp:1043-1327``)
    and the terms that enter it, as ``(field name, measure / tolerance)`` in
    HiGHS' order.

    *values* holds the 16 HighsInfo KKT fields and
    ``primal_dual_objective_error`` (a ``HighsInfo``, a :class:`KktReport`
    or a mapping with those names).  With a valid basis X takes the
    absolute infeasibilities (when their count is > 0), the residuals only
    when they were computed (count not ``-1``), and the P-D objective error.
    Without one it takes the relative infeasibilities and residuals, and the
    P-D error when it exceeds its tolerance.  HiGHS keeps an ``"Optimal"``
    status for 1 < X <= 100 ("no more than X") and turns it ``"Unknown"``
    above 100.  The rule is the same in HiGHS 1.16.

    Without a basis HiGHS' rule needs the residuals: a *values* whose
    residuals were not computed (a residual count ``-1``, as in
    ``getInfo()`` after a user-scaled solve, which HiGHS recomputes
    without them) raises :class:`ValueError`; use :func:`kkt_report` there."""
    t, pd = tolerances, float(_get(values, "primal_dual_objective_error"))
    if not basis_valid and -1 in (
        _get(values, "num_primal_residual_errors"),
        _get(values, "num_dual_residual_errors"),
    ):
        raise ValueError(
            "tolerance_violation: no basis and the residuals were not computed "
            "(count -1); HiGHS' X needs them: use kkt_report(h)"
        )
    pairs = (("primal", t.primal_feasibility), ("dual", t.dual_feasibility))
    if basis_valid:
        terms = [
            (f"max_{k}_infeasibility", float(_get(values, f"max_{k}_infeasibility")) / tol)
            for k, tol in pairs
            if _get(values, f"num_{k}_infeasibilities") > 0
        ]
    else:
        terms = [
            (
                f"max_relative_{k}_infeasibility",
                float(_get(values, f"max_relative_{k}_infeasibility")) / tol,
            )
            for k, tol in pairs
        ]
    if not basis_valid or _get(values, "num_primal_residual_errors") != -1:
        for k, tol in (("primal", t.primal_residual), ("dual", t.dual_residual)):
            name = f"max_relative_{k}_residual_error"
            terms.append((name, float(_get(values, name)) / tol))
    if basis_valid or pd > t.optimality:
        terms.append(("primal_dual_objective_error", pd / t.optimality))
    return max((v for _n, v in terms), default=0.0), tuple(terms)


@dataclass(frozen=True, kw_only=True)
class KktReport:
    """HiGHS' KKT check of a solution, in the space HiGHS solved.

    The first 17 fields have the names and conventions of ``HighsInfo``
    (residuals not computed: count ``-1``, maximum ``inf``).
    ``max_tolerance_relative_violation`` is HiGHS' X, ``terms`` what enters
    it (see :func:`tolerance_violation`).  ``bound_norm`` / ``cost_norm`` are
    the norms of the relative measures (``highs_norm_bounds`` /
    ``highs_norm_costs``).  Keyword-only: fields may be added."""

    num_primal_infeasibilities: int
    max_primal_infeasibility: float
    num_dual_infeasibilities: int
    max_dual_infeasibility: float
    num_relative_primal_infeasibilities: int
    max_relative_primal_infeasibility: float
    num_relative_dual_infeasibilities: int
    max_relative_dual_infeasibility: float
    num_primal_residual_errors: int
    max_primal_residual_error: float
    num_dual_residual_errors: int
    max_dual_residual_error: float
    num_relative_primal_residual_errors: int
    max_relative_primal_residual_error: float
    num_relative_dual_residual_errors: int
    max_relative_dual_residual_error: float
    primal_dual_objective_error: float
    max_tolerance_relative_violation: float
    terms: tuple[tuple[str, float], ...]
    basis_valid: bool
    tolerances: KktTolerances
    user_bound_scale: int
    user_objective_scale: int
    bound_norm: float
    cost_norm: float
    highs_version: str


def _num_max(measure: np.ndarray, tol: float) -> tuple[int, float]:
    return int(np.count_nonzero(measure > tol)), float(measure.max(initial=0.0))


def _vec(values, dtype=np.float64) -> np.ndarray:
    """A highspy vector (a list or an array) as a numpy array."""
    if isinstance(values, np.ndarray):
        return values.astype(dtype, copy=False)
    return np.fromiter(values, dtype, len(values))


def _cat(first, second) -> np.ndarray:
    return np.concatenate([_vec(first), _vec(second)])


def _matrix(lp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The LP matrix as (column id, row id, value) of each entry."""
    a = lp.a_matrix_
    start = _vec(a.start_, np.int64)
    nnz = int(start[-1])
    index = _vec(a.index_, np.int64)[:nnz]
    value = _vec(a.value_)[:nnz]
    outer = np.repeat(np.arange(start.size - 1), np.diff(start))
    if a.format_ == highspy.MatrixFormat.kRowwise:
        return index, outer, value
    return outer, index, value


def _sums(key: np.ndarray, terms: np.ndarray, size: int) -> np.ndarray:
    """``sum(terms[key == i])`` for each ``i < size``, rounded once from an
    almost exact sum, as HiGHS' ``HighsCDouble`` sums (``productQuad``).

    Error-free split (Rump, Ogita and Oishi, "Accurate floating-point
    summation", 2008): with ``sigma`` a power of two >= 2 * sum |terms| of
    the key, ``hi = (sigma + t) - sigma`` are multiples of eps * sigma, so
    their float64 sum is exact; ``lo = t - hi`` is exact too, and its sum
    errs by at most ~4 k**2 eps**2 sum |terms| (k terms of the key)."""
    sigma = np.ldexp(1.0, np.frexp(2.0 * np.bincount(key, np.abs(terms), size))[1])[key]
    hi = (sigma + terms) - sigma
    return np.bincount(key, hi, size) + np.bincount(key, terms - hi, size)


# HiGHS judges (and prints X for) these statuses only; X turns a kUnknown
# with X <= 100 into kOptimal (``lpKktCheck``).
_JUDGED = (highspy.HighsModelStatus.kOptimal, highspy.HighsModelStatus.kUnknown)


def _unsupported(h) -> tuple[str | None, Any]:
    if (h.versionMajor(), h.versionMinor()) != (1, 15) or h.versionPatch() < 1:
        return f"HiGHS {h.version()}: definitions implemented for 1.15.x, x >= 1", None
    sol = h.getSolution()
    if not (sol.value_valid and sol.dual_valid):
        return "no valid primal and dual solution", None
    status = h.getModelStatus()
    if status not in _JUDGED:
        return f"model status {h.modelStatusToString(status)}: not judged by HiGHS", None
    if h.getHessianNumNz() > 0:
        return "the model has a Hessian (LP only)", None
    return None, sol


def _integral(lp) -> bool:
    return any(v != highspy.HighsVarType.kContinuous for v in lp.integrality_)


def _bounds_and_costs(h, n: int, m: int):
    """Costs, column and row bounds, objective sense and offset off *h*
    without the ``getLp()`` copy (which holds the matrix)."""
    _st, _n, cost, col_lo, col_up, _nz = h.getCols(n, np.arange(n, dtype=np.int32))
    _st, _m, row_lo, row_up, _nz = h.getRows(m, np.arange(m, dtype=np.int32))
    return (
        cost,
        (col_lo, row_lo),
        (col_up, row_up),
        h.getObjectiveSense()[1],
        h.getObjectiveOffset()[1],
    )


def kkt_report(h) -> KktReport | None:
    """HiGHS' KKT check of the solution on handle *h*, in the space HiGHS
    solved (user-scaled when ``user_bound_scale`` / ``user_objective_scale``
    are set, otherwise as is).  HiGHS 1.15 definitions; LP only.

    Reproduces ``getKktFailures`` (``HighsSolution.cpp:73-560``) and the X of
    ``lpKktCheck``.  ``None`` (reason at DEBUG) when HiGHS is not 1.15.x with
    x >= 1 (tested on 1.15.1), there is no valid primal and dual solution,
    the model status is not ``kOptimal`` or ``kUnknown`` (the statuses HiGHS
    judges; e.g. ``kUnbounded`` has valid vectors but no check), a column is
    not continuous, or the model has a Hessian.

    Memory: with a valid basis the residuals are not computed (as in HiGHS)
    and the matrix is not read: costs and bounds come from ``getCols`` /
    ``getRows``, O(rows + cols).  Without one, ``getLp()`` copies the LP
    (about 12 bytes per nonzero, plus names), and reading it and computing
    the residuals peaks at about 90 bytes per nonzero of Python / numpy
    memory (measured; freed on return).

    Rules, over the columns then the rows (``getVariableKktFailures``,
    ``:567-661``; ``infeasibility()``, ``HighsUtils.h:219-250``):

    - *at status*: lower if ``(lower - value)**2 <= ptol``, else upper if
      ``(value - upper)**2 <= ptol``, else none.  *Mid status* (bounds
      ``lower < upper``, not free, ``length**2 > ptol``): lower below the
      midpoint, else upper; none otherwise.
    - Two passes (``:249``): first ``bound_norm`` = max |bound| at the at
      status, ``cost_norm`` = max |cost| over columns with
      ``dual**2 < dtol`` (``:261``).
    - Relative primal infeasibility = abs / (1 + measure), measure
      ``bound_norm``, for an entry at no bound raised to |upper| (mid status
      upper) or |lower| (``:319-345``).  Relative dual infeasibility = abs /
      (1 + measure), measure ``cost_norm``, raised to |cost| for a column
      with ``cost != 0`` and ``dual**2 >= dtol`` (``:366-378``).
    - Residuals (no basis): ``|Ax - r|`` / (1 + ``bound_norm``) and
      ``|(A'y - c) + d|`` / (1 + ``cost_norm``) (``:388-423``).  The float64
      products are summed to an almost exact sum rounded once, as HiGHS'
      quad-precision sums (``productQuad``), so the residuals equal HiGHS'
      up to the last bit of ``Ax`` / ``A'y``, also on rows whose activity
      is far above ``bound_norm`` (free rows, inactive rows with large
      coefficients), where plain float64 sums err by up to
      ``k * eps * sum_j |a_ij x_j|`` and can move X across 1 or 100.
    - P-D objective error = ``|p - d| / (1 + |p| + |d|)``, ``p = offset +
      c'x``, ``d = offset + sum bound * dual`` over columns and rows: the
      lower bound below the midpoint, else the upper, 1 for a free entry
      (``:469-494``, ``:1345-1389``).  The dot products are float64 in a
      different order than HiGHS': the two differ by rounding, about
      ``eps * sum |c_j x_j| / (1 + |p| + |d|)``, which is far below the
      optimality tolerance unless the objective cancels heavily.
    """
    reason, sol = _unsupported(h)
    basis_valid = reason is None and bool(h.getBasis().valid)
    lp = None
    if reason is None and (not basis_valid or _highs_int_option(h, "solve_relaxation", 0)):
        # Valid duals with integrality need solve_relaxation (a MIP solve
        # has no duals; changing integrality resets the status).
        lp = h.getLp()
        if _integral(lp):
            reason = "a column is not continuous (LP only)"
    if reason is not None:
        _logger.debug("kkt_report: None: %s", reason)
        return None
    tol = kkt_tolerances(h)
    ptol, dtol = tol.primal_feasibility, tol.dual_feasibility
    bscale = _highs_int_option(h, "user_bound_scale", 0)
    oscale = _highs_int_option(h, "user_objective_scale", 0)
    fb, fo = 2.0**bscale, 2.0**oscale
    n, m = int(h.getNumCol()), int(h.getNumRow())
    if lp is not None:
        cost, obj_sense, offset = lp.col_cost_, lp.sense_, lp.offset_
        lower, upper = (lp.col_lower_, lp.row_lower_), (lp.col_upper_, lp.row_upper_)
    else:
        cost, lower, upper, obj_sense, offset = _bounds_and_costs(h, n, m)
    sense = -1.0 if obj_sense == highspy.ObjSense.kMaximize else 1.0
    offset = float(offset)
    cost = _vec(cost) * fo
    lo, up = _cat(*lower) * fb, _cat(*upper) * fb
    val, dual = _cat(sol.col_value, sol.row_value) * fb, _cat(sol.col_dual, sol.row_dual) * fo
    col_dual = dual[:n]
    with np.errstate(invalid="ignore", over="ignore"):
        # Pass 0: the norms.
        cost_norm = float(np.abs(cost[col_dual * col_dual < dtol]).max(initial=0.0))
        at_lo = (lo - val) ** 2 <= ptol
        at_up = ~at_lo & ((val - up) ** 2 <= ptol)
        bound_norm = max(
            float(np.abs(lo[at_lo]).max(initial=0.0)), float(np.abs(up[at_up]).max(initial=0.0))
        )
        # Pass 1: absolute and relative infeasibilities.
        middle = (lo + up) * 0.5
        free = np.isneginf(lo) & np.isposinf(up)
        meaningful = (lo < up) & ~free & ((up - lo) ** 2 > ptol)
        pinf = _primal_infeasibilities(lo, val, up, ptol)
        measure = np.where(meaningful & (val >= middle), np.abs(up), np.abs(lo))
        measure = np.where(at_lo | at_up, bound_norm, np.maximum(measure, bound_norm))
        rel_pinf = pinf / (1.0 + measure)
        dinf = _dual_infeasibilities(lo, val, up, dual * sense, ptol)
        measure = np.full(dinf.size, cost_norm)
        raised = (cost != 0) & (col_dual * col_dual >= dtol)
        measure[:n][raised] = np.maximum(np.abs(cost[raised]), cost_norm)
        rel_dinf = dinf / (1.0 + measure)
        # P-D objective error.
        bound = np.where(free, 1.0, np.where(val < middle, lo, up))
        pobj = offset + float(cost @ val[:n])
        dobj = offset + float(bound @ dual)
    values: dict[str, Any] = {}
    for name, measure, t in (
        ("primal_infeasibilit", pinf, ptol),
        ("dual_infeasibilit", dinf, dtol),
        ("relative_primal_infeasibilit", rel_pinf, ptol),
        ("relative_dual_infeasibilit", rel_dinf, dtol),
    ):
        values[f"num_{name}ies"], values[f"max_{name}y"] = _num_max(measure, t)
    if basis_valid:
        for name in ("primal_residual_error", "dual_residual_error"):
            for rel in ("", "relative_"):
                values[f"num_{rel}{name}s"], values[f"max_{rel}{name}"] = -1, np.inf
    else:
        col, row, a = _matrix(lp)
        del lp
        pres = np.abs(_sums(row, a * val[:n][col], m) - val[n:])
        # double(A'y) - c, then + d, in float64 (HighsSolution.cpp:262-266, :409).
        dres = np.abs((_sums(col, a * dual[n:][row], n) - cost) + col_dual)
        for name, res, norm, t in (
            ("primal_residual_error", pres, bound_norm, tol.primal_residual),
            ("dual_residual_error", dres, cost_norm, tol.dual_residual),
        ):
            values[f"num_{name}s"], values[f"max_{name}"] = _num_max(res, t)
            rel = res / (1.0 + norm)
            values[f"num_relative_{name}s"], values[f"max_relative_{name}"] = _num_max(rel, t)
    values["primal_dual_objective_error"] = abs(pobj - dobj) / (1.0 + abs(pobj) + abs(dobj))
    x, terms = tolerance_violation(values, basis_valid=basis_valid, tolerances=tol)
    return KktReport(
        **values,
        max_tolerance_relative_violation=x,
        terms=terms,
        basis_valid=basis_valid,
        tolerances=tol,
        user_bound_scale=bscale,
        user_objective_scale=oscale,
        bound_norm=bound_norm,
        cost_norm=cost_norm,
        highs_version=h.version(),
    )
