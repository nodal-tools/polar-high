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
    above 100.  The rule is the same in HiGHS 1.16."""
    t, pd = tolerances, float(_get(values, "primal_dual_objective_error"))
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


@dataclass(frozen=True)
class KktReport:
    """HiGHS' KKT check of a solution, in the space HiGHS solved.

    The first 17 fields have the names and conventions of ``HighsInfo``
    (residuals not computed: count ``-1``, maximum ``inf``).
    ``max_tolerance_relative_violation`` is HiGHS' X, ``terms`` what enters
    it (see :func:`tolerance_violation`).  ``bound_norm`` / ``cost_norm`` are
    the norms of the relative measures (``highs_norm_bounds`` /
    ``highs_norm_costs``)."""

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


def _unsupported(h) -> tuple[str | None, Any]:
    if (h.versionMajor(), h.versionMinor()) != (1, 15):
        return f"HiGHS {h.version()}: definitions not implemented (HiGHS 1.15 only)", None
    sol = h.getSolution()
    if not (sol.value_valid and sol.dual_valid):
        return "no valid primal and dual solution", None
    if h.getHessianNumNz() > 0:
        return "the model has a Hessian (LP only)", None
    return None, sol


def kkt_report(h) -> KktReport | None:
    """HiGHS' KKT check of the solution on handle *h*, in the space HiGHS
    solved (user-scaled when ``user_bound_scale`` / ``user_objective_scale``
    are set, otherwise as is).  HiGHS 1.15 definitions; LP only.

    Reproduces ``getKktFailures`` (``HighsSolution.cpp:73-560``) and the X of
    ``lpKktCheck``.  With a valid basis, residuals are not computed (as in
    HiGHS) and the matrix is not read.  ``None`` (reason at DEBUG) when HiGHS
    is not 1.15.x, there is no valid primal and dual solution, a column is
    not continuous, or the model has a Hessian.

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
      ``|A'y - c + d|`` / (1 + ``cost_norm``) (``:388-423``).  HiGHS sums in
      quad precision, this in float64: an absolute residual below about
      ``k * eps * sum_j |a_ij x_j|`` (k entries in the row) is rounding noise
      and can differ from HiGHS'; counts can differ only when a residual
      tolerance sits at that floor.  X uses the relative residuals, whose
      floor (~1e-16) is far below the tolerances.
    - P-D objective error = ``|p - d| / (1 + |p| + |d|)``, ``p = offset +
      c'x``, ``d = offset + sum bound * dual`` over columns and rows: the
      lower bound below the midpoint, else the upper, 1 for a free entry
      (``:469-494``, ``:1345-1389``).
    """
    reason, sol = _unsupported(h)
    lp = h.getLp() if reason is None else None
    if lp is not None and set(lp.integrality_) - {highspy.HighsVarType.kContinuous}:
        reason = "a column is not continuous (LP only)"
    if reason is not None:
        _logger.debug("kkt_report: None: %s", reason)
        return None
    tol = kkt_tolerances(h)
    ptol, dtol = tol.primal_feasibility, tol.dual_feasibility
    bscale = _highs_int_option(h, "user_bound_scale", 0)
    oscale = _highs_int_option(h, "user_objective_scale", 0)
    fb, fo = 2.0**bscale, 2.0**oscale
    n = int(lp.num_col_)
    sense = -1.0 if lp.sense_ == highspy.ObjSense.kMaximize else 1.0
    offset = float(lp.offset_)
    cost = _vec(lp.col_cost_) * fo
    lo, up = _cat(lp.col_lower_, lp.row_lower_) * fb, _cat(lp.col_upper_, lp.row_upper_) * fb
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
    basis_valid = bool(h.getBasis().valid)
    if basis_valid:
        for name in ("primal_residual_error", "dual_residual_error"):
            for rel in ("", "relative_"):
                values[f"num_{rel}{name}s"], values[f"max_{rel}{name}"] = -1, np.inf
    else:
        col, row, a = _matrix(lp)
        ax = np.bincount(row, weights=a * val[:n][col], minlength=val.size - n)
        pres = np.abs(ax - val[n:])
        aty = np.bincount(col, weights=a * dual[n:][row], minlength=n)
        dres = np.abs(aty - cost + col_dual)
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
