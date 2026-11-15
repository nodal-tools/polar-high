"""Name the entries at one end of one family's magnitudes.

:func:`locate_extremes` walks ONE variable or constraint family (the one a
:class:`RangeReport` extreme names) with the bounded coefficient walk and a
:class:`TopKAbsReducer`, and decodes the row / column of each of the ``k``
smallest or largest entries into its dimension values.  Peak memory is that
of one walk batch, so it is meant for a warning / error path: one family,
once.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from .._collect import collect_streaming_in_join_order
from ._coef_walk import (
    CoefWalkRecipe,
    CountWithinReducer,
    Reducer,
    TopKAbsReducer,
    bounded_coefficient_walk,
)
from ._config import ScalingConfig
from ._precedence import resolve_infinite_bound

_BATCH_ROWS = 256_000

_COMPONENTS = ("bound", "rhs", "matrix")


def _find_constraint(problem: Any, family: str) -> tuple[int, Any, pl.DataFrame | None] | None:
    """``(base_row, proto, over)`` of constraint family ``family``."""
    next_row = 0
    for cname, proto, over in problem._cstrs:
        if cname == family:
            return next_row, proto, over
        next_row += 1 if over is None else int(over.height)
    return None


def _row_dims(over: pl.DataFrame | None, rid: int) -> dict[str, Any]:
    if over is None or rid < 0:
        return {}
    return dict(over.row(int(rid), named=True))


def _col_dims(problem: Any, col_id: int) -> tuple[str | None, dict[str, Any]]:
    """``(variable family, dims)`` of column ``col_id``."""
    if col_id < 0:
        return None, {}
    for v in problem._vars.values():
        ids = v.frame["col_id"].to_numpy()
        if ids.size == 0 or not (ids.min() <= col_id <= ids.max()):
            continue
        hit = np.flatnonzero(ids == col_id)
        if hit.size:
            row = v.frame.row(int(hit[0]), named=True)
            row.pop("col_id", None)
            return str(v.name), dict(row)
    return None, {}


def _integer_cols(problem: Any) -> np.ndarray | None:
    int_vars = [v for v in problem._vars.values() if getattr(v, "integer", False)]
    if not int_vars:
        return None
    mask = np.zeros(int(problem._next_col), dtype=bool)
    for v in int_vars:
        mask[v.frame["col_id"].to_numpy()] = True
    return mask


def _bake(term: Any) -> tuple[pl.LazyFrame, tuple[str, ...]]:
    from ..engine import _apply_where_frames, _apply_where_map_frames

    lf = _apply_where_frames(term.lazy, term.dims, term.where_frames)
    return _apply_where_map_frames(lf, term.dims, term.where_map_frames)


def _feed_rows(reducers: list, n_rows: int, cids: np.ndarray, coef: np.ndarray) -> None:
    """Feed a term broadcast over ``n_rows`` rows (every row carries the same
    ``(col_id, coef)`` list), in bounded row chunks."""
    if cids.size == 0 or n_rows == 0:
        return
    per = max(1, _BATCH_ROWS // max(1, cids.size))
    for start in range(0, n_rows, per):
        stop = min(n_rows, start + per)
        rows = np.arange(start, stop, dtype=np.int64)
        rid = np.repeat(rows, cids.size)
        cid = np.tile(cids, rows.size)
        cf = np.tile(coef, rows.size)
        for r in reducers:
            r.update(rid, cid, cf)


class _Keep(Reducer):
    """Pass-through that ignores the walk's per-call ``init`` so one reducer
    accumulates over several walks (one per matrix term)."""

    def __init__(self, inner: Reducer) -> None:
        self.inner = inner

    def init(self) -> None:
        pass

    def update(self, rid_arr, col_id_arr, coef_arr) -> None:
        self.inner.update(rid_arr, col_id_arr, coef_arr)

    def finalize(self) -> None:
        return None


def _walk_constraint(problem, proto, over, base_row, component, reducers) -> None:
    from ..engine import Param, _align_enum_join_keys

    reducers = [_Keep(r) for r in reducers]

    l2_rf = getattr(problem, "_layer2_row_factor", None)
    l2_cf = getattr(problem, "_layer2_col_factor", None)
    dense_axes = getattr(problem, "_dense_axes", None)
    row_count = 1 if over is None else int(over.height)
    if component == "rhs":
        rhs = proto.rhs
        scale = (l2_rf, base_row, None)
        if isinstance(rhs, (int, float)):
            _feed_rows(reducers, row_count, np.array([-1]), np.array([float(rhs)]))
        elif isinstance(rhs, Param):
            if over is not None and rhs.dims:
                if isinstance(rhs._sources, list) and rhs._sources:
                    recipe = CoefWalkRecipe.from_rhs_chain(rhs)
                else:
                    recipe = CoefWalkRecipe.from_rhs_param(rhs)
                bounded_coefficient_walk(
                    over, recipe, scale, reducers, batch_rows=_BATCH_ROWS, dense_axes=dense_axes
                )
            else:
                f = rhs.frame
                if "value" in f.columns and f.height > 0:
                    vals = f["value"].to_numpy().astype(np.float64)
                    _feed_rows(reducers, row_count, np.full(vals.size, -1), vals)
        # A Var / Expr right-hand side folds into the matrix: no RHS entry.
        return
    # Matrix: every term of the family, the same dispatch as the range walk.
    scale = (l2_rf, base_row, l2_cf)
    axis_cols = [] if over is None else list(over.columns)
    for term in proto.expr.terms:
        term_lazy, term_dims = _bake(term)
        if term_dims and over is not None and CoefWalkRecipe.is_buildable(term):
            bounded_coefficient_walk(
                over,
                CoefWalkRecipe.from_term(term),
                scale,
                reducers,
                batch_rows=_BATCH_ROWS,
                dense_axes=dense_axes,
            )
            continue
        if term_dims and over is not None:
            # A fully collapsed Sum: its reduced plan is small.
            on = [d for d in term_dims if d in axis_cols]
            rid_lf = over.with_columns(_rid=pl.int_range(0, over.height, dtype=pl.Int64)).lazy()
            rl_a, tl_a = _align_enum_join_keys(rid_lf, term_lazy, on)
            df = collect_streaming_in_join_order(
                rl_a.join(tl_a, on=on, how="inner").select("_rid", "col_id", "coef")
            )
            if df.height:
                rid = df["_rid"].to_numpy().astype(np.int64)
                cid = df["col_id"].to_numpy().astype(np.int64)
                coef = df["coef"].to_numpy().astype(np.float64)
                for r in reducers:
                    r.update(rid, cid, coef)
            continue
        df = collect_streaming_in_join_order(term_lazy.select("col_id", "coef"))
        if df.height:
            _feed_rows(
                reducers,
                row_count,
                df["col_id"].to_numpy().astype(np.int64),
                df["coef"].to_numpy().astype(np.float64),
            )


def _walk_bounds(problem, var, reducers) -> None:
    l2_cf = getattr(problem, "_layer2_col_factor", None)
    ids = var.frame["col_id"].to_numpy().astype(np.int64)
    if ids.size == 0:
        return
    factor = None if l2_cf is None else np.abs(l2_cf[ids])
    for b in (var.lower, var.upper):
        if isinstance(b, np.ndarray):
            post = np.abs(np.asarray(b, dtype=np.float64))
        else:
            post = np.full(ids.size, abs(float(b)))
        model = post if factor is None else post * factor
        for start in range(0, ids.size, _BATCH_ROWS):
            sl = slice(start, start + _BATCH_ROWS)
            for r in reducers:
                r.update_magnitudes(post[sl], model[sl], None, ids[sl])


def locate_extremes(
    problem: Any,
    family: str,
    group: str,
    side: str,
    *,
    k: int = 5,
    floor: float | None = None,
    factor: float = 10.0,
) -> list[dict[str, Any]]:
    """The ``k`` smallest (``side="min"``) or largest (``side="max"``)
    entries of one family, best first.

    ``group`` is the component the family is walked for: ``bound``
    (variable family bounds), ``rhs`` (constraint right-hand sides) or
    ``matrix`` (constraint matrix terms); the :attr:`Extreme.component` of a
    :class:`RangeReport` extreme.  ``integer_bound`` / ``integer_matrix``
    take the integer columns only; ``bound`` / ``matrix`` the continuous
    ones (``bound_rhs`` is accepted as ``bound`` for a variable family and
    ``rhs`` for a constraint).  Entries as in the readout: finite, non-zero,
    and for bounds / RHS below the solver's ``infinite_bound``; on the
    ``min`` side of bounds / RHS also at least ``floor`` in model units
    (``None``: the problem's residue floor, as :class:`RangeReport` uses).

    Each dict: ``family``, ``component``, ``post`` (reported magnitude),
    ``model`` (model units), ``dims`` (the row's dimension values for RHS /
    matrix, the column's for bounds; Enum codes decoded), ``rid`` / ``col_id``
    (local row id, column id; ``-1`` when not applicable), ``column`` (for
    matrix entries: ``{"family", "dims"}`` of the column), and
    ``count_within_factor`` (how many entries of the family lie within
    ``factor`` of the best one, the best included) with ``factor``.  An
    empty list when the family has no such entry.
    """
    from ._ranges import _resolve_residue_floor

    if side not in ("min", "max"):
        raise ValueError(f"side must be 'min' or 'max'; got {side!r}")
    columns = "continuous"
    component = group
    if group in ("integer_bound", "integer_matrix"):
        columns = "integer"
        component = group.removeprefix("integer_")
    if component == "bound_rhs":
        component = "bound" if family in problem._vars else "rhs"
    if component not in _COMPONENTS:
        raise ValueError(f"group must name bound, rhs or matrix; got {group!r}")
    if component == "matrix":
        fl, infinite = 0.0, math.inf
    else:
        fl = _resolve_residue_floor(problem, ScalingConfig()) if floor is None else float(floor)
        infinite = resolve_infinite_bound(problem)
    int_cols = _integer_cols(problem)
    if component == "rhs":
        columns = "all"  # a right-hand side has no column

    def _make_topk(scale) -> TopKAbsReducer:
        return TopKAbsReducer(
            scale,
            k,
            side,
            fl,
            infinite=infinite,
            integer_cols=int_cols,
            columns=columns,
            factor=factor,
        )

    over = None
    if component == "bound":
        var = problem._vars.get(family)
        if var is None:
            raise KeyError(f"no variable family {family!r}")
        topk = _make_topk((None, 0, None))
        _walk_bounds(problem, var, [topk])
        if topk.near_overflow and topk.best is not None:
            counter = CountWithinReducer(topk, topk.best)
            _walk_bounds(problem, var, [counter])
            count = counter.finalize()
        else:
            count = None
    else:
        found = _find_constraint(problem, family)
        if found is None:
            raise KeyError(f"no constraint family {family!r}")
        base_row, proto, over = found
        l2_rf = getattr(problem, "_layer2_row_factor", None)
        l2_cf = getattr(problem, "_layer2_col_factor", None)
        scale = (l2_rf, base_row, None if component == "rhs" else l2_cf)
        topk = _make_topk(scale)
        _walk_constraint(problem, proto, over, base_row, component, [topk])
        count = None
        if topk.near_overflow and topk.best is not None:
            counter = CountWithinReducer(topk, topk.best)
            _walk_constraint(problem, proto, over, base_row, component, [counter])
            count = counter.finalize()
    entries, near = topk.finalize()
    if count is None:
        count = near
    out: list[dict[str, Any]] = []
    for e in entries:
        rec: dict[str, Any] = {"family": family, "component": component, **e}
        if component == "bound":
            _fam, rec["dims"] = _col_dims(problem, e["col_id"])
            rec["rid"] = -1
        else:
            rec["dims"] = _row_dims(over, e["rid"])
            if component == "matrix":
                cfam, cdims = _col_dims(problem, e["col_id"])
                rec["column"] = {"family": cfam, "dims": cdims}
            else:
                rec["col_id"] = -1
        rec["count_within_factor"] = int(count) if count is not None else 0
        rec["factor"] = float(factor)
        out.append(rec)
    return out


__all__ = ["locate_extremes"]
