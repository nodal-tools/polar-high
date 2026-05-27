"""Solver-agnostic LP/MIP view.

:class:`LpView` is the single typed surface that every adapter in
:mod:`polar_high.solvers` consumes.  It wraps the column and row arrays
produced by :meth:`polar_high.engine.Problem._build_lp_arrays` together
with the column-side bookkeeping (objective coefficients, bounds, names,
integrality) that ``_build_lp_arrays`` does **not** compute.

The whole point of this module is that *engine-private attribute access*
(``problem._vars``, ``problem._obj_terms``, ``problem._next_col``,
``problem._build_lp_arrays``, ``problem._obj_sense``, ``problem._obj_offset``)
happens here and **only** here.  Adapters in ``_highs.py``, ``_gurobi.py``,
``_cplex.py``, etc. take an :class:`LpView` and read its public numpy
arrays — they never reach into a :class:`Problem` directly.

LP-only / MIP-only.  Quadratic objectives, SOS sets and indicator
constraints are out of scope (matching what HiGHS supports today).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from ..engine import Problem


@dataclass(frozen=True)
class LpView:
    """Frozen snapshot of an LP/MIP suitable for any solver adapter.

    All arrays are plain numpy; ``a_*`` describe a column-major (CSC)
    sparse matrix because HiGHS, Gurobi (via scipy), and COPT all consume
    CSC natively.  Adapters preferring CSR (Xpress) call :meth:`to_csr`.

    Bounds use ``±np.inf`` for unbounded entries; adapters that need a
    solver-specific sentinel (e.g. ``highspy.kHighsInf``) convert inside
    their own ``run`` function.
    """

    n_cols: int
    n_rows: int
    col_obj: np.ndarray  # shape (n_cols,) float64
    col_lb: np.ndarray  # shape (n_cols,) float64, -inf allowed
    col_ub: np.ndarray  # shape (n_cols,) float64, +inf allowed
    integrality: np.ndarray | None  # shape (n_cols,) int8 or None for pure LP
    row_lb: np.ndarray  # shape (n_rows,) float64
    row_ub: np.ndarray  # shape (n_rows,) float64
    a_start: np.ndarray  # shape (n_cols+1,) int32 or int64
    a_index: np.ndarray  # shape (nnz,)
    a_value: np.ndarray  # shape (nnz,) float64
    col_names: list[str] = field(default_factory=list)
    row_names: list[str] = field(default_factory=list)
    sense: str = "min"
    obj_offset: float = 0.0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_problem(cls, problem: Problem) -> LpView:
        """Extract a fully-populated :class:`LpView` from a :class:`Problem`.

        This is the single place in the package that reaches into
        :class:`~polar_high.engine.Problem`'s private attributes.  After
        this call, every solver adapter consumes the view and never
        touches ``problem._vars`` etc. directly.

        Has zero side effects on ``problem``: lazy ``_Term`` plans are
        collected into local frames that drop at the end of the function,
        and the dedup-sum runs on local triples only.
        """
        # ----- column extraction ----------------------------------------
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
                            pl.concat_str(
                                [pl.col(d).cast(pl.String) for d in v.dims], separator=","
                            ),
                        ).alias("__name")
                    )
                )["__name"].to_list()
                ids_list = ids.tolist()
                for cid, nm in zip(ids_list, tagged):
                    col_names[cid] = nm
            else:
                cid0 = int(ids[0])
                col_names[cid0] = v.name

        # ----- objective scatter ----------------------------------------
        # Materialize each term into a local DataFrame and let it drop;
        # never populate an eager cache on _Term.
        # Layer 2 col-factor (off ⇒ no-op).  Mirrors the four
        # in-engine consumers; cost has no row-factor entry.
        _cf_obj = problem._layer2_col_factor
        for t in problem._obj_terms:
            f = t.lazy.collect()
            cids = f["col_id"].to_numpy()
            vals = f["coef"].to_numpy()
            if _cf_obj is not None:
                vals = vals * _cf_obj[cids]
            np.add.at(col_obj, cids, vals)
            del f

        # ----- row + matrix build ---------------------------------------
        # _build_lp_arrays converts bounds from ±inf to ±kHighsInf for
        # immediate HiGHS consumption; the view stores ±inf for adapter
        # portability, so we convert back here.
        import highspy

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

        inf = highspy.kHighsInf
        col_lb_v = np.where(col_lb_h == -inf, -np.inf, col_lb_h).astype(np.float64)
        col_ub_v = np.where(col_ub_h == inf, np.inf, col_ub_h).astype(np.float64)
        row_lb_v = np.where(row_lb_h == -inf, -np.inf, row_lb_h).astype(np.float64)
        row_ub_v = np.where(row_ub_h == inf, np.inf, row_ub_h).astype(np.float64)

        integrality = col_int if col_int.any() else None

        return cls(
            n_cols=int(n_cols),
            n_rows=int(n_rows),
            col_obj=col_obj,
            col_lb=col_lb_v,
            col_ub=col_ub_v,
            integrality=integrality,
            row_lb=row_lb_v,
            row_ub=row_ub_v,
            a_start=starts,
            a_index=sorted_r,
            a_value=sorted_v,
            col_names=col_names,
            row_names=row_names,
            sense=problem._obj_sense,
            obj_offset=float(problem._obj_offset),
        )

    # ------------------------------------------------------------------
    # Format conversion
    # ------------------------------------------------------------------
    def to_csr(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(start, index, value)`` for a row-major (CSR) view.

        Used by adapters whose underlying solver prefers row-oriented
        input (e.g. Xpress' ``loadproblem``, CPLEX's
        ``linear_constraints.add`` with ``SparsePair`` rows).  Allocates
        fresh arrays; the view's CSC arrays remain unchanged.
        """
        n_rows = int(self.n_rows)
        n_cols = int(self.n_cols)
        nnz = int(self.a_value.size)
        a_start = self.a_start
        a_index = self.a_index
        a_value = self.a_value

        if nnz == 0:
            idx_dtype = a_index.dtype if a_index.size else np.int32
            return (
                np.zeros(n_rows + 1, dtype=idx_dtype),
                np.zeros(0, dtype=idx_dtype),
                np.zeros(0, dtype=np.float64),
            )

        # Reconstruct (row, col) pairs from CSC, then re-sort by row.
        col_of = np.empty(nnz, dtype=np.int64)
        for c in range(n_cols):
            col_of[a_start[c] : a_start[c + 1]] = c
        row_of = a_index.astype(np.int64)

        order = np.lexsort((col_of, row_of))  # primary: row, secondary: col
        sorted_c = col_of[order].astype(a_index.dtype)
        sorted_v = a_value[order].astype(np.float64)
        sorted_r = row_of[order]

        row_start = np.zeros(n_rows + 1, dtype=a_start.dtype)
        np.add.at(row_start[1:], sorted_r, 1)
        row_start = np.cumsum(row_start).astype(a_start.dtype)

        return row_start, sorted_c, sorted_v

    # ------------------------------------------------------------------
    # Row-sense conversion
    # ------------------------------------------------------------------
    def row_sense_rhs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert row bounds into CPLEX-style ``(senses, rhs, range_values)``.

        Mapping:

        * ``row_lb == row_ub``                 → ``"E"``, rhs = ``row_lb``,         range = 0
        * ``row_lb == -inf``                   → ``"L"``, rhs = ``row_ub``,         range = 0
        * ``row_ub == +inf``                   → ``"G"``, rhs = ``row_lb``,         range = 0
        * otherwise (ranged, both finite)      → ``"R"``, rhs = ``row_ub``,
          range = ``row_lb - row_ub`` (CPLEX convention: ``range`` is negative;
          the lower bound is ``rhs + range``)

        Adapters whose underlying solver lacks native range support
        should call :meth:`split_ranged_rows` first.
        """
        n_rows = int(self.n_rows)
        senses = np.empty(n_rows, dtype="U1")
        rhs = np.zeros(n_rows, dtype=np.float64)
        range_values = np.zeros(n_rows, dtype=np.float64)

        lb = self.row_lb
        ub = self.row_ub
        eq = lb == ub
        l_only = (~eq) & np.isneginf(lb)
        g_only = (~eq) & np.isposinf(ub) & (~l_only)
        ranged = ~(eq | l_only | g_only)

        senses[eq] = "E"
        rhs[eq] = lb[eq]
        senses[l_only] = "L"
        rhs[l_only] = ub[l_only]
        senses[g_only] = "G"
        rhs[g_only] = lb[g_only]
        senses[ranged] = "R"
        rhs[ranged] = ub[ranged]
        range_values[ranged] = lb[ranged] - ub[ranged]

        return senses, rhs, range_values

    # ------------------------------------------------------------------
    # Range-row splitting
    # ------------------------------------------------------------------
    def split_ranged_rows(self) -> LpView:
        """Return a new :class:`LpView` with each ranged row replaced by
        a ``>=``/``<=`` pair.

        Each ranged row (both ``row_lb`` and ``row_ub`` finite, and
        unequal) gets split into:

        * a ``>=`` row with ``row_lb`` of the original (``row_ub = +inf``),
          name suffixed ``_lo``
        * a ``<=`` row with ``row_ub`` of the original (``row_lb = -inf``),
          name suffixed ``_hi``

        The CSC matrix entries of the original row are duplicated into
        both new rows (same column positions, same coefficient values).
        Useful for solver APIs that lack a native range concept.

        If no rows are ranged, the result is a shallow copy of ``self``.
        """
        senses, rhs, range_values = self.row_sense_rhs()
        ranged_mask = senses == "R"
        if not ranged_mask.any():
            return replace(self)

        n_rows = int(self.n_rows)
        # New row-index mapping: for each original row, list of new rids
        # it expands to.
        ranged_idx = np.flatnonzero(ranged_mask)
        n_extra = int(ranged_idx.size)
        new_n_rows = n_rows + n_extra

        # Build the new row->old row map.  Order:
        #   for each original row in order,
        #       if non-ranged: emit one row
        #       if ranged:     emit two rows (lo, hi)
        # Layout new rows so the original-row prefix is preserved (the
        # non-ranged rows keep their old rid), then the extra "_hi"
        # copies of ranged rows are appended at the end.  This keeps row
        # numbering stable for non-ranged rows.
        new_row_lb = np.empty(new_n_rows, dtype=np.float64)
        new_row_ub = np.empty(new_n_rows, dtype=np.float64)
        new_row_names: list[str] = [""] * new_n_rows

        # First-n_rows slots: the original row, but for ranged rows it
        # becomes the ">=" half ("_lo").
        new_row_lb[:n_rows] = self.row_lb
        new_row_ub[:n_rows] = self.row_ub
        for i in range(n_rows):
            new_row_names[i] = self.row_names[i]
        # ranged rows: clamp the upper to +inf (>= half), suffix "_lo"
        new_row_ub[ranged_idx] = np.inf
        for k, i in enumerate(ranged_idx.tolist()):
            new_row_names[i] = f"{self.row_names[i]}_lo"
            # appended "_hi" half
            new_row_names[n_rows + k] = f"{self.row_names[i]}_hi"

        # Extra slots at the tail: the "<=" half, ub = original ub,
        # lb = -inf.
        new_row_lb[n_rows : n_rows + n_extra] = -np.inf
        new_row_ub[n_rows : n_rows + n_extra] = self.row_ub[ranged_idx]

        # ---- matrix rebuild ---------------------------------------------
        # Duplicate every CSC entry whose row index is in ranged_idx into
        # the corresponding new row (`n_rows + k`).
        a_start = self.a_start
        a_index = self.a_index
        a_value = self.a_value
        n_cols = int(self.n_cols)

        # Old row id -> new row id for the "_hi" half (only meaningful
        # for ranged rows).
        hi_of_old = np.full(n_rows, -1, dtype=np.int64)
        hi_of_old[ranged_idx] = np.arange(n_rows, n_rows + n_extra, dtype=np.int64)

        # Walk columns, build new CSC.
        new_starts = np.zeros(n_cols + 1, dtype=a_start.dtype)
        new_index_chunks: list[np.ndarray] = []
        new_value_chunks: list[np.ndarray] = []
        nnz_running = 0
        for c in range(n_cols):
            lo, hi = int(a_start[c]), int(a_start[c + 1])
            if lo == hi:
                new_starts[c + 1] = nnz_running
                continue
            col_rows = a_index[lo:hi]
            col_vals = a_value[lo:hi]
            # Identify the ranged rows touched by this column.
            in_ranged = ranged_mask[col_rows.astype(np.int64)]
            n_dup = int(in_ranged.sum())
            new_index_chunks.append(col_rows)
            new_value_chunks.append(col_vals)
            if n_dup:
                dup_old = col_rows[in_ranged].astype(np.int64)
                dup_new = hi_of_old[dup_old].astype(a_index.dtype)
                new_index_chunks.append(dup_new)
                new_value_chunks.append(col_vals[in_ranged])
            nnz_running += (hi - lo) + n_dup
            new_starts[c + 1] = nnz_running

        if new_index_chunks:
            new_index = np.concatenate(new_index_chunks).astype(a_index.dtype)
            new_value = np.concatenate(new_value_chunks).astype(np.float64)
        else:
            new_index = np.zeros(0, dtype=a_index.dtype)
            new_value = np.zeros(0, dtype=np.float64)

        return LpView(
            n_cols=n_cols,
            n_rows=new_n_rows,
            col_obj=self.col_obj,
            col_lb=self.col_lb,
            col_ub=self.col_ub,
            integrality=self.integrality,
            row_lb=new_row_lb,
            row_ub=new_row_ub,
            a_start=new_starts,
            a_index=new_index,
            a_value=new_value,
            col_names=list(self.col_names),
            row_names=new_row_names,
            sense=self.sense,
            obj_offset=self.obj_offset,
        )


__all__ = ["LpView"]
