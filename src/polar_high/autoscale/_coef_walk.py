"""Phase D-5 step 1 — one general bounded coefficient-walk primitive.

:func:`bounded_coefficient_walk` consumes a block-COO *recipe* (the same
``(var_source, param_sources, coef_scalar, where_frames, where_map_frames,
sum_block_meta_or_reduce_dims)`` tuple the canonical-matrix builders
consume) and walks the resulting ``(row/col index, coef)`` stream in
fixed-size *batches*, calling a list of stateful **reducers** on each
batch and freeing the batch product before the next.  No more than one
batch's worth of ``(rid/col_id, coef)`` triple is held at any time, so
peak memory is bounded by ``batch_rows`` (× the per-factor numpy buffers
the block-COO chain needs for that slice), NOT by the full
``Var × P1 × P2 …`` product.

It is the single primitive that later steps (Phase D-5 steps 2-5) will
wire in to replace BOTH:

* the autoscale range readout's materialising fallback in
  :mod:`polar_high.autoscale._ranges` (via :class:`MinMaxAbsReducer`), and
* FlexTool's ``bucket_coefficients`` log2-histogram aggregation (via
  :class:`Log2HistogramReducer`).

This module does NOT reimplement the block-COO chain math.  It reuses
:func:`polar_high.engine._build_block_coo_plan` /
:func:`_build_block_coo_plan_joined` (non-Sum arm),
:func:`_build_sum_block_coo_plan` /
:func:`_build_sum_block_coo_relabel` (Sum arm), and
:func:`_build_lhs_pruned_plan` (the always-correct merged-lazy
prune-down backstop) per batch.  The batch loop slices the spine and
restricts the builders' ``row_index_lf`` to the slice's rows, so the
builders prune the Var/Param leaves to the batch keys and emit only the
batch's triples.

Two spine modes
---------------

* **constraint** — the spine is the constraint ``over`` grid; each batch
  carries an ``_rid`` int-range column (``base_row + _rid`` indexes the
  Layer-2 row factor).  Builders attach ``_rid`` by inner-joining the
  batch ``row_index_lf`` on the term's ``on`` keys, so the emitted triple
  is ``(_rid, col_id, coef)`` for exactly the batch's rows.
* **column** — the spine is the ``Var.frame`` (objective / bare-Var,
  no constraint row).  There is no ``_rid``; each batch slices the Var
  seed and the recipe's per-cell product is rebuilt positionally on that
  slice, emitting ``(col_id, coef)`` (``rid`` reported as ``-1`` — the
  objective row has no row factor, GLPK convention).

A constraint spine may carry EITHER an LHS ``Var × Param`` recipe (the
default) OR a Var-LESS ``param_only`` recipe — the constraint RHS
Param-product chain.  The Var-less batch builder seeds ``coef =
coef_scalar`` over the batch's ``over`` rows and multiplies the Param
chain onto them (positional fast path reusing the RHS bounded builder's
three alignment cases, or a batched merged-lazy prune-down for the shapes
that decline), emitting ``(_rid, coef)`` with ``col_id`` absent (an
all-``-1`` array — the RHS carries no col factor; the reducer is fed
``l2_cf=None`` and never indexes it).

Reducer ordering / parity
--------------------------

* **min/max** (:class:`MinMaxAbsReducer`) is associative AND
  order-independent, so batching is trivially exact — the result is
  BYTE-IDENTICAL to a whole-collect min/max for every ``batch_rows``.
* **log2 histogram** (:class:`Log2HistogramReducer`) accumulates per
  bucket ``(sum log2|coef|, count, min, max)``.  ``count``, ``min`` and
  ``max`` combine exactly across batches; the ``sum log2`` summation
  ORDER differs from a single whole-collect sum, so the histogram matches
  a whole-collect histogram only to within floating-point reassociation
  (exactly for a single batch).  Documented; min/max stays exact.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Any

import numpy as np
import polars as pl

from ..engine import (
    Param,
    SumBlockMeta,
    Var,
    _align_enum_join_keys,
    _apply_where_frames,
    _apply_where_map_frames,
    _block_coo_classify,
    _block_coo_disabled,
    _build_block_coo_plan,
    _build_lhs_pruned_plan,
    _build_sum_block_coo_plan,
    _emit_block_coo_path,
    _sum_block_coo_classify,
    _SumBlockCooFallback,
    _verify_dense_sorted,
)

__all__ = [
    "CoefBatch",
    "CoefWalkRecipe",
    "Log2HistogramReducer",
    "MinMaxAbsReducer",
    "Reducer",
    "bounded_coefficient_walk",
]


# ---------------------------------------------------------------------------
# Recipe + batch containers


class CoefWalkRecipe:
    """The block-COO term recipe a walk consumes.

    Mirrors exactly the fields the canonical-matrix / range builders read
    off a ``_Term`` (non-Sum arm) or its ``sum_block_meta`` (Sum arm):

    * ``var_source`` — the originating :class:`Var` (the ``col_id``
      source).  ``None`` is not accepted — every block-evaluable term has
      a Var seed.
    * ``param_sources`` — the FULL ``list[(Param, direction)]`` chain
      (``direction`` ``+1`` numerator / ``-1`` denominator).  May be empty
      (a bare Var term).
    * ``coef_scalar`` — the cumulative constant folded into ``coef``.
    * ``where_frames`` — deferred pure-filter Where frames (semi-join,
      order-preserving) or ``None``.
    * ``where_map_frames`` — deferred map-effect Where frames
      (``(frame, extras)`` tuples, dim-extending inner-join) or ``None``.
    * ``sum_block_meta`` — a :class:`SumBlockMeta` when the term is a
      ``Sum``-wrapped chain (``var_source`` on the term itself is then
      cleared; the recipe carries the meta's pre-Sum state); ``None`` for
      a non-Sum term.

    For a Sum recipe, ``var_source`` / ``param_sources`` / ``coef_scalar``
    / ``where_frames`` / ``where_map_frames`` are taken from the meta (the
    pre-Sum, un-survivor-filtered state) so the builders rebuild the
    unreduced product; the meta is passed through to the Sum builders.

    Param-only (Var-LESS) mode
    --------------------------
    When ``param_only`` is set (``var_source is None``) the recipe
    describes a *Var-less* chain: the spine is the constraint ``over``
    grid (carrying ``_rid``); ``param_sources`` is a pure ``[(Param,
    direction)]`` chain; ``coef_scalar`` is the chain's accumulated
    constant (the composite RHS Param's ``_value_scalar``); there is NO
    ``col_id`` source.  The per-batch builder seeds ``coef = coef_scalar``
    over the batch's ``over`` rows and multiplies the Param chain onto
    them — the same numpy op sequence the canonical builder's RHS
    prune-down (and the inline ``_rhs_chain_bounded_coef`` positional
    fast path) produce — emitting ``(_rid, coef)`` with ``col_id`` absent
    (the reducer is fed an empty ``col_id`` and skips the col factor).
    This is the mode the RHS decline branch of
    :func:`_ranges._ranges_via_streaming` and (later) FlexTool's
    ``bucket_coefficients`` RHS families route through.
    """

    __slots__ = (
        "var_source",
        "param_sources",
        "coef_scalar",
        "where_frames",
        "where_map_frames",
        "sum_block_meta",
        "reduced_lazy",
        "reduced_dims",
        "param_only",
    )

    def __init__(
        self,
        var_source: Var | None,
        param_sources: Sequence[tuple[Param, int]],
        coef_scalar: float = 1.0,
        where_frames: tuple[pl.LazyFrame, ...] | None = None,
        where_map_frames: tuple[tuple[pl.LazyFrame, frozenset[str]], ...]
        | None = None,
        sum_block_meta: SumBlockMeta | None = None,
        reduced_lazy: pl.LazyFrame | None = None,
        reduced_dims: tuple[str, ...] | None = None,
        param_only: bool = False,
    ) -> None:
        if param_only:
            if var_source is not None:
                raise TypeError(
                    "CoefWalkRecipe.param_only requires var_source=None "
                    f"(Var-less chain); got {type(var_source).__name__}"
                )
            if sum_block_meta is not None:
                raise TypeError(
                    "CoefWalkRecipe.param_only does not support a "
                    "sum_block_meta (the Var-less RHS chain is unreduced)"
                )
        elif not isinstance(var_source, Var):
            raise TypeError(
                "CoefWalkRecipe.var_source must be a Var; got "
                f"{type(var_source).__name__}"
            )
        self.var_source = var_source
        self.param_sources = list(param_sources)
        self.coef_scalar = float(coef_scalar)
        self.where_frames = where_frames
        self.where_map_frames = where_map_frames
        self.sum_block_meta = sum_block_meta
        self.param_only = bool(param_only)
        # ``reduced_lazy`` is the term's OWN post-Sum lazy plan (columns
        # ``*reduced_dims, col_id, coef``).  It is the always-correct
        # backstop the engine uses when the Sum block-COO classifier
        # DECLINES (e.g. the map got baked eagerly, clearing
        # ``where_map_frames``): the reduced coef cannot be rebuilt from
        # the seed + chain alone, so we emit this reduced plan verbatim
        # (attaching ``_rid`` per batch).  ``None`` for non-Sum recipes,
        # which use the merged-lazy prune-down backstop instead.
        self.reduced_lazy = reduced_lazy
        self.reduced_dims = reduced_dims

    @classmethod
    def from_term(cls, term: Any) -> CoefWalkRecipe:
        """Build a recipe from a polar-high ``_Term``.

        For a Sum-wrapped term (``term.sum_block_meta`` set) the recipe's
        seed/chain fields come from the meta's pre-Sum state.  For a
        non-Sum term they come from the term directly.
        """
        meta = getattr(term, "sum_block_meta", None)
        if meta is not None:
            return cls(
                var_source=meta.var_source,
                param_sources=list(meta.param_sources),
                coef_scalar=meta.coef_scalar,
                where_frames=meta.where_frames,
                where_map_frames=meta.where_map_frames,
                sum_block_meta=meta,
                reduced_lazy=term.lazy,
                reduced_dims=tuple(term.dims),
            )
        return cls(
            var_source=term.var_source,
            param_sources=list(term.param_sources or []),
            coef_scalar=term.coef_scalar,
            where_frames=term.where_frames,
            where_map_frames=term.where_map_frames,
            sum_block_meta=None,
        )

    @staticmethod
    def is_buildable(term: Any) -> bool:
        """Routability predicate for the matrix / LHS coefficient walk.

        Returns ``True`` iff :meth:`from_term` would SUCCEED for ``term``
        (treated as a NON-``param_only`` term).  This MUST stay in lockstep
        with :meth:`from_term`: that method raises ``TypeError`` from
        ``__init__`` ("var_source must be a Var; got NoneType") whenever the
        Var it selects is ``None``, and the Var it selects depends on
        whether the term is ``Sum``-wrapped:

        * ``term.sum_block_meta is not None`` → the recipe's Var is
          ``meta.var_source`` (the meta branch); buildable iff
          ``meta.var_source is not None``.
        * ``term.sum_block_meta is None`` → the recipe's Var is
          ``term.var_source`` (the non-Sum branch); buildable iff
          ``term.var_source is not None``.

        A SHALLOW ``var_source is not None OR sum_block_meta is not None``
        check is WRONG: a ``Sum`` term whose ``meta.var_source`` is ``None``
        (a fully-collapsed ``Sum``) passes that check yet still crashes in
        :meth:`from_term` (it takes the meta branch and selects a ``None``
        Var).  This predicate mirrors the real selection so callers route
        such terms to their non-buildable collect fallback instead of
        crashing.  Cheap (attribute reads only — no collects).
        """
        meta = getattr(term, "sum_block_meta", None)
        if meta is not None:
            return getattr(meta, "var_source", None) is not None
        return getattr(term, "var_source", None) is not None

    @classmethod
    def from_rhs_chain(cls, rhs: Param) -> CoefWalkRecipe:
        """Build a Var-LESS (Param-only) recipe from a constraint RHS
        composite ``Param`` chain.

        The composite RHS Param tracks its atomic constituents in
        ``rhs._sources`` (a ``[(Param, direction)]`` list) and its folded
        constant in ``rhs._value_scalar`` — the SAME fields the canonical
        builder's RHS prune-down (``_build_canonical_matrix``) consumes.
        We carry them verbatim as ``param_sources`` / ``coef_scalar`` so
        the per-batch Var-less build reproduces that prune-down's numpy op
        sequence value-for-value.  ``var_source`` is ``None`` (no Var, no
        ``col_id`` on the RHS).

        Raises ``ValueError`` if the chain does not expose a ``_sources``
        list (single / anonymous Params have no constituent list; the
        caller's bounded positional path / merged-lazy collect handles
        those — they never reach here).
        """
        sources = rhs._sources if isinstance(rhs._sources, list) else None
        if sources is None:
            raise ValueError(
                "CoefWalkRecipe.from_rhs_chain requires a composite RHS "
                "Param tracking its atomic constituents via _sources; the "
                "given Param has _sources=None (single / anonymous chain)."
            )
        return cls(
            var_source=None,
            param_sources=list(sources),
            coef_scalar=rhs._value_scalar,
            param_only=True,
        )


class CoefBatch:
    """One batch's ``(rid, col_id, coef)`` numpy triple handed to reducers.

    ``rid`` is the absolute Layer-2 row index (``base_row + local _rid``)
    in *constraint* mode, or an all-``-1`` array in *column* mode (the
    objective row carries no row factor).  All three arrays have the same
    length and are paired index-for-index.
    """

    __slots__ = ("rid", "col_id", "coef")

    def __init__(
        self, rid: np.ndarray, col_id: np.ndarray, coef: np.ndarray
    ) -> None:
        self.rid = rid
        self.col_id = col_id
        self.coef = coef

    def __len__(self) -> int:
        return int(self.coef.size)


# ---------------------------------------------------------------------------
# Reducer protocol + concrete reducers


class Reducer:
    """Stateful streaming reducer interface.

    A reducer is :meth:`init`-ialised once, fed every batch via
    :meth:`update`, and produces its result via :meth:`finalize`.  All
    reducers MUST combine batches associatively so the result is
    batch-invariant (exactly for min/max; up to FP reassociation for the
    log2-sum histogram).
    """

    def init(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def update(
        self, rid_arr: np.ndarray, col_id_arr: np.ndarray, coef_arr: np.ndarray
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def finalize(self) -> Any:  # pragma: no cover - interface
        raise NotImplementedError


def _scaled_abs(
    rid_arr: np.ndarray,
    col_id_arr: np.ndarray,
    coef_arr: np.ndarray,
    scale: tuple[np.ndarray | None, int, np.ndarray | None],
) -> np.ndarray:
    """Reproduce the side-vectors-on magnitude expression EXACTLY:

        vals = coef * |l2_rf[base_row + rid]|     # only if l2_rf and rid>=0
        vals *= |l2_cf[col_id]|                    # only if l2_cf is not None

    Mirrors ``_ranges._reduce_lhs_block`` / the cost-loop ``coef *
    |_l2_cf[col_id]|``.  In column mode ``rid`` is ``-1`` (no row factor),
    so the row-factor multiply is skipped (the objective has no row in the
    row-factor vector).  Returns the per-row ``|scaled coef|`` array (NOT
    yet finite/non-zero masked — each reducer applies its own mask).
    """
    l2_rf, base_row, l2_cf = scale
    vals = coef_arr.astype(np.float64, copy=True)
    if l2_rf is not None:
        # Constraint mode: rid is the local _rid (>= 0).  Column mode:
        # rid is -1 (no row factor) — guard so we never index l2_rf[-1].
        if rid_arr.size and rid_arr.min() >= 0:
            vals = vals * np.abs(l2_rf[base_row + rid_arr])
    if l2_cf is not None:
        vals = vals * np.abs(l2_cf[col_id_arr])
    return np.abs(vals)


class MinMaxAbsReducer(Reducer):
    """Running ``(min, max)`` of ``|scaled coef|`` over finite non-zero
    entries — reproducing ``_ranges._reduce_abs`` applied to the
    scale-multiplied coefficient stream.

    Order-independent (min/max are associative + commutative), so the
    finalized ``(lo, hi)`` is BYTE-IDENTICAL to a whole-collect
    ``_reduce_abs`` over the fully-materialised chain, for any
    ``batch_rows``.  Returns ``(None, None)`` when no finite non-zero
    entry was seen (the same sentinel ``_reduce_abs`` uses for an empty
    reduction).
    """

    def __init__(
        self, scale: tuple[np.ndarray | None, int, np.ndarray | None]
    ) -> None:
        self._scale = scale
        self._lo = math.inf
        self._hi = 0.0

    def init(self) -> None:
        self._lo = math.inf
        self._hi = 0.0

    def update(
        self, rid_arr: np.ndarray, col_id_arr: np.ndarray, coef_arr: np.ndarray
    ) -> None:
        if coef_arr.size == 0:
            return
        a = _scaled_abs(rid_arr, col_id_arr, coef_arr, self._scale)
        # Same finite & non-zero mask as ``_reduce_abs``.
        mask = np.isfinite(a) & (a != 0)
        if not mask.any():
            return
        m = a[mask]
        lo = float(m.min())
        hi = float(m.max())
        if lo < self._lo:
            self._lo = lo
        if hi > self._hi:
            self._hi = hi

    def finalize(self) -> tuple[float | None, float | None]:
        if self._hi == 0.0:
            return None, None
        return self._lo, self._hi


# Per-bucket accumulator tuple: (sum_log2, count, abs_min, abs_max).
_HistAcc = tuple[float, int, float, float]


class Log2HistogramReducer(Reducer):
    """Per-bucket ``(sum log2|scaled coef|, count, abs_min, abs_max)``
    accumulator, keyed by a caller-supplied ``col_id -> bucket-key``
    classification.

    This is the skeleton FlexTool's ``bucket_coefficients`` will drive in
    a later step: each LP column maps (via ``classify``) to a bucket key
    (in FlexTool, a ``QuantityType``), and every nonzero finite scaled
    coefficient magnitude folds into that bucket's accumulator with the
    same ``(log2_sum, count, min, max)`` packing FlexTool uses.

    ``classify`` is ``col_id (int) -> bucket key (Hashable) | None``.  A
    ``None`` classification drops the entry (mirrors ``bucket_coefficients``
    skipping columns with no registered family).  Entries with a
    non-positive or non-finite magnitude are skipped (``log2`` would be
    ``-inf`` / ``nan``), matching ``bucket_coefficients``' guard.

    Batch combination
    ------------------
    ``count`` / ``min`` / ``max`` combine exactly across batches.  The
    ``sum log2`` is the sum of per-batch partial sums — associative, so it
    matches a whole-collect histogram to within floating-point
    reassociation (and EXACTLY for a single batch).  See module docstring.
    """

    def __init__(
        self,
        scale: tuple[np.ndarray | None, int, np.ndarray | None],
        classify: Callable[[int], Hashable | None],
    ) -> None:
        self._scale = scale
        self._classify = classify
        self._acc: dict[Hashable, _HistAcc] = {}

    def init(self) -> None:
        self._acc = {}

    def update(
        self, rid_arr: np.ndarray, col_id_arr: np.ndarray, coef_arr: np.ndarray
    ) -> None:
        if coef_arr.size == 0:
            return
        a = _scaled_abs(rid_arr, col_id_arr, coef_arr, self._scale)
        # Drop non-finite / non-positive magnitudes (log2 undefined) —
        # same guard ``bucket_coefficients`` applies.
        mask = np.isfinite(a) & (a > 0)
        if not mask.any():
            return
        a = a[mask]
        cids = col_id_arr[mask]
        logs = np.log2(a)
        # Per-batch grouping by bucket key.  We classify the (typically
        # few) DISTINCT col_ids in this batch, then accumulate every row's
        # (log2, magnitude) into its bucket via a single vectorized
        # segmented reduction.  ``classify`` runs per DISTINCT col_id
        # (cheap); the per-row work is O(n) bincount + ufunc.at, never the
        # old O(n_uniq x batch_n) per-uniq full-mask loop.
        uniq_cids, inv = np.unique(cids, return_inverse=True)
        bucket_of_uniq = [self._classify(int(c)) for c in uniq_cids.tolist()]
        # Map each DISTINCT non-None bucket key to a dense int index; a
        # ``None`` classification gets -1 so its rows are dropped below
        # (mirrors ``bucket_coefficients`` skipping unregistered families).
        key_to_idx: dict[Hashable, int] = {}
        keys_in_order: list[Hashable] = []
        uniq_bidx = np.full(uniq_cids.size, -1, dtype=np.int64)
        for u_idx, bkey in enumerate(bucket_of_uniq):
            if bkey is None:
                continue
            j = key_to_idx.get(bkey)
            if j is None:
                j = len(keys_in_order)
                key_to_idx[bkey] = j
                keys_in_order.append(bkey)
            uniq_bidx[u_idx] = j
        nb = len(keys_in_order)
        if nb == 0:
            return
        # One bucket index per surviving row; drop rows whose col_id
        # classified to None.
        row_bidx = uniq_bidx[inv]
        keep = row_bidx >= 0
        if not keep.all():
            row_bidx = row_bidx[keep]
            logs = logs[keep]
            a = a[keep]
        # Segmented reduction over dense bucket indices.  bincount (count,
        # log2-sum) and minimum/maximum.at (abs-min/max) are order-free, so
        # count/min/max stay byte-identical to the per-uniq loop; the
        # log2-sum differs only by FP reassociation.
        slog = np.bincount(row_bidx, weights=logs, minlength=nb)
        cnt = np.bincount(row_bidx, minlength=nb).astype(np.int64)
        bmin = np.full(nb, np.inf, dtype=np.float64)
        bmax = np.full(nb, -np.inf, dtype=np.float64)
        np.minimum.at(bmin, row_bidx, a)
        np.maximum.at(bmax, row_bidx, a)
        for bkey, j in key_to_idx.items():
            if cnt[j] == 0:
                # Empty bucket — never fold (matches old ``if not sel.any``).
                continue
            self._fold(
                bkey,
                float(slog[j]),
                int(cnt[j]),
                float(bmin[j]),
                float(bmax[j]),
            )

    def _fold(
        self, bkey: Hashable, slog: float, cnt: int, amin: float, amax: float
    ) -> None:
        prev = self._acc.get(bkey)
        if prev is None:
            self._acc[bkey] = (slog, cnt, amin, amax)
            return
        p_slog, p_cnt, p_min, p_max = prev
        self._acc[bkey] = (
            p_slog + slog,
            p_cnt + cnt,
            min(p_min, amin),
            max(p_max, amax),
        )

    def finalize(self) -> dict[Hashable, _HistAcc]:
        return dict(self._acc)


# ---------------------------------------------------------------------------
# The walk


def _resolve_spine_rids(
    spine: pl.DataFrame,
    base_row: int,
) -> tuple[pl.DataFrame, str]:
    """Return the spine annotated for batching, plus the mode.

    * If the spine carries ``col_id`` (and no ``_rid``) it is a
      *column* spine (objective / bare-Var); we keep it as-is.
    * Otherwise it is a *constraint* spine; we attach an ``_rid``
      int-range (the local row index) so each batch slice's
      ``base_row + _rid`` indexes the row factor.

    ``base_row`` only matters for the row-factor index applied later by
    the reducer's ``scale`` — the spine just carries the local ``_rid``.
    """
    cols = spine.columns
    if "col_id" in cols and "_rid" not in cols:
        return spine, "column"
    if "_rid" not in cols:
        spine = spine.with_columns(
            _rid=pl.int_range(0, spine.height, dtype=pl.Int64)
        )
    return spine, "constraint"


class _Hoist:
    """Per-term batch-INVARIANT state, computed ONCE before the batch loop
    and threaded into every per-batch builder so the loop is O(n), not
    O(n²).

    The block-COO chain math depends on the batch's row set ONLY through
    which seed rows survive the row_index prune; the following pieces do
    NOT depend on the batch at all and so were redundantly recomputed every
    batch by the original per-batch builders:

    * ``spec`` — the block-COO classification (dense/non-dense split, on,
      keep …).  A function of ``var.dims``, ``axis_cols`` and the declared
      ``dense_axes`` only — never the batch rows.  Classifying once also
      makes the per-batch builder's only work the bounded chain build.
    * ``verified`` — the dense-axis sort contract is a property of the WHOLE
      Var frame (``_verify_dense_sorted`` collects + struct-scans the entire
      frame).  Verifying it once per term, not once per batch, removes the
      single biggest O(n²) term (a full-frame collect + ``is_sorted`` scan
      per batch).
    * ``dense_param_vectors`` — ``{id(atomic): sorted-dense value array}``
      for every dense-only Param (``shared == dense_dims``).  Each such
      vector is tiled identically onto every block of every batch, so it is
      collected ONCE and threaded into the engine builders via their
      ``dense_param_vectors`` kwarg (they tile the cached buffer instead of
      re-running ``atomic.lazy.collect().sort()``).
    * ``col_seed`` / ``col_coef`` (column / objective mode ONLY) — the whole
      Var seed and its whole-frame ``(col_id, coef)`` product, computed once
      in seed order.  Each batch is then a positional slice of these arrays
      — NO per-batch ``var.frame`` semi-join + collect, NO full scan, no
      ``_rid`` (the objective row carries no row factor).

    Memory bound
    ------------
    Everything held here is column / low-dim scale, NOT the wide product:

    * ``spec`` is a tiny dict; ``verified`` is a bool.
    * ``dense_param_vectors`` holds one array of length ``n_dense`` per
      dense-only Param — the dense suffix cardinality (e.g. ``|d|·|t|``),
      bounded and shared across all blocks, NOT ``n_lead · n_dense``.
    * ``col_seed`` / ``col_coef`` are column-scale (one entry per Var cell =
      one LP column) — the SAME order as the objective itself, which the
      caller already holds.

    The WIDE ``Var × Param`` per-cell product is NOT hoisted: it is still
    built and released per batch by the constraint-mode builders, so the
    peak stays bounded by ``batch_rows`` exactly as before.
    """

    __slots__ = (
        "mode",
        "spec",
        "verified",
        "dense_param_vectors",
        "col_coef_cids",
        "col_coef",
        "col_coef_order",
    )

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.spec: dict | None = None
        self.verified = False
        self.dense_param_vectors: dict[int, np.ndarray] | None = None
        # Column / objective mode: the whole-frame (col_id, coef) product
        # (one entry per LP column) + a sort order over col_id for the
        # per-batch positional lookup.  None in constraint / param-only mode.
        self.col_coef_cids: np.ndarray | None = None
        self.col_coef: np.ndarray | None = None
        self.col_coef_order: np.ndarray | None = None


def _dense_param_vectors(
    param_sources: Sequence[tuple[Param, int]],
    var_dims: Sequence[str],
    dense_dims: Sequence[str],
) -> dict[int, np.ndarray]:
    """Collect, ONCE, the sorted-by-``dense_dims`` value array for every
    dense-only Param in the chain (``shared == dense_dims``).

    These are precisely the Params the engine builders tile across blocks in
    their dense-only case; the vector is batch-invariant, so we collect it
    here once and thread it via ``dense_param_vectors``.  Params that are not
    dense-only (lead-only, lead+dense, scalar, foreign) are skipped — the
    builder handles them per-batch via its other (cheap, batch-keyed) cases.
    A Param whose collect does not yield a clean dense vector is omitted, so
    the builder falls back to its inline collect (and its own length guard).
    """
    dense_list = list(dense_dims)
    var_list = list(var_dims)
    out: dict[int, np.ndarray] = {}
    for atomic, _direction in param_sources:
        shared = [d for d in var_list if d in atomic.dims]
        if shared != dense_list:
            continue
        if id(atomic) in out:
            continue
        atomic_df = atomic.lazy.collect().sort(dense_list)
        out[id(atomic)] = (
            atomic_df["value"].to_numpy().astype(np.float64, copy=False)
        )
    return out


def _build_constraint_batch_triple(
    batch_over: pl.DataFrame,
    axis_cols: list[str],
    recipe: CoefWalkRecipe,
    hoist: _Hoist,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the ``(_rid, col_id, coef)`` triple for ONE batch of the
    constraint spine, reusing the engine's block-COO builders.

    ``batch_over`` is a slice of the constraint ``over`` grid carrying an
    ``_rid`` column (the LOCAL row index for the family).  We hand it to
    the builders as ``row_index_lf``; their final inner-join on the term's
    ``on`` keys restricts the emitted product to exactly the batch's rows,
    and they prune the Var/Param leaves to the batch's key set, so the
    per-batch peak is bounded by the batch (not the full product).

    Dispatch mirrors ``_ranges._ranges_via_streaming``'s LHS arm:

    * Non-Sum recipe (``sum_block_meta is None``): classify via
      :func:`_block_coo_classify`; on a hit build via
      :func:`_build_block_coo_plan` (positional, falling back internally
      to the order-preserving joined backstop).
    * Sum recipe (relabel or combining): classify via
      :func:`_sum_block_coo_classify`; build via
      :func:`_build_sum_block_coo_plan` (delegates to the relabel arm when
      ``reduce_dims ⊆ var.dims``).
    * Anything the block-COO classifiers decline: rebuild via the
      always-correct merged-lazy prune-down :func:`_build_lhs_pruned_plan`
      and collect (still bounded — the chain prunes against the batch
      row_index key set).

    Every arm produces the SAME ``(_rid, col_id, coef)`` per cell as the
    whole-frame builders would for those rows, because the coef math is
    identical and the row set is the batch's rows; batching only changes
    WHICH rows each call emits, never their values.
    """
    var = recipe.var_source
    # Each builder computes its own ``on`` (the relabel/Sum arms may carry
    # map-extras; for a non-Sum term ``on ⊆ var.dims``), so we only need
    # the batch's row_index here.
    row_index_lf = batch_over.lazy()

    df: pl.DataFrame | None = None
    # ``hoist.spec`` was classified ONCE per term (batch-invariant); the
    # dense-axis sort contract was verified ONCE (``hoist.verified``); the
    # dense-only Param vectors were collected ONCE
    # (``hoist.dense_param_vectors``).  This builder only does the bounded,
    # batch-keyed chain build here.
    spec = hoist.spec

    if recipe.sum_block_meta is None:
        blk_on = [d for d in var.dims if d in axis_cols]
        if spec is not None and recipe.where_map_frames is None:
            df = _build_block_coo_plan(
                row_index_lf,
                axis_cols,
                var,
                recipe.param_sources,
                blk_on,
                recipe.coef_scalar,
                recipe.where_frames,
                spec,
                dense_param_vectors=hoist.dense_param_vectors,
            )
    else:
        meta = recipe.sum_block_meta
        keep_on = [d for d in meta.keep if d in axis_cols]
        if spec is not None:
            try:
                df = _build_sum_block_coo_plan(
                    row_index_lf,
                    axis_cols,
                    meta,
                    keep_on,
                    spec,
                    dense_param_vectors=hoist.dense_param_vectors,
                )
            except _SumBlockCooFallback:
                df = None
        if df is None:
            # Sum block-COO declined / fell back.  The reduced coef cannot
            # be reconstructed from the seed + chain (e.g. the map-effect
            # Where was baked eagerly, clearing ``where_map_frames``), so
            # emit the term's OWN reduced lazy plan verbatim — the SAME
            # backstop the engine uses — attaching ``_rid`` by inner-join.
            df = _reduced_lazy_collect(row_index_lf, axis_cols, recipe)

    if df is None:
        # Non-Sum block-COO declined — always-correct merged-lazy
        # prune-down backstop.  Bake the map-effect frames so ``on`` (which
        # may now include map extras) finds its columns, then rebuild +
        # collect.
        df = _lhs_prune_down_collect(row_index_lf, axis_cols, recipe)

    if df.height == 0:
        z = np.empty(0, dtype=np.float64)
        zi = np.empty(0, dtype=np.int64)
        return zi, zi, z
    rids = df["_rid"].to_numpy().astype(np.int64)
    cids = df["col_id"].to_numpy().astype(np.int64)
    coef = df["coef"].to_numpy().astype(np.float64)
    return rids, cids, coef


def _lhs_prune_down_collect(
    row_index_lf: pl.LazyFrame,
    axis_cols: list[str],
    recipe: CoefWalkRecipe,
) -> pl.DataFrame:
    """Rebuild the LHS term via the engine's merged-lazy prune-down chain
    and collect ``(_rid, col_id, coef)`` for the batch.

    The prune-down helper takes a ``var_source`` + the full Param chain +
    the deferred Where frames and rebuilds ``row_index → Var → P1 → P2 …``
    bounded by the ``row_index_lf`` key set, so each batch's collect is
    bounded by the batch.  ``on`` is the intersection of the post-map
    open dims and the axis columns (the same key the engine uses to attach
    ``_rid``).  Used as the always-correct backstop for any shape the
    block-COO classifiers decline (sparse / non-dense, map-Where the
    block arm rejects, etc.).
    """
    var = recipe.var_source
    # Post-map open dims for the on-key: var dims plus any map extras.
    open_dims = list(var.dims)
    if recipe.where_map_frames is not None:
        for mf, extras in recipe.where_map_frames:
            for c in mf.collect_schema().names():
                if c in extras and c not in open_dims:
                    open_dims.append(c)
    on = [d for d in open_dims if d in axis_cols]
    plan = _build_lhs_pruned_plan(
        row_index_lf,
        axis_cols,
        var,
        recipe.param_sources,
        on,
        recipe.coef_scalar,
        recipe.where_frames,
        recipe.where_map_frames,
    )
    return plan.select("_rid", "col_id", "coef").collect()


def _deferred_where_map_frames(
    reduced: pl.LazyFrame,
    reduced_dims: tuple[str, ...],
    where_map_frames: tuple[tuple[pl.LazyFrame, frozenset[str]], ...] | None,
) -> tuple[tuple[pl.LazyFrame, frozenset[str]], ...] | None:
    """Filter ``where_map_frames`` to the entries still DEFERRED on the
    reduced lazy plan — the post-``Sum`` map-effect Wheres whose introduced
    dim(s) the reduced plan does not yet physically carry.

    ``recipe.where_map_frames`` (taken from the :class:`SumBlockMeta`)
    carries BOTH:

    * **pre-Sum** map frames — baked into the seed and then reduced by the
      ``Sum``'s group_by, so their extras are already accounted for in
      ``reduced`` (either surviving as a kept column, or summed out and no
      longer claimed by ``reduced_dims``); re-applying their inner-join
      would FAN OUT rows (a double-apply bug), and
    * **post-Sum** map frames — recorded by a map-effect :func:`Where`
      AFTER the ``Sum`` (the D1 forwarding path).  Their extras are claimed
      by the term's ``dims`` (``reduced_dims``) but NOT yet physically in
      ``reduced``'s schema; these are the ONLY ones the reduced plan must
      still bake to reconstruct the block-COO path's coefficient support.

    An entry is deferred iff it introduces at least one extra that is BOTH
    claimed by ``reduced_dims`` AND absent from ``reduced``'s physical
    schema.  A pre-Sum map whose extra survived sits in the schema; one
    whose extra was summed out is not in ``reduced_dims`` — either way it is
    correctly skipped, so no transform already baked into ``reduced`` is
    re-applied.  Returns ``None`` when nothing is deferred (the common
    case), so :func:`_apply_where_map_frames` is a no-op.
    """
    if where_map_frames is None:
        return None
    schema_cols = set(reduced.collect_schema().names())
    rdim_set = set(reduced_dims)
    deferred = tuple(
        (mf, extras)
        for (mf, extras) in where_map_frames
        if any(c in rdim_set and c not in schema_cols for c in extras)
    )
    return deferred or None


def _reduced_lazy_collect(
    row_index_lf: pl.LazyFrame,
    axis_cols: list[str],
    recipe: CoefWalkRecipe,
) -> pl.DataFrame:
    """Attach ``_rid`` to the term's OWN reduced lazy plan and collect
    ``(_rid, col_id, coef)`` for the batch.

    Used as the Sum backstop when the Sum block-COO classifier declines or
    raises :class:`_SumBlockCooFallback`: the reduced coefficient (the
    post-``Sum`` group_by result) is materialised in
    ``recipe.reduced_lazy`` (the term's ``.lazy``).  We inner-join the
    batch ``row_index_lf`` on the kept axis dims to attach ``_rid``,
    restricting the output (and the collect) to the batch's rows.  Reduced
    groups are already complete in the reduced plan, so no group is split
    across batches.

    Deferred recipe transforms
    --------------------------
    ``reduced_lazy`` carries the coefficient values, but a map-effect
    :func:`Where` applied AFTER the ``Sum`` (the D1 forwarding path) leaves
    its introduced dim(s) DEFERRED in ``recipe.where_map_frames`` — they
    are claimed by ``reduced_dims`` (so the ``_rid`` ``on``-key references
    them) yet NOT physically present in ``reduced_lazy``.  We must
    reconstruct the SAME coefficient support the block-COO path would, so
    before the ``_rid`` join we:

    * bake the deferred map-effect frames via :func:`_apply_where_map_frames`
      (the SAME helper the block-COO Sum builders use — reused, not
      reimplemented) to MATERIALISE the introduced dim(s); only the
      genuinely-deferred frames are applied (see
      :func:`_deferred_where_map_frames`) so a pre-``Sum`` map already baked
      into ``reduced_lazy`` is never re-applied (which would fan rows out),
      and
    * apply ``where_frames`` via :func:`_apply_where_frames` (an
      order-preserving, idempotent semi-join on the columns physically
      present), so a pure-filter :func:`Where`-after-``Sum`` constrains the
      same row set the block-COO path would.  Filters already reflected in
      ``reduced_lazy`` semi-join to a no-op, so applying the full set is
      safe.

    With the deferred dims materialised this is byte-identical to the
    block-COO Sum path's emission for the batch's rows.
    """
    reduced = recipe.reduced_lazy
    if reduced is None:
        raise ValueError(
            "Sum recipe declined block-COO but carries no reduced_lazy "
            "fallback plan; build the recipe via CoefWalkRecipe.from_term "
            "so the term's reduced lazy plan is captured."
        )
    rdims = recipe.reduced_dims or ()
    # Apply the deferred pure-filter Wheres (idempotent semi-join on the
    # columns currently present) BEFORE the map bake, mirroring the Sum
    # builders' ``where_frames`` → ``where_map_frames`` order.
    reduced = _apply_where_frames(reduced, rdims, recipe.where_frames)
    # Bake ONLY the genuinely-deferred map-effect frames so the introduced
    # dim(s) (e.g. ``n``) the ``_rid`` ``on``-key needs are materialised.
    deferred_map = _deferred_where_map_frames(
        reduced, rdims, recipe.where_map_frames
    )
    if deferred_map is not None:
        reduced, _red_dims = _apply_where_map_frames(reduced, rdims, deferred_map)
    on = [d for d in rdims if d in axis_cols]
    ri_a, red_a = _align_enum_join_keys(row_index_lf, reduced, on)
    plan = ri_a.join(red_a, on=on, how="inner").select("_rid", "col_id", "coef")
    return plan.collect()


def _column_whole_product(
    seed: pl.DataFrame,
    recipe: CoefWalkRecipe,
    spec: dict | None,
    dense_param_vectors: dict[int, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the WHOLE-frame ``(col_id, coef)`` per-cell product for a
    *column* spine (objective / bare-Var) ONCE, in seed order.

    The objective spine IS the Var seed, so the whole product is
    batch-invariant: every batch is a positional slice of it.  We feed the
    pre-collected ``seed`` (the whole Var frame, carrying ``(*var_dims,
    col_id)``) to :func:`_build_block_coo_plan` as the identity row_index
    (``on = var.dims`` — each seed row maps to itself), so the builder emits
    one ``(col_id, coef)`` per Var cell.  The dense-only Param vectors are
    threaded in pre-collected (``dense_param_vectors``) so even this single
    whole-frame build does no redundant collect.

    Returns ``(col_id, coef)`` numpy arrays paired index-for-index, one
    entry per Var cell (= one LP column).  Computed ONCE per term; the batch
    loop then maps each batch's ``col_id`` against these via a sorted lookup
    — NO per-batch ``var.frame`` semi-join / collect / scan.
    """
    var = recipe.var_source
    var_dims = list(var.dims)
    if not recipe.param_sources:
        # Bare Var (no Param chain): coef is the constant coef_scalar per
        # cell.  No product to build.
        cids = seed["col_id"].to_numpy().astype(np.int64)
        coef = np.full(cids.size, recipe.coef_scalar, dtype=np.float64)
        return cids, coef

    # Pure-relabel Sum objective: the term's OWN reduced lazy plan already
    # carries exactly one ``(col_id, coef)`` per LP column.  For a relabel
    # Sum (``reduce_dims ⊆ var.dims``, no map-effect Where frames) ``col_id``
    # is 1:1 with Var cells, so the reduced plan's ``group_by(col_id).sum()``
    # is over single-element groups — the reduced ``coef`` EQUALS the per-cell
    # product coef the joined whole-product build would emit.  Read it
    # directly: no identity self-join, no rebuild of the ~unreduced
    # ``Var × Param`` whole product through ``_build_block_coo_plan``'s joined
    # branch (the 40 s/term hot spot).  Same ``(col_id, coef)`` the
    # ``_ref_histogram_column`` reference / the ranges reduced-plan collect
    # read.  Strictly less memory: ~one row per LP column vs the whole
    # product.  Non-relabel column terms (bare Var with a Param chain, or any
    # recipe failing this gate) fall through to the identity-row_index build
    # below, unchanged.
    if (
        recipe.reduced_lazy is not None
        and recipe.sum_block_meta is not None
        and recipe.where_map_frames is None
        and set(recipe.reduced_dims or ()).issubset(set(var_dims))
    ):
        df = recipe.reduced_lazy.select("col_id", "coef").collect()
        return (
            df["col_id"].to_numpy().astype(np.int64),
            df["coef"].to_numpy().astype(np.float64),
        )

    axis_cols = list(var_dims)
    # Identity row_index: the whole seed keyed on the var dims (every row
    # joins to itself).  The injected _rid is ignored (column mode reports
    # rid = -1); we read only (col_id, coef).
    row_index = seed.select(*var_dims).with_columns(
        _rid=pl.int_range(0, seed.height, dtype=pl.Int64)
    )
    blk_on = list(var_dims)
    df: pl.DataFrame | None = None
    if spec is not None and recipe.where_map_frames is None:
        df = _build_block_coo_plan(
            row_index.lazy(),
            axis_cols,
            var,
            recipe.param_sources,
            blk_on,
            recipe.coef_scalar,
            recipe.where_frames,
            spec,
            dense_param_vectors=dense_param_vectors,
        )
    if df is None:
        df = _lhs_prune_down_collect(row_index.lazy(), axis_cols, recipe)

    if df.height == 0:
        zi = np.empty(0, dtype=np.int64)
        return zi, np.empty(0, dtype=np.float64)
    cids = df["col_id"].to_numpy().astype(np.int64)
    coef = df["coef"].to_numpy().astype(np.float64)
    return cids, coef


def _build_column_batch_triple(
    batch_seed: pl.DataFrame,
    recipe: CoefWalkRecipe,
    hoist: _Hoist,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the ``(rid, col_id, coef)`` triple for ONE batch of a *column*
    spine by a positional LOOKUP into the hoisted whole-frame product — NO
    per-batch ``var.frame`` semi-join, collect, sorted-scan, or chain build.

    ``hoist.col_coef`` is the whole-frame ``coef`` (one per LP column),
    aligned to ``hoist.col_seed``'s ``col_id``.  Each batch carries a slice
    of the objective spine's ``col_id``; we map those ``col_id`` to their
    coef via a sorted index into the hoisted arrays.  ``col_id`` is unique
    per Var cell (one LP column), so the map is 1:1 and exact.  ``rid`` is
    reported ``-1`` (the objective row carries no row factor).
    """
    batch_cids = batch_seed["col_id"].to_numpy().astype(np.int64)
    if batch_cids.size == 0:
        z = np.empty(0, dtype=np.float64)
        zi = np.empty(0, dtype=np.int64)
        return zi, zi, z

    full_cids = hoist.col_coef_cids
    full_coef = hoist.col_coef
    order = hoist.col_coef_order
    if full_cids is None or full_cids.size == 0:
        z = np.empty(0, dtype=np.float64)
        zi = np.empty(0, dtype=np.int64)
        return zi, zi, z

    # Sorted-index lookup of the batch col_ids into the whole-frame product.
    # ``clip`` keeps searchsorted in-bounds; the equality check below drops
    # any batch col_id NOT in the product — exactly the cells a sparse Param
    # dropped from the per-cell chain (the original per-batch builder emitted
    # height < batch for those), so the surviving set matches byte-for-byte.
    pos = np.searchsorted(full_cids, batch_cids, sorter=order)
    pos_clipped = np.clip(pos, 0, order.size - 1)
    src = order[pos_clipped]
    hit = full_cids[src] == batch_cids
    if not hit.all():
        batch_cids = batch_cids[hit]
        src = src[hit]
    coef = full_coef[src]
    rid = np.full(batch_cids.size, -1, dtype=np.int64)
    return rid, batch_cids, coef


def _build_param_only_batch_triple(
    batch_over: pl.DataFrame,
    axis_cols: list[str],
    recipe: CoefWalkRecipe,
    dense_axes: tuple[str, ...] | None,
    dense_param_vectors: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the ``(_rid, coef)`` triple for ONE batch of a Var-LESS
    (Param-only) constraint chain — the RHS Param-product readout.

    ``batch_over`` is a slice of the constraint ``over`` grid carrying an
    ``_rid`` column (the LOCAL row index for the family).  There is no Var
    / no ``col_id``: the spine IS ``over``.  We seed ``coef =
    coef_scalar`` over the batch rows and multiply the Param chain onto
    them, in ``recipe.param_sources`` order.  Returns ``(_rid, col_id,
    coef)`` where ``col_id`` is an all-``-1`` array (no col factor on the
    RHS; the reducer is given ``l2_cf=None`` and never indexes it).

    Two builds, byte-identical to each other and to the whole-collect:

    * **positional fast path** — when the batch over slice is
      dense-complete (the same dense-suffix contract block-COO's LHS seed
      and the inline ``_rhs_chain_bounded_coef`` obey), align each atomic
      by the SAME three cases ``_rhs_chain_bounded_coef`` uses: lead-only
      ``np.repeat``, dense-only ``np.tile``, lead+dense positional
      ``maintain_order`` left-join.  Same numpy op sequence (``*`` for
      ``direction >= 0`` / ``/`` for ``< 0``, seeded with ``coef_scalar``)
      ⇒ value-for-value identical to the canonical RHS prune-down.
    * **batched prune-down fallback** — when the slice declines the
      positional regime (not dense-complete, an atomic dim outside the
      over grid, a null after alignment), seed the batch over with
      ``value=coef_scalar`` and left-join each atomic on its
      ``dims ∩ axis_cols`` (semi-join pre-pruned to the batch keys),
      multiplying / dividing the running ``value`` — the SAME merged-lazy
      prune-down the engine's RHS path runs, but bounded to the batch's
      rows (NO dense suffix needed; this generality bounds every shape the
      positional path declines).  Then read ``value`` back in ``_rid``
      order.
    """
    n = int(batch_over.height)
    if n == 0:
        z = np.empty(0, dtype=np.float64)
        zi = np.empty(0, dtype=np.int64)
        return zi, zi, z

    coef = _param_only_positional(
        batch_over, axis_cols, recipe, dense_axes, dense_param_vectors
    )
    if coef is not None:
        # Positional build keeps the batch's row order, so ``_rid`` pairs
        # index-for-index with ``coef``.
        rids = batch_over["_rid"].to_numpy().astype(np.int64)
    else:
        rids, coef = _param_only_prune_down(batch_over, axis_cols, recipe)

    col_id = np.full(coef.size, -1, dtype=np.int64)
    return rids, col_id, coef.astype(np.float64, copy=False)


def _param_only_positional(
    batch_over: pl.DataFrame,
    axis_cols: list[str],
    recipe: CoefWalkRecipe,
    dense_axes: tuple[str, ...] | None,
    dense_param_vectors: dict[int, np.ndarray] | None = None,
) -> np.ndarray | None:
    """Positional Param-only product over the batch ``over`` slice — the
    reuse of ``_rhs_chain_bounded_coef``'s three alignment cases.

    Returns the per-row ``coef`` numpy array (length ``batch_over.height``,
    one entry per ``_rid`` in the slice's row order), or ``None`` to
    decline (caller falls to :func:`_param_only_prune_down`).  Declining is
    always safe — a false decline only changes which (byte-identical) build
    produces the batch.

    ``dense_param_vectors`` is the hoisted ``{id(atomic): sorted-dense value
    array}`` cache (batch-invariant); the dense-only case tiles the cached
    buffer instead of re-running ``atomic.lazy.collect().sort()`` per batch.
    """
    if not dense_axes or _block_coo_disabled():
        return None
    sources = recipe.param_sources
    if not sources:
        return None
    dense_dims = list(dense_axes)
    if len(dense_dims) > len(axis_cols):
        return None
    if tuple(axis_cols[-len(dense_dims):]) != tuple(dense_dims):
        return None
    non_dense_dims = [d for d in axis_cols if d not in set(dense_dims)]
    for atomic, _direction in sources:
        for d in atomic.dims:
            if d not in axis_cols:
                return None

    # Verify the dense-sort contract on the batch over slice (loud error on
    # violation — same guard block-COO's LHS seed runs).
    _verify_dense_sorted(batch_over, non_dense_dims, dense_dims, None)

    n = int(batch_over.height)
    n_dense = batch_over.select(dense_dims).n_unique() if dense_dims else 1
    n_lead = (
        batch_over.select(non_dense_dims).n_unique() if non_dense_dims else 1
    )
    if n != n_lead * n_dense:
        return None

    coef = np.full(n, float(recipe.coef_scalar), dtype=np.float64)
    non_dense_set = set(non_dense_dims)
    dense_set = set(dense_dims)

    lead_table = None
    if non_dense_dims:
        lead_table = batch_over.select(non_dense_dims).unique(
            maintain_order=True
        )
        if lead_table.height != n_lead:
            return None

    for atomic, direction in sources:
        shared = [d for d in axis_cols if d in atomic.dims]
        if not shared:
            f = atomic.frame
            if "value" not in f.columns or f.height == 0:
                return None
            scalar_val = float(f["value"][0])
            if direction >= 0:
                coef = coef * scalar_val
            else:
                coef = coef / scalar_val
            continue

        shared_set = set(shared)
        has_lead = bool(shared_set & non_dense_set)
        has_dense = bool(shared_set & dense_set)

        if has_lead and not has_dense:
            lt_a, at_a = _align_enum_join_keys(
                lead_table.lazy(), atomic.lazy, shared
            )
            aligned = lt_a.join(
                at_a, on=shared, how="left", maintain_order="left"
            ).collect()
            if aligned.height != n_lead or aligned["value"].null_count() > 0:
                return None
            block_vals = (
                aligned["value"].to_numpy().astype(np.float64, copy=False)
            )
            repeated = np.repeat(block_vals, n_dense)
            if direction >= 0:
                coef = coef * repeated
            else:
                coef = coef / repeated

        elif has_dense and not has_lead and shared == dense_dims:
            # Dense-only: the sorted dense vector is batch-invariant — use the
            # hoisted buffer when present, else collect+sort inline.
            dense_vals = (
                dense_param_vectors.get(id(atomic))
                if dense_param_vectors is not None
                else None
            )
            if dense_vals is None:
                atomic_df = atomic.lazy.collect().sort(dense_dims)
                if atomic_df.height != n_dense:
                    return None
                dense_vals = (
                    atomic_df["value"].to_numpy().astype(np.float64, copy=False)
                )
            elif dense_vals.size != n_dense:
                return None
            tiled = np.tile(dense_vals, n_lead)
            if direction >= 0:
                coef = coef * tiled
            else:
                coef = coef / tiled

        elif has_dense:
            grid_a, at_a = _align_enum_join_keys(
                batch_over.lazy(), atomic.lazy, shared
            )
            aligned = grid_a.join(
                at_a, on=shared, how="left", maintain_order="left"
            ).collect()
            if aligned.height != n or aligned["value"].null_count() > 0:
                return None
            vals = aligned["value"].to_numpy().astype(np.float64, copy=False)
            if direction >= 0:
                coef = coef * vals
            else:
                coef = coef / vals

        else:
            return None

    _emit_block_coo_path("rhs_positional_walk")
    return coef


def _param_only_prune_down(
    batch_over: pl.DataFrame,
    axis_cols: list[str],
    recipe: CoefWalkRecipe,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched merged-lazy prune-down for the Var-less Param chain — the
    always-correct backstop for any batch slice the positional regime
    declines.

    Mirrors the canonical builder's RHS prune-down
    (``_build_canonical_matrix``) bounded to the batch: seed the batch
    ``over`` (carrying ``_rid``) with ``value=coef_scalar``, then for each
    ``(atomic, direction)`` in chain order semi-join the atomic to the
    accumulator's key projection (so the intermediate stays bounded to the
    batch's rows) and left-join its ``value``, multiplying (``direction >=
    0``) or dividing (``< 0``) the running ``value``.  Scalar atomics fold
    their constant directly.  Reads ``value`` back in ``_rid`` order and
    ``fill_null(0.0)`` exactly as the engine does, so the result is
    byte-identical to the whole-collect.  Needs NO dense suffix — that is
    the generality that bounds the declined shapes.

    Returns ``(rids, coef)`` paired index-for-index (both read off the
    collected frame, so the ``_rid`` order matches the ``coef`` order
    regardless of how polars laid the join out).
    """
    acc = batch_over.lazy().with_columns(
        value=pl.lit(float(recipe.coef_scalar), dtype=pl.Float64)
    )
    for atomic, direction in recipe.param_sources:
        atomic_on = [d for d in atomic.dims if d in axis_cols]
        if atomic_on:
            acc_for_keys, atomic_lazy = _align_enum_join_keys(
                acc, atomic.lazy, atomic_on
            )
            keys_lazy = acc_for_keys.select(atomic_on).unique()
            keys_a, atomic_a = _align_enum_join_keys(
                keys_lazy, atomic_lazy, atomic_on
            )
            atomic_pruned = atomic_a.join(keys_a, on=atomic_on, how="semi")
            acc_a, atomic_pruned_a = _align_enum_join_keys(
                acc_for_keys, atomic_pruned, atomic_on
            )
            joined = acc_a.join(
                atomic_pruned_a,
                on=atomic_on,
                how="left",
                suffix="__rhs_chain",
            )
            if direction >= 0:
                acc = joined.with_columns(
                    value=pl.col("value") * pl.col("value__rhs_chain")
                ).drop("value__rhs_chain")
            else:
                acc = joined.with_columns(
                    value=pl.col("value") / pl.col("value__rhs_chain")
                ).drop("value__rhs_chain")
        else:
            scalar_val = float(atomic.frame["value"][0])
            if direction >= 0:
                acc = acc.with_columns(value=pl.col("value") * scalar_val)
            else:
                acc = acc.with_columns(value=pl.col("value") / scalar_val)
    out = acc.select("_rid", "value").collect()
    rids = out["_rid"].to_numpy().astype(np.int64)
    coef = out["value"].fill_null(0.0).to_numpy().astype(np.float64, copy=False)
    return rids, coef


class _NonSumTermProxy:
    """Minimal duck-typed ``_Term`` exposing the fields the block-COO
    classifier / builder read on a non-Sum term."""

    __slots__ = (
        "var_source",
        "param_sources",
        "coef_scalar",
        "where_frames",
        "where_map_frames",
        "sum_block_meta",
        "dims",
    )

    def __init__(self, recipe: CoefWalkRecipe) -> None:
        self.var_source = recipe.var_source
        self.param_sources = recipe.param_sources
        self.coef_scalar = recipe.coef_scalar
        self.where_frames = recipe.where_frames
        self.where_map_frames = recipe.where_map_frames
        self.sum_block_meta = None
        self.dims = recipe.var_source.dims


class _SumTermProxy:
    """Minimal duck-typed ``_Term`` exposing the fields the Sum block-COO
    classifier reads (``sum_block_meta`` + post-Sum ``dims`` = keep)."""

    __slots__ = ("sum_block_meta", "dims")

    def __init__(self, recipe: CoefWalkRecipe) -> None:
        meta = recipe.sum_block_meta
        self.sum_block_meta = meta
        self.dims = meta.keep


def _precompute_hoist(
    recipe: CoefWalkRecipe,
    annotated: pl.DataFrame,
    mode: str,
    axis_cols: list[str],
    dense_axes: tuple[str, ...] | None,
) -> _Hoist:
    """Compute, ONCE per term, all batch-INVARIANT state the per-batch
    builders would otherwise recompute every batch (the source of the
    O(n²)).  See :class:`_Hoist` for the memory bound (everything held here
    is column / low-dim scale, NOT the wide product).

    * Classify the block-COO spec once (a function of var.dims / axis_cols /
      dense_axes — never the batch rows).
    * Verify the dense-axis sort contract once on the WHOLE Var frame
      (``_verify_dense_sorted`` is a full-frame collect + scan; doing it per
      batch is the single biggest O(n²) term).
    * Collect the dense-only Param vectors once.
    * Column / objective mode: collect the whole Var seed and compute its
      whole-frame ``(col_id, coef)`` product once, in seed order — each batch
      then positionally maps its ``col_id`` against it.

    The ``param_only`` (Var-less RHS) mode keeps its per-batch positional
    build (its ``_verify_dense_sorted`` runs on the BATCH slice, already
    O(batch)); we still hoist its dense-only Param vectors.
    """
    hoist = _Hoist(mode)

    if recipe.param_only:
        # Var-less RHS chain: no Var frame to verify (the batch slice is
        # verified per batch, O(batch)); hoist only the dense-only Param
        # vectors keyed by the over-grid dense suffix.
        if dense_axes:
            hoist.dense_param_vectors = _dense_param_vectors(
                recipe.param_sources, axis_cols, list(dense_axes)
            )
        return hoist

    # --- Classify the block-COO spec ONCE (batch-invariant).
    if recipe.sum_block_meta is None:
        var = recipe.var_source
        blk_on = list(var.dims) if mode == "column" else [
            d for d in var.dims if d in axis_cols
        ]
        spec = _block_coo_classify(
            _NonSumTermProxy(recipe), axis_cols, blk_on, dense_axes
        )
        verify_frame = var.frame
        verify_name = getattr(var, "name", None)
        fires = spec is not None and recipe.where_map_frames is None
    else:
        meta = recipe.sum_block_meta
        keep_on = [d for d in meta.keep if d in axis_cols]
        spec = _sum_block_coo_classify(
            _SumTermProxy(recipe), axis_cols, keep_on, dense_axes
        )
        verify_frame = meta.var_source.frame
        verify_name = getattr(meta.var_source, "name", None)
        fires = spec is not None
    hoist.spec = spec

    if fires:
        # --- Verify the dense-axis sort contract ONCE on the whole frame.
        _verify_dense_sorted(
            verify_frame,
            spec["non_dense_dims"],
            spec["dense_dims"],
            verify_name,
        )
        hoist.verified = True
        # --- Collect the dense-only Param vectors ONCE.
        var_dims = spec["var_dims"]
        hoist.dense_param_vectors = _dense_param_vectors(
            recipe.param_sources, var_dims, spec["dense_dims"]
        )

    # --- Column / objective mode: compute the whole-frame product ONCE.
    # ``_column_whole_product`` uses the positional builder when ``spec``
    # fires (the verify above has run) and the prune-down fallback otherwise.
    if mode == "column":
        seed = annotated  # the objective spine IS the whole Var seed
        col_cids, col_coef = _column_whole_product(
            seed, recipe, spec, hoist.dense_param_vectors
        )
        hoist.col_coef_cids = col_cids
        hoist.col_coef = col_coef
        hoist.col_coef_order = np.argsort(col_cids, kind="stable")

    return hoist


def bounded_coefficient_walk(
    spine: pl.DataFrame | pl.LazyFrame,
    recipe: CoefWalkRecipe,
    scale: tuple[np.ndarray | None, int, np.ndarray | None],
    reducers: Iterable[Reducer],
    *,
    batch_rows: int = 1_000_000,
    dense_axes: tuple[str, ...] | None = None,
) -> list[Any]:
    """Walk a block-COO term's ``(rid/col_id, coef)`` stream in bounded
    ``batch_rows`` slices, feeding each batch to every reducer, and return
    ``[r.finalize() for r in reducers]``.

    Parameters
    ----------
    spine:
        The row/col index frame.  In *constraint* mode it is the
        constraint ``over`` grid (the builders attach ``_rid`` and emit
        ``(_rid, col_id, coef)``); in *column* mode it is ``Var.frame``
        carrying ``col_id`` (objective / bare-Var, ``rid`` reported
        ``-1``).  The mode is inferred from whether the spine carries
        ``col_id`` (and no ``_rid``).  For the Sum/relabel arm the spine
        is the constraint ``over`` — already sorted so reduction groups
        are contiguous, mirroring ``_build_sum_block_coo_relabel``'s sort
        contract.  Lazy spines are collected once (index only — cheap).
    recipe:
        The :class:`CoefWalkRecipe` (block-COO term recipe).
    scale:
        ``(l2_rf, base_row, l2_cf)`` — the Layer-2 row factor (forward),
        the family's absolute base row, and the col factor (inverse
        forward).  Any may be ``None``.  Reducers apply ``coef *
        |l2_rf[base_row + rid]| * |l2_cf[col_id]|`` exactly as the range
        readout does.
    reducers:
        Stateful :class:`Reducer` objects.  Each is :meth:`init`-ed,
        :meth:`update`-d per batch, then :meth:`finalize`-d.
    batch_rows:
        The spine slice size.  Peak memory is bounded by this (× the
        per-factor numpy buffers the block-COO chain needs for the
        slice).  ``batch_rows >= spine.height`` walks the whole spine in
        one batch; ``batch_rows == 1`` walks one spine row at a time.
        Min/max is exact for every value; the log2-sum histogram matches a
        whole-collect to within FP reassociation (exact for one batch).

    Returns
    -------
    The list of each reducer's :meth:`finalize` result, in input order.

    Carry-over / group contiguity
    ------------------------------
    For the *relabel* Sum arm (``reduce_dims ⊆ var.dims``) every reduction
    group is single-element (``col_id`` is a function of the Var
    instance), so NO group spans a batch boundary — the relabel builder
    pre-prunes its Var seed against the batch ``row_index`` key set, so it
    is genuinely bounded by the batch, and batching is exact.

    For the *combining* Sum arm (a reduced dim is NOT a Var dim — e.g. a
    map-introduced dim summed out) a single ``(*keep, col_id)`` reduce
    group draws coef terms from several Var cells, and
    :func:`_build_sum_block_coo_plan`'s combining path reduces over the
    FULL unreduced product seeded from the WHOLE Var frame (only Step 6
    attaches ``_rid`` from ``row_index``).  Restricting ``row_index`` to a
    batch therefore would NOT bound that builder's internal peak AND could
    split a reduce group whose contributing Var cells land in different
    batches — silently wrong.  So the combining arm is processed in a
    SINGLE whole-spine call regardless of ``batch_rows``: min/max stays
    BYTE-IDENTICAL to the whole-collect (it IS the whole collect for that
    term) and the histogram is exact (single batch).  Its memory envelope
    equals the existing production combining builder's — bounded by the
    Var grid + the unreduced product, the same as today.  This is the one
    shape the per-batch slicing does not tighten; it is flagged here
    rather than producing a split-group error.
    """
    spine_df = spine.collect() if isinstance(spine, pl.LazyFrame) else spine
    if batch_rows <= 0:
        raise ValueError(f"batch_rows must be positive; got {batch_rows}")

    reducer_list = list(reducers)
    for r in reducer_list:
        r.init()

    annotated, mode = _resolve_spine_rids(spine_df, scale[1])
    n = annotated.height

    # Combining Sum arm forces a single whole-spine batch (see above).
    if (
        mode == "constraint"
        and recipe.sum_block_meta is not None
        and not set(recipe.sum_block_meta.reduce_dims).issubset(
            set(recipe.sum_block_meta.var_source.dims)
        )
    ):
        batch_rows = max(batch_rows, n) if n > 0 else batch_rows

    if mode == "column":
        axis_cols = list(recipe.var_source.dims)
    else:
        # Constraint mode: axis_cols = the over grid columns (minus the
        # injected _rid), matching ``_ranges_via_streaming``'s
        # ``list(over.columns)``.
        axis_cols = [c for c in annotated.columns if c != "_rid"]

    # Hoist all batch-INVARIANT full-frame work to ONCE per term, BEFORE the
    # batch loop — the classified spec, the dense-axis sort verification, the
    # dense-only Param vectors, and (column mode) the whole-frame product.
    # This is what turns the walk from O(n²) (full-frame work × n_batches)
    # into O(n).  The held state is column / low-dim scale, NOT the wide
    # product (which is still built + freed per batch below).
    hoist = _precompute_hoist(recipe, annotated, mode, axis_cols, dense_axes)

    start = 0
    while start < n:
        stop = min(start + batch_rows, n)
        batch = annotated.slice(start, stop - start)
        if mode == "column":
            rid, cid, coef = _build_column_batch_triple(batch, recipe, hoist)
        elif recipe.param_only:
            rid, cid, coef = _build_param_only_batch_triple(
                batch, axis_cols, recipe, dense_axes, hoist.dense_param_vectors
            )
        else:
            rid, cid, coef = _build_constraint_batch_triple(
                batch, axis_cols, recipe, hoist
            )
        for r in reducer_list:
            r.update(rid, cid, coef)
        # Free the batch product before the next slice — peak stays O(batch).
        del rid, cid, coef, batch
        start = stop

    return [r.finalize() for r in reducer_list]
