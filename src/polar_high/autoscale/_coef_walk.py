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
    _block_coo_classify,
    _build_block_coo_plan,
    _build_lhs_pruned_plan,
    _build_sum_block_coo_plan,
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
    )

    def __init__(
        self,
        var_source: Var,
        param_sources: Sequence[tuple[Param, int]],
        coef_scalar: float = 1.0,
        where_frames: tuple[pl.LazyFrame, ...] | None = None,
        where_map_frames: tuple[tuple[pl.LazyFrame, frozenset[str]], ...]
        | None = None,
        sum_block_meta: SumBlockMeta | None = None,
        reduced_lazy: pl.LazyFrame | None = None,
        reduced_dims: tuple[str, ...] | None = None,
    ) -> None:
        if not isinstance(var_source, Var):
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
        # few) DISTINCT col_ids in this batch, then group rows by their
        # bucket key.  No Python-per-row loop over the coef stream.
        uniq_cids, inv = np.unique(cids, return_inverse=True)
        bucket_of_uniq = [self._classify(int(c)) for c in uniq_cids.tolist()]
        for u_idx, bkey in enumerate(bucket_of_uniq):
            if bkey is None:
                continue
            sel = inv == u_idx
            if not sel.any():
                continue
            sub_log = logs[sel]
            sub_a = a[sel]
            self._fold(
                bkey,
                float(sub_log.sum()),
                int(sub_a.size),
                float(sub_a.min()),
                float(sub_a.max()),
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


def _build_constraint_batch_triple(
    batch_over: pl.DataFrame,
    axis_cols: list[str],
    recipe: CoefWalkRecipe,
    dense_axes: tuple[str, ...] | None,
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

    if recipe.sum_block_meta is None:
        # Reconstruct a minimal term-shaped object the classifier reads.
        term_proxy = _NonSumTermProxy(recipe)
        blk_on = [d for d in var.dims if d in axis_cols]
        spec = _block_coo_classify(term_proxy, axis_cols, blk_on, dense_axes)
        if spec is not None and recipe.where_map_frames is None:
            _verify_dense_sorted(
                var.frame,
                spec["non_dense_dims"],
                spec["dense_dims"],
                getattr(var, "name", None),
            )
            df = _build_block_coo_plan(
                row_index_lf,
                axis_cols,
                var,
                recipe.param_sources,
                blk_on,
                recipe.coef_scalar,
                recipe.where_frames,
                spec,
            )
    else:
        meta = recipe.sum_block_meta
        term_proxy = _SumTermProxy(recipe)
        keep_on = [d for d in meta.keep if d in axis_cols]
        spec = _sum_block_coo_classify(term_proxy, axis_cols, keep_on, dense_axes)
        if spec is not None:
            _verify_dense_sorted(
                meta.var_source.frame,
                spec["non_dense_dims"],
                spec["dense_dims"],
                getattr(meta.var_source, "name", None),
            )
            try:
                df = _build_sum_block_coo_plan(
                    row_index_lf, axis_cols, meta, keep_on, spec
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


def _reduced_lazy_collect(
    row_index_lf: pl.LazyFrame,
    axis_cols: list[str],
    recipe: CoefWalkRecipe,
) -> pl.DataFrame:
    """Attach ``_rid`` to the term's OWN reduced lazy plan and collect
    ``(_rid, col_id, coef)`` for the batch.

    Used as the Sum backstop when the Sum block-COO classifier declines:
    the reduced coefficient (the post-Sum group_by result) is already
    materialised in ``recipe.reduced_lazy`` (the term's ``.lazy``), so we
    inner-join the batch ``row_index_lf`` on the kept axis dims to attach
    ``_rid``, restricting the output (and the collect) to the batch's
    rows.  Reduced groups are already complete in the reduced plan, so no
    group is split across batches.  This is byte-identical to the engine's
    "emit the reduced ``term.lazy`` verbatim" fallback.
    """
    reduced = recipe.reduced_lazy
    if reduced is None:
        raise ValueError(
            "Sum recipe declined block-COO but carries no reduced_lazy "
            "fallback plan; build the recipe via CoefWalkRecipe.from_term "
            "so the term's reduced lazy plan is captured."
        )
    rdims = recipe.reduced_dims or ()
    on = [d for d in rdims if d in axis_cols]
    ri_a, red_a = _align_enum_join_keys(row_index_lf, reduced, on)
    plan = ri_a.join(red_a, on=on, how="inner").select("_rid", "col_id", "coef")
    return plan.collect()


def _build_column_batch_triple(
    batch_seed: pl.DataFrame,
    recipe: CoefWalkRecipe,
    dense_axes: tuple[str, ...] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the ``(col_id, coef)`` per-cell product for ONE batch of a
    *column* spine (objective / bare-Var), reusing the positional
    block-COO slice-multiply on the batch's Var-seed slice.

    ``batch_seed`` is a slice of the Var frame carrying ``(*var_dims,
    col_id)``.  The objective has no constraint row, so we attach NO
    ``_rid``; we feed the seed to :func:`_build_block_coo_plan` as the
    ``row_index_lf`` keyed on the var dims themselves (an identity attach:
    ``on = var.dims`` so each seed row maps to exactly itself), then read
    ``(col_id, coef)`` and report ``rid = -1``.

    This mirrors ``_ranges._obj_chain_bounded`` (the bounded objective
    Param-chain readout) but generalised to an arbitrary batch slice and
    routed through the shared block-COO builder rather than a bespoke
    positional loop.  Param chains are required (a bare Var with no Param
    yields ``coef == coef_scalar`` directly).
    """
    var = recipe.var_source
    var_dims = list(var.dims)

    if not recipe.param_sources:
        # Bare Var (no Param chain): coef is the constant coef_scalar per
        # cell.  No product to build.
        cids = batch_seed["col_id"].to_numpy().astype(np.int64)
        coef = np.full(cids.size, recipe.coef_scalar, dtype=np.float64)
        rid = np.full(cids.size, -1, dtype=np.int64)
        return rid, cids, coef

    # Use the batch seed itself as the row_index, keyed on the var dims —
    # an identity attach (every seed row joins to itself).  ``axis_cols``
    # = var dims so the block-COO ``on`` is the full var dim set.
    axis_cols = list(var_dims)
    # Attach an _rid so the builder's emission schema has it; we ignore it
    # (column mode reports rid = -1).  The identity row_index carries the
    # var dims + _rid.
    row_index = batch_seed.select(*var_dims).with_columns(
        _rid=pl.int_range(0, batch_seed.height, dtype=pl.Int64)
    )
    term_proxy = _NonSumTermProxy(recipe)
    blk_on = list(var_dims)
    spec = _block_coo_classify(term_proxy, axis_cols, blk_on, dense_axes)
    df: pl.DataFrame | None = None
    if spec is not None and recipe.where_map_frames is None:
        _verify_dense_sorted(
            var.frame, spec["non_dense_dims"], spec["dense_dims"],
            getattr(var, "name", None),
        )
        df = _build_block_coo_plan(
            row_index.lazy(),
            axis_cols,
            var,
            recipe.param_sources,
            blk_on,
            recipe.coef_scalar,
            recipe.where_frames,
            spec,
        )
    if df is None:
        df = _lhs_prune_down_collect(row_index.lazy(), axis_cols, recipe)

    if df.height == 0:
        z = np.empty(0, dtype=np.float64)
        zi = np.empty(0, dtype=np.int64)
        return zi, zi, z
    cids = df["col_id"].to_numpy().astype(np.int64)
    coef = df["coef"].to_numpy().astype(np.float64)
    rid = np.full(cids.size, -1, dtype=np.int64)
    return rid, cids, coef


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

    start = 0
    while start < n:
        stop = min(start + batch_rows, n)
        batch = annotated.slice(start, stop - start)
        if mode == "column":
            rid, cid, coef = _build_column_batch_triple(
                batch, recipe, dense_axes
            )
        else:
            rid, cid, coef = _build_constraint_batch_triple(
                batch, axis_cols, recipe, dense_axes
            )
        for r in reducer_list:
            r.update(rid, cid, coef)
        # Free the batch product before the next slice — peak stays O(batch).
        del rid, cid, coef, batch
        start = stop

    return [r.finalize() for r in reducer_list]
