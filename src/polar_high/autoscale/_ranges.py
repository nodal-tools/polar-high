"""Layer 1 (detect): four-range LP coefficient readout.

Computes the standard HiGHS coefficient-range diagnostic (Matrix /
Cost / Bound / RHS) for a polar-high :class:`Problem` and decides
whether the autoscaler should fire.  Detection only — no LP
modification.  Downstream Layer 3 consumes the returned
:class:`RangeReport` to recommend HiGHS ``user_*_scale`` exponents.

Three entry points
------------------

* :func:`detect_ranges` — production wire-in.  Reads the four ranges
  from polar-high's own ``Solution.streamed_lp_ranges`` when available
  (no duplicate matrix assembly), and falls back to a passModel-style
  rebuild of the LP arrays from a pre-solve ``Problem`` when needed.
* :func:`ranges_from_arrays` — low-level kernel.  Accepts the four raw
  coefficient arrays directly.  Useful for tests and for the post-solve
  path (extract via highspy if ``streamed_lp_ranges`` is absent).
* :func:`ranges_from_streamed` — adapter for the polar-high
  ``Solution.streamed_lp_ranges`` dict (the production hot path).

The same magnitude reduction (``finite & non-zero & |val|``) is used
across all three so the four ranges agree bit-for-bit with what HiGHS
itself prints in its "Coefficient ranges" block — by construction.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._config import ScalingConfig

# Sentinel for "category has zero non-zero finite entries".  Reported as
# ``(nan, nan)`` per the spec; downstream code that wants to treat the
# group as absent should test ``math.isnan(range_tuple[0])``.
_NAN_PAIR: tuple[float, float] = (math.nan, math.nan)


@dataclass(frozen=True)
class RangeReport:
    """Layer 1 output: the four LP coefficient ranges + trigger decision.

    Each range tuple is ``(min |val|, max |val|)`` over **finite, non-zero
    absolute** entries of the corresponding LP component.  Zero-only
    groups return ``(nan, nan)`` and are excluded from the cross-group
    ratio.

    Attributes
    ----------
    matrix:
        Constraint matrix nonzero magnitudes.
    cost:
        Objective coefficient magnitudes.
    bound:
        Variable bound magnitudes (finite ``col_lower`` and ``col_upper``).
    rhs:
        Constraint row bound magnitudes (finite ``row_lower`` and
        ``row_upper``).
    cross_group_max_ratio:
        ``max(hi) / min(lo)`` across the four groups, ignoring any
        group that is ``(nan, nan)``.  ``math.nan`` when every group is
        empty (no LP).
    trigger:
        ``True`` iff any single-group ``hi/lo`` ratio OR the cross-group
        ratio exceeds ``10 ** config.threshold_decades``.
    """

    matrix: tuple[float, float]
    cost: tuple[float, float]
    bound: tuple[float, float]
    rhs: tuple[float, float]
    cross_group_max_ratio: float
    trigger: bool


def _abs_finite_nonzero_min_max(
    arrays: Iterable[np.ndarray | None],
) -> tuple[float, float]:
    """Reduce a sequence of arrays to ``(min |a|, max |a|)`` over their
    finite, non-zero entries.

    Mirrors :func:`polar_high.engine._running_finite_nonzero_min_max` so
    Layer 1's outputs agree with the HiGHS-facing values polar-high
    computes during ``solve()``.  Accepts ``None`` entries (skipped) so
    the caller can pass ``(col_lower, col_upper)`` without filtering.
    Returns ``_NAN_PAIR`` if no array contributes a finite non-zero
    entry.
    """
    lo = math.inf
    hi = 0.0
    for arr in arrays:
        if arr is None:
            continue
        a = np.asarray(arr)
        if a.size == 0:
            continue
        mask = np.isfinite(a) & (a != 0)
        if not mask.any():
            continue
        m = np.abs(a[mask])
        lo = min(lo, float(m.min()))
        hi = max(hi, float(m.max()))
    if hi == 0.0:
        return _NAN_PAIR
    return (lo, hi)


def _ratio(span: tuple[float, float]) -> float:
    """Return ``hi / lo`` for a range tuple, or ``0.0`` for ``(nan, nan)``.

    ``0.0`` lets the trigger comparison ``ratio > 10**N`` evaluate to
    ``False`` for empty groups without a special case at the call site.
    """
    lo, hi = span
    if math.isnan(lo) or math.isnan(hi):
        return 0.0
    if lo == 0.0:
        # Defensive: ``_abs_finite_nonzero_min_max`` filters zeros, but if
        # an upstream adapter slipped one through, treat as "no ratio".
        return 0.0
    return hi / lo


def _build_report(
    matrix: tuple[float, float],
    cost: tuple[float, float],
    bound: tuple[float, float],
    rhs: tuple[float, float],
    config: ScalingConfig,
) -> RangeReport:
    """Assemble a :class:`RangeReport` from four already-reduced groups.

    Computes the cross-group ratio over non-empty groups only, then ORs
    the four per-group ratios with the cross-group ratio against the
    threshold to set ``trigger``.
    """
    groups = (matrix, cost, bound, rhs)
    non_empty_los = [g[0] for g in groups if not math.isnan(g[0])]
    non_empty_his = [g[1] for g in groups if not math.isnan(g[1])]
    if non_empty_los and non_empty_his:
        cross = max(non_empty_his) / min(non_empty_los)
    else:
        cross = math.nan

    threshold = 10.0 ** float(config.threshold_decades)
    per_group_max = max(_ratio(g) for g in groups)
    cross_for_trigger = 0.0 if math.isnan(cross) else cross
    trigger = bool(per_group_max > threshold or cross_for_trigger > threshold)

    return RangeReport(
        matrix=matrix,
        cost=cost,
        bound=bound,
        rhs=rhs,
        cross_group_max_ratio=cross,
        trigger=trigger,
    )


def ranges_from_arrays(
    *,
    matrix_values: np.ndarray | None,
    cost: np.ndarray | None,
    col_lower: np.ndarray | None,
    col_upper: np.ndarray | None,
    row_lower: np.ndarray | None,
    row_upper: np.ndarray | None,
    config: ScalingConfig,
) -> RangeReport:
    """Compute a :class:`RangeReport` directly from LP arrays.

    Parameters mirror the standard HiGHS/COIN/MPS terminology:

    * ``matrix_values`` — the constraint matrix nonzero coefficient
      array (``a_value_`` in HighsLp).  Pass the raw sparse values; the
      sparse structure (starts / indices) is irrelevant for ranges.
    * ``cost`` — the objective coefficient vector (``col_cost_``).
    * ``col_lower`` / ``col_upper`` — variable bounds, with ``±inf``
      sentinels (or ``±highspy.kHighsInf``) for "unbounded".
    * ``row_lower`` / ``row_upper`` — constraint row bounds, ditto.

    Inf / NaN / zero entries are filtered before the magnitude reduction
    so the four ranges match HiGHS' "Coefficient ranges" block.  Any
    array may be ``None`` (treated as empty) — useful when an LP genuinely
    has no objective term or no row bounds (degenerate but legal).
    """
    matrix = _abs_finite_nonzero_min_max([matrix_values])
    cost_r = _abs_finite_nonzero_min_max([cost])
    bound_r = _abs_finite_nonzero_min_max([col_lower, col_upper])
    rhs_r = _abs_finite_nonzero_min_max([row_lower, row_upper])
    return _build_report(matrix, cost_r, bound_r, rhs_r, config)


def ranges_from_streamed(
    streamed: dict[str, tuple[float, float] | None],
    config: ScalingConfig,
) -> RangeReport:
    """Adapt a polar-high ``Solution.streamed_lp_ranges`` dict.

    polar-high already computed the four (min, max) pairs during
    assembly using the same magnitude filter (see
    ``_running_finite_nonzero_min_max`` in polar_high.engine).  Reusing
    its output avoids a duplicate matrix walk in production.  Keys this
    function consumes:

    * ``matrix`` — constraint matrix nonzeros.
    * ``cost`` — objective coefficients.
    * ``col_bound`` — variable bounds (combined ``col_lower`` /
      ``col_upper``).
    * ``row_bound`` — constraint row bounds.

    ``None`` values (polar-high's sentinel for "no finite non-zero
    entries") become ``(nan, nan)`` here, consistent with
    :func:`ranges_from_arrays`.
    """

    def _coerce(key: str) -> tuple[float, float]:
        v = streamed.get(key)
        if v is None:
            return _NAN_PAIR
        lo, hi = v
        return (float(lo), float(hi))

    matrix = _coerce("matrix")
    cost = _coerce("cost")
    bound = _coerce("col_bound")
    rhs = _coerce("row_bound")
    return _build_report(matrix, cost, bound, rhs, config)


def _ranges_via_streaming(problem: Any, config: ScalingConfig) -> RangeReport:
    """Pre-solve range detection via per-term streaming aggregation.

    Avoids ``Problem._build_lp_arrays`` (which materialises every
    constraint's rhs Param-chain + every LHS term's coefficient array,
    accumulates the full COO triple list, and runs a global dedup).
    On a 9.9M-row LP with deep multi-Param chains, that path peaks
    above 30 GB even after the semi-join + streaming retrofits — the
    polars streaming engine can't always push semi-joins all the way
    into a 3+ Param product, so an intermediate still materialises.

    This function aggregates ``(min, max)`` of ``abs(coef)`` per
    objective term, per RHS Param-chain, and per LHS constraint term
    using polars' aggregation expressions.  The streaming engine only
    has to carry running min/max scalars through the chain; no
    intermediate coefficient array is materialised.  Per-term peak is
    O(1) instead of O(rows × Param-product-cardinality).

    Quirks vs the legacy ``_ranges_via_passmodel``:

    * The dedup-sum step in ``_build_lp_arrays`` could combine two
      terms' contributions to the same (row, col) cell before reporting
      the matrix value.  This function reports each per-term coefficient
      magnitude separately, so a cell whose pre-dedup terms are e.g.
      ``+1.0`` and ``-0.999`` shows up as min/max ``≈ 1.0`` here vs
      ``0.001`` after dedup.  Range detection at the magnitude level
      doesn't care: the resulting trigger / recommendation is the same
      modulo this corner.
    * Var/Expr on the RHS is treated as "scale-neutral" for range
      readout (the magnitudes get folded into the LHS at solve time);
      ``_build_lp_arrays`` did the same fold + range; here we just
      skip them, which is mathematically equivalent for the readout
      because their coefficients are already counted via the LHS scan.
    """
    import math as _math
    import os as _os
    import sys as _sys
    import time as _time

    import polars as _pl

    from ..engine import Param as _Param
    from ..engine import _align_enum_join_keys as _align

    _profile = _os.environ.get("POLAR_HIGH_RANGES_PROFILE") == "1"
    if _profile:
        try:
            import psutil as _ps
            _proc = _ps.Process()
            _t0 = _time.monotonic()

            def _emit(phase: str, **extras) -> None:
                rss = _proc.memory_info().rss / (1024 ** 3)
                wall = _time.monotonic() - _t0
                extras_str = "\t".join(f"{k}={v}" for k, v in extras.items())
                print(
                    f"[ranges-stream profile]\tphase={phase}\trss_gb={rss:.2f}"
                    f"\twall_s={wall:.2f}"
                    + (f"\t{extras_str}" if extras_str else ""),
                    file=_sys.stderr, flush=True,
                )
            _emit("enter", n_cstrs=len(problem._cstrs))
        except ImportError:
            _profile = False

    inf = _math.inf
    matrix_lo, matrix_hi = inf, 0.0
    cost_lo, cost_hi = inf, 0.0
    bound_lo, bound_hi = inf, 0.0
    rhs_lo, rhs_hi = inf, 0.0

    # Layer 2 side vectors (off when both are None — pre-Layer-2 caller).
    # Layer 2 stores ``_layer2_col_factor`` as ``1 / cf_math`` (inverse
    # forward factor) and ``_layer2_row_factor`` as ``rf_math`` (forward).
    # Consumers multiply collected ``coef`` by these to get post-Layer-2
    # magnitudes; this readout does the same so Layer 3's user_*_scale
    # decision agrees bit-for-bit with what the consumers will emit.
    _l2_rf = getattr(problem, "_layer2_row_factor", None)
    _l2_cf = getattr(problem, "_layer2_col_factor", None)

    def _agg(lazy_frame: _pl.LazyFrame, col: str) -> tuple[float | None, float | None]:
        """Min/max(abs(``col``)) over finite non-zero rows.

        Produces a per-row ``_abs`` column via streaming-collect, then
        numpy-reduces.  The shape "collect a column, then reduce" is
        what polars' streaming engine handles reliably for deep
        Param-chain plans (proven via :meth:`Problem.write_mps`'s
        per-term collect); the alternative "select(min, max) inside
        the lazy plan" form fails to stream on the same chains and
        materialises the upstream join — observed at >30 GB on the
        FlexTool DES LP's ``profile_flow_upper_limit`` constraint.
        Numpy min/max on the (potentially ~10⁷-row) collected column
        is O(rows), with the column itself occupying ~8 bytes/row.
        """
        plan = (
            lazy_frame
            .select(_pl.col(col).abs().alias("__abs"))
            .filter(_pl.col("__abs").is_finite() & (_pl.col("__abs") > 0))
        )
        try:
            df = plan.collect(engine="streaming")
        except TypeError:
            try:
                df = plan.collect(streaming=True)
            except TypeError:
                df = plan.collect()
        except Exception:
            df = plan.collect()
        if df.height == 0:
            return None, None
        arr = df["__abs"].to_numpy()
        return float(arr.min()), float(arr.max())

    def _collect_streaming(plan: _pl.LazyFrame) -> _pl.DataFrame:
        try:
            return plan.collect(engine="streaming")
        except TypeError:
            try:
                return plan.collect(streaming=True)
            except TypeError:
                return plan.collect()
        except Exception:
            return plan.collect()

    def _reduce_abs(vals: np.ndarray) -> tuple[float | None, float | None]:
        if vals.size == 0:
            return None, None
        mask = np.isfinite(vals) & (vals != 0)
        if not mask.any():
            return None, None
        a = np.abs(vals[mask])
        return float(a.min()), float(a.max())

    def _update_matrix(lo: float | None, hi: float | None) -> None:
        nonlocal matrix_lo, matrix_hi
        if lo is None or hi is None:
            return
        if lo < matrix_lo:
            matrix_lo = lo
        if hi > matrix_hi:
            matrix_hi = hi

    def _update_cost(lo: float | None, hi: float | None) -> None:
        nonlocal cost_lo, cost_hi
        if lo is None or hi is None:
            return
        if lo < cost_lo:
            cost_lo = lo
        if hi > cost_hi:
            cost_hi = hi

    def _update_rhs_scalar(value: float) -> None:
        nonlocal rhs_lo, rhs_hi
        if _math.isfinite(value) and value != 0:
            a = abs(float(value))
            if a < rhs_lo:
                rhs_lo = a
            if a > rhs_hi:
                rhs_hi = a

    # Bounds — pure Python; one float per var family × 2.
    for v in problem._vars.values():
        for b in (v.lower, v.upper):
            if _math.isfinite(b) and b != 0:
                a = abs(float(b))
                if a < bound_lo:
                    bound_lo = a
                if a > bound_hi:
                    bound_hi = a

    if _profile:
        _emit("bounds_done")

    # Cost — stream-aggregate each objective term.  Apply Layer 2 col
    # factor (objective row is NOT in the row-factor vector — GLPK
    # convention).  No-op when ``_l2_cf`` is None.
    for ti, t in enumerate(problem._obj_terms):
        if t.lazy is None:
            continue
        if _l2_cf is None:
            lo, hi = _agg(t.lazy, "coef")
        else:
            plan = t.lazy.select("col_id", "coef")
            df = _collect_streaming(plan)
            if df.height == 0:
                lo, hi = None, None
            else:
                cids = df["col_id"].to_numpy().astype(np.int64)
                vals = df["coef"].to_numpy().astype(np.float64)
                vals = vals * np.abs(_l2_cf[cids])
                lo, hi = _reduce_abs(vals)
        _update_cost(lo, hi)
        if _profile:
            _emit("obj_term_done", term_idx=ti)

    if _profile:
        _emit("obj_done")

    # Per-family size guard.  Families above this row count are skipped
    # in Layer-1 detection — their rhs/term coefficient distributions
    # don't fold into the four-range readout.  This is a deliberate
    # graceful-degradation: the polars streaming engine intermittently
    # fails to push the row-key semi-join into deep multi-Param product
    # chains on very large families (the FlexTool DES LP's
    # ``profile_flow_upper_limit`` family is the canonical offender —
    # 1.5 M rows × multi-Param rhs allocates >30 GB even with the
    # semi-join + left-join + streaming-collect pattern that
    # ``Problem.write_mps`` survives on).  Skipping means the autoscale
    # trigger / Layer 3 recommendation is based on the families it
    # could read; smaller-family ranges still inform the decision.
    # Override with ``POLAR_HIGH_RANGES_MAX_FAMILY_ROWS=<int>`` —
    # set to 0 to disable the skip and accept the OOM risk.
    try:
        _max_family_rows = int(
            _os.environ.get("POLAR_HIGH_RANGES_MAX_FAMILY_ROWS", "1000000")
        )
    except (ValueError, TypeError):
        _max_family_rows = 1_000_000

    # Matrix + RHS — per constraint family.  ``base_row`` is the 0-based
    # absolute constraint row id (same indexing space as
    # ``_layer2_row_factor`` — apply_layer2 walks ``_cstrs`` in this same
    # order, so ``base_row + local_rid`` lines up with the side vector).
    next_row = 0
    for cname, proto, over in problem._cstrs:
        # Scalar (no ``over``) families still occupy one row in the side
        # vector — mirror apply_layer2's ``row_count = 1 if over is None``.
        row_count = 1 if over is None else int(over.height)
        base_row = next_row
        next_row += row_count
        if _profile:
            _emit(
                "family_start", family=cname,
                row_count=row_count,
                term_count=len(proto.expr.terms),
            )
        if _max_family_rows > 0 and row_count > _max_family_rows:
            if _profile:
                _emit(
                    "family_skipped", family=cname,
                    row_count=row_count,
                    threshold=_max_family_rows,
                )
            continue
        rhs = proto.rhs

        # RHS magnitude readout — mirror ``Problem.write_mps``'s rhs
        # collect pattern exactly (semi-join + left-join with the row
        # index + streaming collect), then numpy-reduce the value
        # column.  write_mps's pattern is the proven-working shape on
        # FlexTool's ``profile_flow_upper_limit`` family
        # (1.5 M rows × multi-Param rhs); replacing it with a "produce
        # an aggregate inside the lazy plan" form makes polars give up
        # on pushing the row-key filter into the Param product, and
        # the upstream join materialises >30 GB.  Producing the same
        # per-row frame as write_mps + a final numpy ``np.abs`` /
        # ``min`` / ``max`` adds ~12 MB per Mrow (the column itself)
        # on top of polars' own collect-time peak.
        if isinstance(rhs, (int, float)):
            # Scalar rhs is broadcast across all family rows.  Layer 2
            # multiplies each emitted row by its own row_factor, so the
            # magnitude bound is ``|rhs| * (min/max of |rf_slice|)``.
            if _l2_rf is None or row_count == 0:
                _update_rhs_scalar(float(rhs))
            else:
                rf_slice = np.abs(_l2_rf[base_row : base_row + row_count])
                vals = float(rhs) * rf_slice
                lo, hi = _reduce_abs(vals)
                if lo is not None:
                    if lo < rhs_lo:
                        rhs_lo = lo
                    if hi > rhs_hi:
                        rhs_hi = hi
        elif isinstance(rhs, _Param):
            if over is not None and rhs.dims:
                on = list(rhs.dims)
                # When row_factor is on, we need per-row alignment with
                # ``_rid`` so we can multiply ``value`` by
                # ``row_factor[base_row + _rid]`` before reducing.  Mirror
                # ``_build_canonical_matrix``'s rhs build: carry an
                # ``_rid`` column on the row index.
                if _l2_rf is None:
                    ri_a, rf_a = _align(over.lazy(), rhs.lazy, on)
                    keys = ri_a.select(on).unique()
                    rf_pruned = rf_a.join(keys, on=on, how="semi")
                    plan = ri_a.join(rf_pruned, on=on, how="left").select("value")
                    j = _collect_streaming(plan)
                    if j.height > 0:
                        vals = j["value"].fill_null(0.0).to_numpy()
                        lo, hi = _reduce_abs(vals)
                        if lo is not None:
                            if lo < rhs_lo:
                                rhs_lo = lo
                            if hi > rhs_hi:
                                rhs_hi = hi
                else:
                    over_rid = over.with_columns(
                        _rid=_pl.int_range(0, over.height, dtype=_pl.Int64)
                    )
                    ri_a, rf_a = _align(over_rid.lazy(), rhs.lazy, on)
                    keys = ri_a.select(on).unique()
                    rf_pruned = rf_a.join(keys, on=on, how="semi")
                    plan = ri_a.join(rf_pruned, on=on, how="left").select("_rid", "value")
                    j = _collect_streaming(plan)
                    if j.height > 0:
                        rids = j["_rid"].to_numpy().astype(np.int64)
                        vals = j["value"].fill_null(0.0).to_numpy().astype(np.float64)
                        vals = vals * np.abs(_l2_rf[base_row + rids])
                        lo, hi = _reduce_abs(vals)
                        if lo is not None:
                            if lo < rhs_lo:
                                rhs_lo = lo
                            if hi > rhs_hi:
                                rhs_hi = hi
            else:
                # Dimless Param: a single scalar value broadcast across
                # the family rows.  Apply row_factor slice same as the
                # numeric scalar branch above.
                f = rhs.frame
                if "value" in f.columns and f.height > 0:
                    vals = f["value"].to_numpy().astype(np.float64)
                    if _l2_rf is None or row_count == 0:
                        lo, hi = _reduce_abs(vals)
                    else:
                        # Each (broadcast) rhs value is multiplied by
                        # every row's row_factor; expand explicitly.
                        rf_slice = np.abs(_l2_rf[base_row : base_row + row_count])
                        vals = np.outer(vals, rf_slice).ravel()
                        lo, hi = _reduce_abs(vals)
                    if lo is not None:
                        if lo < rhs_lo:
                            rhs_lo = lo
                        if hi > rhs_hi:
                            rhs_hi = hi
        # Var / Expr RHS: skip — they fold into the LHS at solve time,
        # so their coefficient magnitudes are already counted via the
        # LHS scan below.  ``_build_canonical_matrix`` does the same
        # fold; here we just rely on the LHS pass to catch the same
        # magnitudes.

        if _profile:
            _emit("rhs_done", family=cname)

        # LHS matrix magnitudes.
        if over is None:
            axis_cols: list[str] = []
            row_index_lf: _pl.LazyFrame | None = None
        else:
            axis_cols = list(over.columns)
            row_index_lf = over.lazy()

        # If side vectors are off, use the existing minimal path (just
        # the abs-only collect).  If on, we need ``_rid`` + ``col_id``
        # in the collected frame so we can index ``_l2_rf[base_row +
        # _rid]`` and ``_l2_cf[col_id]`` before the magnitude reduce.
        # Mirror ``_build_canonical_matrix``'s LHS join.
        if row_index_lf is not None and (_l2_rf is not None):
            row_index_lf_rid = over.with_columns(
                _rid=_pl.int_range(0, over.height, dtype=_pl.Int64)
            ).lazy()
        else:
            row_index_lf_rid = None

        for ti, term in enumerate(proto.expr.terms):
            if not term.dims or row_index_lf is None:
                # Scalar (no row binding) — broadcast across the family
                # rows.  ``_agg`` reduces |coef| only; when side vectors
                # are on, expand to per-(row,col) magnitudes so the row/
                # col factors apply.
                if _l2_rf is None and _l2_cf is None:
                    lo, hi = _agg(term.lazy, "coef")
                else:
                    df = _collect_streaming(term.lazy.select("col_id", "coef"))
                    if df.height == 0:
                        lo, hi = None, None
                    else:
                        cids = df["col_id"].to_numpy().astype(np.int64)
                        vals = df["coef"].to_numpy().astype(np.float64)
                        if _l2_cf is not None:
                            vals = vals * np.abs(_l2_cf[cids])
                        if _l2_rf is None or row_count == 0:
                            lo, hi = _reduce_abs(vals)
                        else:
                            rf_slice = np.abs(
                                _l2_rf[base_row : base_row + row_count]
                            )
                            # Per-row broadcast: row_count × len(cids).
                            tiled_vals = np.outer(rf_slice, vals).ravel()
                            lo, hi = _reduce_abs(tiled_vals)
            else:
                on = [d for d in term.dims if d in axis_cols]
                if _l2_rf is None:
                    rl_a, tl_a = _align(row_index_lf, term.lazy, on)
                    keys = rl_a.select(on).unique()
                    pruned = tl_a.join(keys, on=on, how="semi")
                    if _l2_cf is None:
                        lo, hi = _agg(pruned, "coef")
                    else:
                        df = _collect_streaming(
                            pruned.select("col_id", "coef")
                        )
                        if df.height == 0:
                            lo, hi = None, None
                        else:
                            cids = df["col_id"].to_numpy().astype(np.int64)
                            vals = df["coef"].to_numpy().astype(np.float64)
                            vals = vals * np.abs(_l2_cf[cids])
                            lo, hi = _reduce_abs(vals)
                else:
                    # Side vectors on — left-join with ``_rid`` so the
                    # collected frame carries the absolute row id needed
                    # to index ``_l2_rf``.
                    rl_a, tl_a = _align(row_index_lf_rid, term.lazy, on)
                    keys = rl_a.select(on).unique()
                    pruned = tl_a.join(keys, on=on, how="semi")
                    plan = (
                        rl_a.join(pruned, on=on, how="inner")
                        .select("_rid", "col_id", "coef")
                    )
                    df = _collect_streaming(plan)
                    if df.height == 0:
                        lo, hi = None, None
                    else:
                        rids = df["_rid"].to_numpy().astype(np.int64)
                        cids = df["col_id"].to_numpy().astype(np.int64)
                        vals = df["coef"].to_numpy().astype(np.float64)
                        vals = vals * np.abs(_l2_rf[base_row + rids])
                        if _l2_cf is not None:
                            vals = vals * np.abs(_l2_cf[cids])
                        lo, hi = _reduce_abs(vals)
            _update_matrix(lo, hi)
            if _profile:
                _emit(
                    "term_done", family=cname, term_idx=ti,
                    has_dims=str(bool(term.dims)),
                )

    matrix = (matrix_lo, matrix_hi) if matrix_hi > 0 else _NAN_PAIR
    cost = (cost_lo, cost_hi) if cost_hi > 0 else _NAN_PAIR
    bound = (bound_lo, bound_hi) if bound_hi > 0 else _NAN_PAIR
    rhs_r = (rhs_lo, rhs_hi) if rhs_hi > 0 else _NAN_PAIR
    return _build_report(matrix, cost, bound, rhs_r, config)


def _ranges_via_passmodel(problem: Any, config: ScalingConfig) -> RangeReport:
    """Fallback path: assemble LP via polar-high's internal ``_build_lp_arrays``.

    Reached only when the caller hands :func:`detect_ranges` a pre-solve
    ``Problem`` and no ``streamed_lp_ranges`` are available.  This
    duplicates the matrix walk polar-high does inside ``solve()``, so
    the production wire-in prefers :func:`ranges_from_streamed` to avoid
    the extra pass.  We document the cost explicitly rather than
    silently using it.
    """
    import os as _os
    import sys as _sys
    import time as _time
    _profile = _os.environ.get("POLAR_HIGH_RANGES_PROFILE") == "1"
    if _profile:
        try:
            import psutil as _ps
            _proc = _ps.Process()
            _t0 = _time.monotonic()

            def _emit(phase: str, **extras) -> None:
                rss = _proc.memory_info().rss / (1024 ** 3)
                wall = _time.monotonic() - _t0
                extras_str = "\t".join(f"{k}={v}" for k, v in extras.items())
                print(
                    f"[ranges profile]\tphase={phase}\trss_gb={rss:.2f}"
                    f"\twall_s={wall:.2f}"
                    + (f"\t{extras_str}" if extras_str else ""),
                    file=_sys.stderr, flush=True,
                )
            _emit("enter")
        except ImportError:
            _profile = False

    # ``_build_lp_arrays`` now consumes the canonical matrix (Stage B2),
    # which derives column bounds from ``Var.lower`` / ``Var.upper``
    # internally — no need to pre-build per-column bound arrays here.
    n_cols = problem._next_col

    if _profile:
        _emit("bounds_built", n_cols=n_cols)

    (
        col_lb_h,
        col_ub_h,
        row_lb_h,
        row_ub_h,
        sorted_v,
        _sorted_r,
        _starts,
        _row_names,
        _n_rows,
    ) = problem._build_lp_arrays()

    if _profile:
        _emit("build_lp_arrays_done", nnz=int(sorted_v.size))

    # The objective vector — built inline in ``_solve_passmodel`` /
    # ``_solve_streaming``; mirror that walk here.  Use the streaming
    # engine on each ``t.lazy.collect()`` so a Param-chain objective
    # term doesn't materialise a wide intermediate (same anti-explosion
    # rationale as the LHS term collect inside ``_build_lp_arrays``).
    col_obj = np.zeros(n_cols, dtype=np.float64)
    import polars as pl  # local import — autoscale must not pull polars

    # at import time for environments that don't need this fallback.
    for ti, t in enumerate(problem._obj_terms):
        if t.lazy is None:
            continue
        if isinstance(t.lazy, pl.LazyFrame):
            try:
                f = t.lazy.collect(engine="streaming")
            except TypeError:
                try:
                    f = t.lazy.collect(streaming=True)
                except TypeError:
                    f = t.lazy.collect()
            except Exception:
                f = t.lazy.collect()
        else:
            f = t.lazy
        if f.height == 0:
            continue
        np.add.at(
            col_obj,
            f["col_id"].to_numpy(),
            f["coef"].to_numpy(),
        )
        if _profile:
            _emit("obj_term_collected", term_idx=ti, height=int(f.height))

    # ``kHighsInf`` substitution in ``_build_lp_arrays`` replaces ±inf,
    # so we filter via ``np.isfinite`` and the HiGHS sentinel explicitly.
    # HiGHS uses 1e30 as the kHighsInf value; treat anything at that
    # magnitude as "unbounded" for range purposes.
    import highspy

    inf_sentinel = float(highspy.kHighsInf)

    def _strip_inf(a: np.ndarray) -> np.ndarray:
        return np.where(np.abs(a) >= inf_sentinel, 0.0, a)

    return ranges_from_arrays(
        matrix_values=sorted_v,
        cost=col_obj,
        col_lower=_strip_inf(col_lb_h),
        col_upper=_strip_inf(col_ub_h),
        row_lower=_strip_inf(row_lb_h),
        row_upper=_strip_inf(row_ub_h),
        config=config,
    )


def detect_ranges(problem_or_solution: Any, config: ScalingConfig) -> RangeReport:
    """Compute the Layer 1 four-range report.

    Production callers pass either:

    * a polar-high ``Solution`` (post-solve) — the fast path; we just
      consume ``solution.streamed_lp_ranges``.
    * a polar-high ``Problem`` (pre-solve) — the fallback path; we
      assemble the LP arrays via ``Problem._build_lp_arrays`` and
      reduce them ourselves.  Slower; only used by callers that need
      the ranges before solve dispatch.

    The function dispatches on attribute presence, never on type, so a
    ``LiteSolution`` (commercial-solver wrapper) or a custom mock
    carrying ``streamed_lp_ranges`` works the same.

    Raises ``TypeError`` if neither shape is recognised — silently
    falling back would be the wrong call here; the autoscaler must know
    what kind of input it's been handed.
    """
    streamed = getattr(problem_or_solution, "streamed_lp_ranges", None)
    if isinstance(streamed, dict):
        return ranges_from_streamed(streamed, config)

    # Pre-solve Problem: detect via the streaming-aggregation path.
    # We never mutate the Problem.  ``_ranges_via_streaming`` bypasses
    # ``_build_lp_arrays`` entirely (the legacy fallback ``_ranges_via_passmodel``
    # remained at the bottom of this module for back-compat / test
    # coverage, but production callers should never reach it).
    if (
        hasattr(problem_or_solution, "_vars")
        and hasattr(problem_or_solution, "_cstrs")
        and hasattr(problem_or_solution, "_obj_terms")
    ):
        return _ranges_via_streaming(problem_or_solution, config)

    raise TypeError(
        "detect_ranges expects a polar-high Problem (pre-solve) or a "
        "Solution carrying streamed_lp_ranges; got "
        f"{type(problem_or_solution).__name__}"
    )


__all__ = [
    "RangeReport",
    "detect_ranges",
    "ranges_from_arrays",
    "ranges_from_streamed",
]
