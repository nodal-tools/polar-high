"""Generic polars-backed LP kernel.

Three primitives — ``Var``, ``Param``, ``Sum`` — and one container
(``Problem``).  Knows nothing about flextool, energy systems, or any
specific model.  A constraint is built as either:

  1. an ``Expr`` produced by overloaded operators (``v <= cap``,
     ``Sum(...) >= rhs``, ``lhs.eq(rhs)``), passed positionally to
     ``Problem.add_cstr``; or

  2. a labelled ``terms`` dict, summed across all entries, with an
     explicit ``sense`` and ``rhs``.  Use this when a constraint is
     naturally a sum of named contributions (storage transitions, sink
     flow, source flow, slack — like flextool's nodeBalance_eq).

A variable is a polars frame ``(*dims, col_id)`` — one LP column per
row.  A parameter is a polars frame ``(*dims, value)``.  ``Var * Param``
joins on shared dims and emits an ``Expr`` term ``(*union_dims, col_id,
coef)``.  ``Sum(expr, over=…)`` group-by-sums one or more dims; the
remaining dims become the constraint's row dims when the term is bound
to ``over=`` at ``add_cstr`` time.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass, replace

import highspy
import numpy as np
import polars as pl

from ._log_routing import route_highs_log_to_stdout

# ---------------------------------------------------------------------------
# Stream-time LP-range helpers
#
# Used inside ``_solve_streaming`` to accumulate the four coefficient-range
# tuples (matrix / cost / col_bound / row_bound) at near-zero cost (a
# handful of O(n) numpy scans on arrays we already build), and to derive a
# col_bound-only ``user_bound_scale`` recommendation that mirrors flextool's
# ``recommend_user_bound_scale_from_lp``.
# ---------------------------------------------------------------------------


def _running_finite_nonzero_min_max(
    arr: np.ndarray,
    cur_lo: float,
    cur_hi: float,
    *,
    chunk: int = 1_048_576,
) -> tuple[float, float]:
    """Update ``(cur_lo, cur_hi)`` with the finite, non-zero ``|arr|``.

    Scans ``arr`` in slices so the transient working set stays bounded at
    ``chunk`` float64s (~16 MB at the default), regardless of how big the
    family-level coefficient array gets.  The old single-pass version
    materialised two full-size temporaries (``arr[finite]`` and
    ``np.abs(...)``); on a 36 M-nonzero family that was ~576 MB peak per
    call for what should be a streaming reduction.

    ``cur_lo`` is seeded with ``math.inf`` and ``cur_hi`` with ``0.0`` so
    that an entirely-empty category leaves the sentinels untouched and is
    later packaged as ``None``.
    """
    if arr.size == 0:
        return cur_lo, cur_hi
    lo, hi = cur_lo, cur_hi
    for start in range(0, arr.size, chunk):
        c = arr[start : start + chunk]
        mask = np.isfinite(c) & (c != 0)
        if not mask.any():
            continue
        a = np.abs(c[mask])
        lo = min(lo, float(a.min()))
        hi = max(hi, float(a.max()))
    return (lo, hi)


def _floor_small_coefs(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Return ``arr`` with every entry whose ``abs`` is strictly below
    ``threshold`` replaced by exactly ``0.0``.

    The cutoff is information-preserving on the small end only: values
    with ``abs(value) == threshold`` are kept, and large values are
    untouched.  ``threshold <= 0.0`` is a no-op (returns ``arr``
    unchanged, no copy).  ``±inf`` and ``NaN`` are never floored —
    ``abs(inf) < threshold`` and ``abs(nan) < threshold`` are both
    ``False`` — so one-sided row-bound sentinels survive verbatim.

    The replacement keeps array shape/length intact (it sets cells to
    ``0.0`` rather than dropping them), so downstream COO/CSC structure
    and ordering are preserved.
    """
    if threshold <= 0.0 or arr.size == 0:
        return arr
    out = np.where(np.abs(arr) < threshold, 0.0, arr)
    return out.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Solve-path memory profiling helper
#
# Activated by ``POLAR_HIGH_SOLVE_PROFILE=1``.  Mirrors the
# ``POLAR_HIGH_WRITE_MPS_PROFILE`` precedent in :meth:`Problem.write_mps`:
# tab-separated stderr lines, zero overhead when the env var is unset
# (one ``os.environ.get`` per :meth:`Problem.solve` call, no psutil
# import, no time.monotonic calls, no stderr output).
# ---------------------------------------------------------------------------


def _make_solve_profile_emitter():
    """Return ``(emit, enabled)`` where ``emit(phase, **extras)`` writes
    a ``[solve profile]`` tab-separated line to stderr, or ``(None, False)``
    when ``POLAR_HIGH_SOLVE_PROFILE`` is unset / psutil is missing.

    Shared by :meth:`Problem._solve_streaming` and
    :meth:`Problem._build_canonical_matrix` (when the latter is reached
    via :meth:`Problem._build_lp_arrays` on the non-streaming path).
    """
    if os.environ.get("POLAR_HIGH_SOLVE_PROFILE") != "1":
        return None, False
    try:
        import psutil as _ps
    except ImportError:
        sys.stderr.write(
            "[solve profile] psutil not installed — "
            "profiling disabled (install psutil to enable).\n"
        )
        sys.stderr.flush()
        return None, False
    _proc = _ps.Process()
    _state = {
        "t0": time.monotonic(),
        "prev_rss_gb": _proc.memory_info().rss / (1024**3),
    }

    def emit(phase: str, **extras: object) -> None:
        rss_gb = _proc.memory_info().rss / (1024**3)
        delta_gb = rss_gb - _state["prev_rss_gb"]
        _state["prev_rss_gb"] = rss_gb
        wall_s = time.monotonic() - _state["t0"]
        parts = [
            "[solve profile]",
            f"phase={phase}",
            f"rss_gb={rss_gb:.2f}",
            f"delta_gb={delta_gb:+.2f}",
            f"wall_s={wall_s:.2f}",
        ]
        for k, v in extras.items():
            parts.append(f"{k}={v}")
        sys.stderr.write("\t".join(parts) + "\n")
        sys.stderr.flush()

    return emit, True


# ---------------------------------------------------------------------------
# Enum-dtype-aware join helpers
#
# polars 1.40 refuses to join two columns of dtype ``pl.Enum`` when the
# Enums carry different categorical vocabularies — even when the values
# present in the rows would coerce cleanly.  Linear-programming DSLs
# routinely have this pattern: one Param defined on a subset of an
# axis, another on the superset, both named ``p`` (process), ``n``
# (node), etc.  The kernel can fix this automatically without losing
# the categorical semantics — we pick the wider Enum and cast the
# narrower side into it before the join.
#
# These helpers are intentionally generic — no domain assumptions
# (axis names, vocab contents, etc).  They sit at every internal
# ``.join`` site that takes axis-aware keys.
# ---------------------------------------------------------------------------


_Frame = pl.DataFrame | pl.LazyFrame


def _column_dtype(frame: _Frame, name: str) -> pl.DataType | None:
    """Return the dtype of column ``name`` in ``frame`` (eager or lazy),
    or ``None`` if the column is absent.  Uses ``collect_schema`` so
    lazy plans don't need to be materialised.
    """
    schema = frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema
    return schema.get(name)


def _align_enum_join_keys(
    left: _Frame,
    right: _Frame,
    on: list[str] | tuple[str, ...],
) -> tuple[_Frame, _Frame]:
    """Align dtypes of shared join keys so polars's strict Enum check
    doesn't reject otherwise-compatible joins.

    For each key in ``on``:

    * If both sides have the same dtype, no change.
    * If both are ``pl.Enum`` and one's categories ⊆ the other's,
      cast the narrower side to the wider Enum dtype (``strict=False``
      — values outside the wider vocab become null, which inner / left
      joins drop or surface naturally).
    * If both are ``pl.Enum`` but neither vocab is a subset of the
      other, raise a clear ``ValueError`` pointing the caller to align
      explicitly (e.g. cast both sides to ``pl.Utf8`` or to a
      pre-built union Enum).
    * If one side is ``pl.Enum`` and the other is ``pl.Utf8`` /
      ``pl.String``, cast the string side to the Enum dtype
      (``strict=False``).
    * Other dtype mismatches are left untouched; polars's normal
      coercion rules apply (typically a clear error from polars).

    Returns the (possibly transformed) ``(left, right)`` pair.  Both
    eager and lazy frames are supported and the same kind is returned
    as was passed in.
    """
    if not on:
        return left, right
    left_out = left
    right_out = right
    for key in on:
        lt = _column_dtype(left_out, key)
        rt = _column_dtype(right_out, key)
        if lt is None or rt is None or lt == rt:
            continue
        l_is_enum = isinstance(lt, pl.Enum)
        r_is_enum = isinstance(rt, pl.Enum)
        if l_is_enum and r_is_enum:
            l_cats = set(lt.categories.to_list())
            r_cats = set(rt.categories.to_list())
            if l_cats <= r_cats:
                left_out = left_out.with_columns(pl.col(key).cast(rt, strict=False))
            elif r_cats <= l_cats:
                right_out = right_out.with_columns(pl.col(key).cast(lt, strict=False))
            else:
                raise ValueError(
                    f"cannot align Enum dtypes on join key {key!r}: "
                    f"left categories {sorted(l_cats)} and right "
                    f"categories {sorted(r_cats)} have no subset "
                    "relation. Cast one or both sides to a common "
                    "dtype before joining (e.g. pl.Utf8, or build a "
                    "union Enum and cast both sides to it)."
                )
        elif l_is_enum and rt in (pl.Utf8, pl.String):
            right_out = right_out.with_columns(pl.col(key).cast(lt, strict=False))
        elif r_is_enum and lt in (pl.Utf8, pl.String):
            left_out = left_out.with_columns(pl.col(key).cast(rt, strict=False))
        # else: leave untouched; polars's normal coercion handles it.
    return left_out, right_out


def _aligned_join(
    left: _Frame,
    right: _Frame,
    *,
    on: list[str] | tuple[str, ...] | None = None,
    left_on: list[str] | tuple[str, ...] | None = None,
    right_on: list[str] | tuple[str, ...] | None = None,
    how: str = "inner",
    suffix: str = "_right",
    coalesce: bool | None = None,
) -> _Frame:
    """Wrap ``polars.DataFrame.join`` / ``LazyFrame.join`` with
    automatic Enum-dtype alignment on shared keys.  See
    :func:`_align_enum_join_keys` for the alignment rules.

    Only ``on`` joins are alignment-aware (the common case in the
    kernel — symmetric join keys).  ``left_on`` / ``right_on`` joins
    pass through unchanged: by definition their key columns are named
    differently, so polars's strict Enum check fires per-column and a
    caller-side cast is the only consistent fix.
    """
    if on is not None:
        on_list = [on] if isinstance(on, str) else list(on)
        left, right = _align_enum_join_keys(left, right, on_list)
        kwargs: dict = {"on": on_list, "how": how, "suffix": suffix}
        if coalesce is not None:
            kwargs["coalesce"] = coalesce
        return left.join(right, **kwargs)
    kwargs = {"left_on": left_on, "right_on": right_on, "how": how, "suffix": suffix}
    if coalesce is not None:
        kwargs["coalesce"] = coalesce
    return left.join(right, **kwargs)


# ---------------------------------------------------------------------------
# Core types


class Param:
    """A parameter table.  ``frame`` carries columns ``*dims, value``.

    Stored internally as a ``polars.LazyFrame`` so that chained algebra
    ops (``Param * Param``, ``Param + Param`` etc.) defer materialization
    until a consumer reads ``.frame`` or the engine collects in
    ``Problem.solve``.  The ``.frame`` property caches the eager
    DataFrame on first read — flextool reads ``.frame.rename(...)``
    repeatedly off the same Param so we want that to be cheap.

    ``name`` (optional) is a logical Param identifier (e.g. "p_inflow").
    It is opt-in metadata used by :class:`WarmProblem`'s Param-tracked
    auto-update (``declare_mutable`` / ``update_param``).  When unset,
    Params are anonymous and carry no tracking overhead.

    ``_sources`` records constituent named Params for composite results
    of ``Param * Param`` / ``Param / Param``.  Each entry is
    ``(param_name, dims_tuple, direction)`` where direction is +1 if the
    Param contributes to the numerator and -1 if to the denominator.
    Anonymous-only chains have ``_sources is None``.
    """

    __slots__ = ("dims", "lazy", "_frame_cache", "name", "_sources", "_value_scalar")

    def __init__(
        self,
        dims: tuple[str, ...],
        frame: pl.DataFrame | pl.LazyFrame,
        name: str | None = None,
        _sources: list[tuple[Param, int]] | None = None,
        _value_scalar: float = 1.0,
    ):
        # accept either eager or lazy; store as lazy
        if isinstance(frame, pl.LazyFrame):
            lf = frame
            cols = lf.collect_schema().names()
        else:
            lf = frame.lazy()
            cols = frame.columns
        if "value" not in cols:
            raise ValueError(f"Param frame missing 'value' column; got {cols}")
        for d in dims:
            if d not in cols:
                raise ValueError(f"Param frame missing dim column {d!r}")
        self.dims = tuple(dims)
        self.lazy = lf.select(*dims, "value")
        self._frame_cache = None
        self.name = name
        self._sources = _sources
        # ``_value_scalar`` records the cumulative constant scalar
        # multiplied into / divided out of ``self.lazy``'s value column
        # outside the Param-chain factorisation.  Mirrors
        # :class:`_Term`'s ``coef_scalar``: when a Param composite is
        # rebuilt by the RHS prune-down by walking ``_sources`` and
        # joining each atomic Param's ``.lazy``, the scalar accumulated
        # by ``Param.__mul__(int/float)`` / ``Param.__truediv__
        # (int/float)`` / ``Param.__neg__`` is folded into the running
        # value via this field rather than re-walking the source list.
        # Atomic Params start at 1.0; composite chains multiply scalars
        # together (or invert for division).
        self._value_scalar = float(_value_scalar)

    def _own_sources(self) -> list[tuple[Param, int]] | None:
        """Return the list of ``(Param, direction)`` sources that this
        Param contributes when multiplied into a Var/Expr.  An anonymous
        composite uses ``_sources``; a named atomic Param contributes
        ``[(self, +1)]``.  Returns None when there is nothing to
        track."""
        if self._sources is not None:
            return self._sources
        if self.name is not None:
            return [(self, 1)]
        return None

    @property
    def frame(self) -> pl.DataFrame:
        """Eager DataFrame view; collects on first read, then caches."""
        if self._frame_cache is None:
            self._frame_cache = self.lazy.collect()
        return self._frame_cache

    @classmethod
    def scalar(cls, value: float) -> Param:
        return cls((), pl.DataFrame({"value": [float(value)]}))

    def __repr__(self) -> str:
        return f"Param(dims={self.dims}, n={self.frame.height})"

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Param(
                self.dims,
                self.lazy.with_columns(value=pl.col("value") * float(other)),
                _sources=self._sources_for_propagation(),
                _value_scalar=self._value_scalar * float(other),
            )
        if isinstance(other, Param):
            shared = [d for d in self.dims if d in other.dims]
            new_dims = tuple(dict.fromkeys(self.dims + other.dims))
            if shared:
                left_lf, right_lf = _align_enum_join_keys(self.lazy, other.lazy, shared)
                j = left_lf.join(right_lf, on=shared, how="inner", suffix="__r")
            else:
                j = self.lazy.join(other.lazy, how="cross", suffix="__r")
            merged = _merge_param_sources(
                self._sources_for_propagation(), other._sources_for_propagation(), flip_other=False
            )
            return Param(
                new_dims,
                j.with_columns(value=pl.col("value") * pl.col("value__r")).select(
                    *new_dims, "value"
                ),
                _sources=merged,
                _value_scalar=self._value_scalar * other._value_scalar,
            )
        if isinstance(other, (Var, Expr)):
            return other * self
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Param(
                self.dims,
                self.lazy.with_columns(value=pl.col("value") / float(other)),
                _sources=self._sources_for_propagation(),
                _value_scalar=self._value_scalar / float(other),
            )
        if isinstance(other, Param):
            shared = [d for d in self.dims if d in other.dims]
            new_dims = tuple(dict.fromkeys(self.dims + other.dims))
            if shared:
                left_lf, right_lf = _align_enum_join_keys(self.lazy, other.lazy, shared)
                j = left_lf.join(right_lf, on=shared, how="inner", suffix="__r")
            else:
                j = self.lazy.join(other.lazy, how="cross", suffix="__r")
            merged = _merge_param_sources(
                self._sources_for_propagation(), other._sources_for_propagation(), flip_other=True
            )
            return Param(
                new_dims,
                j.with_columns(value=pl.col("value") / pl.col("value__r")).select(
                    *new_dims, "value"
                ),
                _sources=merged,
                _value_scalar=self._value_scalar / other._value_scalar,
            )
        return NotImplemented

    def _sources_for_propagation(self) -> list[tuple[Param, int]] | None:
        """Return the constituent ``(Param, direction)`` list that this
        Param should contribute when merged into a chain via
        :meth:`__mul__` / :meth:`__truediv__`.

        Both named and anonymous atomic Params return ``[(self, +1)]``
        so the chain rebuild in the prune-down branches (RHS in
        :meth:`Problem._build_canonical_matrix`; LHS via
        :func:`_build_lhs_pruned_plan`) walks every constituent.
        Previously anonymous atomic Params returned ``None`` and were
        silently dropped by :func:`_merge_param_sources` — the
        composite's ``.lazy`` still carried their contribution, but the
        prune-down rebuilt the chain without them, producing the wrong
        coefficient/RHS value (off by the anonymous Param's value).

        Returns ``None`` only when ``_sources`` is explicitly ``None``
        AND ``self`` has nothing to contribute — currently never (the
        anonymous case now returns ``[(self, +1)]``).  Composite Params
        already carry their merged ``_sources``."""
        if self._sources is not None:
            return self._sources
        return [(self, 1)]

    def __neg__(self) -> Param:
        return self * -1.0

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Param(self.dims, self.lazy.with_columns(value=pl.col("value") + float(other)))
        if isinstance(other, Param):
            shared = [d for d in self.dims if d in other.dims]
            new_dims = tuple(dict.fromkeys(self.dims + other.dims))
            # Param + Param uses a full-outer join with 0-fill so a sparse
            # Param added to a dense Param does not drop the dense rows
            # outside the sparse key set.  (Cf. __mul__ / __truediv__ which
            # keep inner-join semantics — multiplying by a sparse Param IS
            # a legitimate "apply where defined" filter.)
            if shared:
                left_lf, right_lf = _align_enum_join_keys(self.lazy, other.lazy, shared)
                j = left_lf.join(right_lf, on=shared, how="full", suffix="__r", coalesce=True)
            else:
                j = self.lazy.join(other.lazy, how="cross", suffix="__r")
            return Param(
                new_dims,
                j.with_columns(
                    value=(pl.col("value").fill_null(0.0) + pl.col("value__r").fill_null(0.0))
                ).select(*new_dims, "value"),
            )
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other if isinstance(other, Param) else -float(other))

    def __rsub__(self, other):
        return (-self) + other


def _collect_streaming(plan: pl.LazyFrame) -> pl.DataFrame:
    """Collect a lazy plan via the polars streaming engine, with
    fallbacks to the legacy ``streaming=True`` kwarg and plain
    ``collect()``.  Shared by ``_build_canonical_matrix`` and
    ``WarmProblem._initial_build``'s tracked-source second pass.
    """
    try:
        return plan.collect(engine="streaming")
    except TypeError:
        try:
            return plan.collect(streaming=True)
        except TypeError:
            return plan.collect()
    except Exception:
        return plan.collect()


def _apply_where_frames(
    lazy: pl.LazyFrame,
    dims: tuple[str, ...] | list[str],
    where_frames: tuple[pl.LazyFrame, ...] | None,
) -> pl.LazyFrame:
    """Bake deferred :func:`Where` filter frames into ``lazy``.

    For each frame ``wf`` in ``where_frames`` (typically recorded by the
    pure-filter branch of :func:`Where`), compute the intersection of
    ``wf``'s columns with ``dims`` and semi-join on the shared keys.
    Frames with no shared dim are no-ops (the filter does not constrain
    this lazy plan's row set).  Used by Sum / Lag / LHS fallback paths
    to apply pending filters when leaf-rebuild prune-down can't fire.
    Mirrors :func:`_build_lhs_pruned_plan`'s use of
    :func:`_align_enum_join_keys` so Enum dtype mismatches don't slip
    through as silent correctness bugs.
    """
    if where_frames is None:
        return lazy
    dim_set = set(dims)
    for wf in where_frames:
        wf_cols = wf.collect_schema().names()
        shared = [c for c in wf_cols if c in dim_set]
        if not shared:
            continue
        keys = wf.select(shared).unique()
        lazy_a, keys_a = _align_enum_join_keys(lazy, keys, shared)
        lazy = lazy_a.join(keys_a, on=shared, how="semi")
    return lazy


def _apply_where_map_frames(
    lazy: pl.LazyFrame,
    dims: tuple[str, ...] | list[str],
    where_map_frames: tuple[tuple[pl.LazyFrame, frozenset[str]], ...] | None,
) -> tuple[pl.LazyFrame, tuple[str, ...]]:
    """Bake deferred map-effect :func:`Where` frames into ``lazy``.

    Each entry is ``(map_frame_lf, extras_frozenset)`` recorded by the
    map-effect branch of :func:`Where` (a frame whose columns introduce
    new open dims the expr did not carry, e.g. ``flow_to_n`` mapping
    ``(p, source, sink) → n``).  For each entry we inner-join the map
    frame onto ``lazy`` on the columns physically present in ``lazy``
    (filter + duplicate per matching row, exactly what the eager
    map-Where did), extending ``dims`` with the entry's extras in the
    frame's column order.  Returns ``(new_lazy, new_dims)``.

    Mirror of :func:`_apply_where_frames` but inner-join (not semi) and
    dim-extending.  Used by Sum / Lag / consumer-fallback paths and by
    :func:`_build_lhs_pruned_plan`'s final assembly step.  No-op (returns
    ``lazy`` plus ``tuple(dims)``) when ``where_map_frames`` is ``None``
    — the common case.

    The ``shared`` set is computed against the lazy plan's ACTUAL schema,
    not the claimed ``dims``: at deferral time the dim claim is recorded
    in ``_Term.dims`` but ``lazy`` does not yet physically carry the
    extras column until this bake.  When a frame shares no column with
    the lazy plan but has non-empty extras, we cross-join so the extras
    columns actually land (mirror of the latent-bug fix in
    :func:`Where`'s eager map-effect branch).
    """
    if where_map_frames is None:
        return lazy, tuple(dims)
    current_dims = list(dims)
    lazy_cols = set(lazy.collect_schema().names())
    for mf, extras in where_map_frames:
        mf_cols = mf.collect_schema().names()
        shared = [c for c in mf_cols if c in lazy_cols]
        if shared:
            lazy_a, mf_a = _align_enum_join_keys(lazy, mf, shared)
            lazy = lazy_a.join(mf_a, on=shared, how="inner")
        elif extras:
            lazy = lazy.join(mf, how="cross")
        # Preserve frame-column order — extras is an (unordered)
        # frozenset, so re-iterate the frame's column list.
        ordered_extras = [c for c in mf_cols if c in extras]
        for c in ordered_extras:
            if c not in current_dims:
                current_dims.append(c)
        for c in mf_cols:
            lazy_cols.add(c)
    return lazy, tuple(current_dims)


def _bake_map_before_mul(
    t: _Term,
    factor_dims: tuple[str, ...],
) -> tuple[pl.LazyFrame, tuple[str, ...], tuple[tuple[pl.LazyFrame, frozenset[str]], ...] | None]:
    """Decide whether a Param multiply must bake the term's pending
    map-effect Where frames before the join.

    A deferred map-effect frame leaves ``t.dims`` claiming the extras
    columns, but ``t.lazy`` does not physically carry them yet.  If the
    multiplying factor's dims overlap any of those pending extras, the
    post-Where dim set is the source of truth for ``shared`` — we MUST
    bake the map frames first so ``t.lazy`` carries the extras columns
    before the Param join (correctness).  Otherwise the deferral
    propagates through unchanged (the win — the leaf-rebuild can still
    defer the map-join).

    Returns ``(use_lazy, use_dims, out_where_map_frames)``.
    """
    if t.where_map_frames is None:
        return t.lazy, t.dims, None
    pending_extras: frozenset[str] = frozenset().union(
        *(extras for (_, extras) in t.where_map_frames)
    )
    if pending_extras and (set(factor_dims) & pending_extras):
        baked_lazy, baked_dims = _apply_where_map_frames(t.lazy, t.dims, t.where_map_frames)
        return baked_lazy, baked_dims, None
    return t.lazy, t.dims, t.where_map_frames


def _build_lhs_pruned_plan(
    row_index_lf: pl.LazyFrame,
    axis_cols: list[str],
    var_source: Var,
    param_sources: list[tuple[Param, int]],
    on: list[str],
    coef_scalar: float = 1.0,
    where_frames: tuple[pl.LazyFrame, ...] | None = None,
    where_map_frames: tuple[tuple[pl.LazyFrame, frozenset[str]], ...] | None = None,
) -> pl.LazyFrame:
    """Rebuild a Var × Param … LHS term as a prune-down chain, mirroring
    the RHS prune-down in :meth:`Problem._build_canonical_matrix`.

    The fully-merged ``term.lazy`` produced by :meth:`Var.__mul__` +
    :meth:`Expr.__mul__` is a chain of inner joins ``Var ⋈ P1 ⋈ P2 …``.
    For wide constraint families with multi-Param chains, that inner
    join chain can materialise wide intermediates the polars optimizer
    does not push the row_index semi-join through (the same bug class
    the RHS prune-down fixes).  Here we rebuild the chain step by step,
    starting from ``row_index ⋈ Var.frame`` (bounded by the constraint's
    row_index key set) and joining each atomic Param one at a time —
    each step's row count is bounded by the row_index projection.

    Returns a lazy plan with columns ``(_rid, col_id, coef)`` ready to
    collect.  Callers handle the streaming collect + side-vector BAKE
    pass.  ``on`` is the join key (intersection of term.dims and
    axis_cols) used to attach the resulting per-cell rows back to
    row_index — same semantics as the unpruned path's
    ``row_index ⋈ term.lazy`` final join.

    ``where_frames`` carries deferred filter frames recorded by
    pure-filter :func:`Where` (see :func:`_apply_where_frames`).  They
    are baked into the rebuilt chain at two points: against ``var_lf``
    up front (narrows the seed) and against each atomic accumulator
    step (narrows mid-chain when a later filter shares a dim that
    enters via a Param).  Both steps semi-join only — they never alter
    coef values, only the row set the rebuilt chain spans.

    ``where_map_frames`` carries deferred *map-effect* :func:`Where`
    frames.  Each entry is ``(frame_lf, extras_frozenset)``.  Their
    shared cols narrow the seed AND each per-atomic accumulator via
    semi-join (exactly like ``where_frames``); the dim-extending
    inner-join (which produces the extras columns) is applied via
    :func:`_apply_where_map_frames` AFTER the chain rebuild, just before
    the final row_index attach — at which point ``on`` may include the
    extras columns (they are now present in the accumulator).
    """
    var_dims = list(var_source.dims)
    var_on = [d for d in var_dims if d in axis_cols]
    var_lf = var_source.frame.lazy()
    # Map-effect frames: their dim-extending inner-join is applied after
    # the chain rebuild via _apply_where_map_frames.  Up front we only
    # use the frame components to narrow the seed by their shared cols
    # (mirror of where_frames) so the rebuilt chain stays bounded.
    _map_semi = tuple(f for (f, _) in where_map_frames) if where_map_frames else ()
    if where_frames is not None:
        # Apply any where_frames whose shared cols overlap the Var's
        # dims up front — narrows the seed before row_index pre-prune.
        var_lf = _apply_where_frames(var_lf, var_dims, where_frames)
    if _map_semi:
        var_lf = _apply_where_frames(var_lf, var_dims, _map_semi)
    if var_on:
        # Pre-prune the Var.frame against the row_index key set so the
        # very first step is bounded by the constraint's row count.
        ri_keys = row_index_lf.select(var_on).unique()
        var_lf_a, ri_keys_a = _align_enum_join_keys(var_lf, ri_keys, var_on)
        var_lf = var_lf_a.join(ri_keys_a, on=var_on, how="semi")
    acc = var_lf.with_columns(coef=pl.lit(float(coef_scalar), dtype=pl.Float64)).select(
        *var_dims, "col_id", "coef"
    )
    acc_dims: list[str] = list(var_dims)
    for atomic, direction in param_sources:
        shared = [d for d in acc_dims if d in atomic.dims]
        new_dims = list(dict.fromkeys(acc_dims + list(atomic.dims)))
        atomic_lf = atomic.lazy
        if shared:
            # Pre-prune the atomic Param by the accumulator's key set —
            # mirror of the RHS prune-down's per-atomic semi-join so the
            # join below stays bounded by ``acc`` height.
            acc_a, atomic_a = _align_enum_join_keys(acc, atomic_lf, shared)
            acc_keys = acc_a.select(shared).unique()
            acc_keys_a, atomic_a2 = _align_enum_join_keys(acc_keys, atomic_a, shared)
            atomic_pruned = atomic_a2.join(acc_keys_a, on=shared, how="semi")
            acc_b, atomic_pruned_b = _align_enum_join_keys(acc_a, atomic_pruned, shared)
            joined = acc_b.join(
                atomic_pruned_b,
                on=shared,
                how="inner",
                suffix="__lhs_chain",
            )
        else:
            joined = acc.join(atomic_lf, how="cross", suffix="__lhs_chain")
        # ``value`` is the Param's coefficient column.  Either it was
        # left unrenamed (no name clash) or, if ``coef`` happened to be
        # present on the right side already (it never is for a Param
        # frame — Param frames carry ``(*dims, value)``), the suffixed
        # name would apply.  The inner-join Param branch above does not
        # rename ``value`` so the direct reference is correct.
        if direction >= 0:
            acc = joined.with_columns(coef=pl.col("coef") * pl.col("value")).select(
                *new_dims, "col_id", "coef"
            )
        else:
            acc = joined.with_columns(coef=pl.col("coef") / pl.col("value")).select(
                *new_dims, "col_id", "coef"
            )
        acc_dims = new_dims
        if where_frames is not None:
            # After each atomic step, narrow the accumulator by any
            # where_frames that now overlap with new acc_dims — mirror of
            # the row_index pre-prune so the chain stays bounded.
            acc = _apply_where_frames(acc, acc_dims, where_frames)
        if _map_semi:
            # Same per-atomic semi-join narrowing for the map-effect
            # frames — their dim-extending inner-join lands AFTER the
            # chain (below) so the extras columns appear at the leaf.
            acc = _apply_where_frames(acc, acc_dims, _map_semi)
    if where_map_frames is not None:
        # Bake the map-effect inner-joins: produces the extras dim
        # columns and extends acc_dims with them, so the final attach's
        # ``on`` (which may now include extras) finds the columns.
        acc, acc_dims_t = _apply_where_map_frames(acc, acc_dims, where_map_frames)
        acc_dims = list(acc_dims_t)
    # Final attach to row_index: same shape as the unpruned path.
    rl_a, acc_a = _align_enum_join_keys(row_index_lf, acc, on)
    return rl_a.join(acc_a, on=on, how="inner")


def _prune_down_disabled() -> bool:
    """Return True when the user has set POLAR_HIGH_DISABLE_PRUNE_DOWN=1.

    Both the RHS prune-down (composite-Param ``_sources`` walk in
    :meth:`Problem._build_canonical_matrix`) and the LHS prune-down
    (``_build_lhs_pruned_plan`` invocations in
    ``_build_canonical_matrix`` / ``_solve_streaming`` /
    ``WarmProblem._initial_build``) honour this flag so users can fall
    back to the original merged-lazy paths verbatim when the prune-down
    optimisation produces unexpected drift.
    """
    return os.environ.get("POLAR_HIGH_DISABLE_PRUNE_DOWN") == "1"


def _block_coo_disabled() -> bool:
    """Return True when ``POLAR_HIGH_DISABLE_BLOCK_COO=1``.

    Mirror of :func:`_prune_down_disabled`.  When set, the block-COO LHS
    evaluation arm (:func:`_block_coo_classify` /
    :func:`_build_block_coo_plan` in
    :meth:`Problem._build_canonical_matrix`) is skipped entirely and the
    term falls through to the existing LHS prune-down / fallback paths
    verbatim.  Block-COO is default ON (see ``specs/block_coo_DECISIONS.md``
    D5) and bit-identical to the polars path, so this flag exists only as
    an escape hatch / parity-test lever, not because the two paths can
    disagree numerically.
    """
    return os.environ.get("POLAR_HIGH_DISABLE_BLOCK_COO") == "1"


def _block_coo_min_dense() -> int:
    """Dense-axis cardinality threshold for firing block-COO.

    Read from ``POLAR_HIGH_BLOCK_COO_MIN_DENSE`` (env), fallback 100.
    A malformed value falls back to 100 rather than raising — the gate
    is a heuristic, not a correctness lever (correctness is guaranteed by
    :func:`_block_coo_classify`'s conservative shape checks).
    """
    raw = os.environ.get("POLAR_HIGH_BLOCK_COO_MIN_DENSE")
    if raw is None:
        return 100
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return 100
    return v if v > 0 else 100


def _verify_dense_sorted(
    frame,
    lead_dims: list[str],
    dense_axes,
    var_name: str | None = None,
) -> None:
    """Cheaply verify the client's dense-axis sort contract on ``frame``.

    The :class:`Problem` ``dense_axes`` contract (see ``Problem.__init__``)
    promises that any client frame carrying the dense axes is globally
    lexicographically sorted by ``lead_dims + list(dense_axes)`` — i.e.
    the dense axes are the trailing sort keys (in declared order) and the
    leading dims form a sorted prefix.  Block-COO slices the dense suffix
    with NO re-sort, so a broken promise would silently produce wrong
    coefficients; this verifier makes it a loud, immediate error instead.

    Implementation: a SINGLE-PASS monotonic check, NOT a sort.  We build a
    struct column over ``lead_dims + dense_axes`` (polars struct compares
    lexicographically, field-by-field in declaration order) and call
    :meth:`polars.Series.is_sorted`, which is an O(n) "is each row >= its
    predecessor?" scan — it never reorders the data.  ``frame`` may be a
    :class:`polars.DataFrame` or a lazy frame (collected here; block-COO
    needs the Var eager anyway, and the verifier touches only the key
    columns).

    Raises ``ValueError`` naming the originating Var/frame and restating
    the contract when the scan finds a descent.
    """
    keys = list(lead_dims) + list(dense_axes)
    df = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
    if df.height <= 1:
        return
    sorted_ok = df.select(pl.struct(keys).alias("__bc_key")).to_series().is_sorted()
    if not sorted_ok:
        who = f" for Var {var_name!r}" if var_name else ""
        raise ValueError(
            f"block-COO dense_axes contract violated{who}: the frame is not "
            f"lexicographically sorted by {tuple(keys)} "
            f"(leading dims {tuple(lead_dims)} then dense axes "
            f"{tuple(dense_axes)}).  The Problem was constructed with "
            f"dense_axes={tuple(dense_axes)}, which promises every frame "
            "carrying those columns is already row-sorted with the dense "
            "axes as the trailing sort keys; polar-high relies on that to "
            "slice the dense suffix without re-sorting.  Re-sort the frame "
            "by (lead..., *dense_axes) before passing it, or do not declare "
            "these dense_axes."
        )


def _block_coo_classify(
    term: _Term,
    axis_cols: list[str],
    on: list[str],
    dense_axes,
) -> dict | None:
    """Classify whether a non-Sum ``Var × Param-chain`` LHS term is
    block-COO evaluable against the client-DECLARED dense suffix.  Returns
    a small spec ``dict`` if it is, else ``None`` (caller falls back to the
    existing path).

    The dense set is no longer GUESSED by cardinality — it is exactly the
    ``dense_axes`` the client declared on the :class:`Problem` (see
    ``Problem.__init__``).  A term is block-evaluable iff ALL of:

    * ``dense_axes`` is non-empty (the client opted in to the contract).
    * ``term.var_source is not None`` — guarantees a non-Sum / non-Lag /
      non-map-Where term (those clear ``var_source``), so there is an
      unreduced ``Var × P1 × P2 …`` chain to slice.
    * ``term.param_sources`` is a non-empty list of ``(Param, dir)`` with
      ``dir ∈ {1, -1}``.
    * Every Param's dims are a subset of ``var.dims ∪ axis_cols`` — no
      foreign dim that the block alignment can't account for.  Low-dim /
      broadcast Params (e.g. ``Pb(p)``) are explicitly ALLOWED: they
      broadcast, and the existing join-based builder handles them
      correctly — the dense set is the declared suffix, not the factor
      intersection.
    * ``on ⊆ var.dims`` (the block-COO seed carries only var dims; if a
      join key came from a Param the seed would mis-key — fall back).
    * **The Var's dims END WITH ``dense_axes`` in the declared order**:
      ``tuple(var.dims[-len(dense_axes):]) == tuple(dense_axes)``.  This is
      the suffix contract.  A Var lacking the dense axes, or carrying them
      non-trailing (e.g. an investment Var ``("p", "d")`` does not end in
      ``("d", "t")``), does NOT fire — it falls back, correctly.

    Conservative by construction: any shape the block loop cannot
    reproduce bit-identically must return ``None`` here.  The firing
    decision is a PERFORMANCE choice (block-COO is bit-identical to the
    polars path); false negatives (fall back, slower) are always safe.
    """
    if not dense_axes:
        return None
    dense_axes = list(dense_axes)

    var = term.var_source
    if var is None:
        return None
    psrc = term.param_sources
    if not isinstance(psrc, list) or len(psrc) == 0:
        return None

    var_dims = list(var.dims)
    var_dim_set = set(var_dims)
    axis_set = set(axis_cols)
    allowed = var_dim_set | axis_set

    # Every Param must carry a (Param, direction) pair with a real Param.
    for entry in psrc:
        if not (isinstance(entry, tuple) and len(entry) == 2):
            return None
        atomic, direction = entry
        if not isinstance(atomic, Param):
            return None
        if direction not in (1, -1):
            # Only the +1 numerator / -1 denominator directions are
            # reproduced by the block multiply loop.
            return None
        if not set(atomic.dims).issubset(allowed):
            # Foreign dim — block alignment can't account for it.
            return None

    # The block-COO seed carries only the Var's dims; the final attach to
    # row_index joins on ``on``.  If a join key in ``on`` came from a Param
    # rather than the Var (e.g. ``Var(p,d) × Pa(d,t)`` where ``t ∈ on`` but
    # ``t ∉ var.dims``), the seed would lack that column and the final join
    # would mis-key.  The polars prune-down path widens its accumulator
    # through the chain so it keeps such dims; block-COO does not, so we
    # conservatively fall back when ``on`` is not a subset of ``var.dims``.
    if not set(on).issubset(var_dim_set):
        return None

    # Suffix contract: the Var's dims must END WITH the declared dense_axes
    # in the declared order.  This is what lets block-COO slice the dense
    # suffix with no re-sort under the client's sort promise.  No firing
    # otherwise (fall back) — the firing decision is perf-only.
    if len(dense_axes) > len(var_dims):
        return None
    if tuple(var_dims[-len(dense_axes) :]) != tuple(dense_axes):
        return None

    dense_dims = list(dense_axes)
    non_dense_dims = [d for d in var_dims if d not in set(dense_axes)]

    # OPTIONAL secondary perf gate on the dense-axis cardinality.  Upper
    # bound = Var frame height (one row per var cell; dense ⊆ var.dims so
    # distinct dense keys <= var height).  This is PERF-ONLY — it never
    # affects correctness or the broadcast case; if the height can't be
    # read we simply skip the gate and fire on the suffix match.
    try:
        dense_card = int(var.frame.height)
    except Exception:
        dense_card = None
    if dense_card is not None and dense_card < _block_coo_min_dense():
        return None

    return {
        "var_dims": var_dims,
        "dense_dims": dense_dims,
        "non_dense_dims": non_dense_dims,
        "dense_card": dense_card if dense_card is not None else 0,
        "on": list(on),
    }


def _emit_block_coo_path(path: str, reason: str = "") -> None:
    """Emit a one-line ``path=`` profile signal naming which evaluation
    arm :func:`_build_block_coo_plan` took (``positional`` vs ``joined``).

    Gated by ``POLAR_HIGH_BLOCK_COO_PROFILE=1`` (same lever as the
    ``phase=block_coo_term`` line emitted at the dispatch site).  This is
    real instrumentation — it tells an operator whether the fast
    positional slice-multiply fired or the builder fell back to the
    order-preserving joined backstop, and why — not a test-only hook.
    """
    if os.environ.get("POLAR_HIGH_BLOCK_COO_PROFILE") != "1":
        return
    extra = f"\treason={reason}" if reason else ""
    sys.stderr.write(f"[block_coo profile]\tpath={path}{extra}\n")
    sys.stderr.flush()


def _block_coo_keep_cols(keep_dims: tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalise the optional ``keep_dims`` into the extra final-select
    columns (empty tuple when None).  Shared by both block-COO builders so
    the warm-path Site-3 caller gets ``(_rid, col_id, coef, *keep_dims)``
    while Sites 1/2 (``keep_dims=None``) stay ``(_rid, col_id, coef)``."""
    return tuple(keep_dims) if keep_dims else ()


def _empty_block_coo_frame(
    schema_src: pl.DataFrame, keep_dims: tuple[str, ...] | None
) -> pl.DataFrame:
    """Build the empty ``(_rid, col_id, coef[, *keep_dims])`` frame both
    builders emit when nothing survives.  ``keep_dims`` columns adopt their
    dtypes from ``schema_src`` (the pre-sorted seed / aligned frame, which
    carries the Var dims) so the warm tracker's downstream re-join keys
    align with the populated case."""
    cols: dict[str, pl.Series] = {
        "_rid": pl.Series("_rid", [], dtype=pl.Int64),
        "col_id": pl.Series("col_id", [], dtype=pl.Int64),
        "coef": pl.Series("coef", [], dtype=pl.Float64),
    }
    for d in _block_coo_keep_cols(keep_dims):
        cols[d] = pl.Series(d, [], dtype=schema_src.schema[d])
    return pl.DataFrame(cols)


def _build_block_coo_plan_joined(
    row_index_lf: pl.LazyFrame,
    axis_cols: list[str],
    var_source: Var,
    param_sources: list[tuple[Param, int]],
    on: list[str],
    coef_scalar: float,
    where_frames: tuple[pl.LazyFrame, ...] | None,
    dense_spec: dict,
    keep_dims: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Evaluate a non-Sum ``Var × Param-chain`` LHS term via block-COO.

    Produces an eager :class:`polars.DataFrame` with columns
    ``(_rid, col_id, coef)`` — identical in shape to
    :func:`_build_lhs_pruned_plan`'s output, so the existing emission
    code in :meth:`Problem._build_canonical_matrix` consumes it
    unchanged.

    ``keep_dims`` (the warm-path Site-3 caller) appends those dim columns
    to the final ``.select(...)`` *in addition* to ``(_rid, col_id,
    coef)``, so the warm param-tracking machinery can re-join each tracked
    Param on its dims.  Every ``keep_dims`` entry is a subset of
    ``var.dims`` for a block-evaluable term, so it rides on the seed
    (``*var_dims``) and survives to the final join.  Default ``None`` ⇒
    the unchanged ``(_rid, col_id, coef)`` shape (Sites 1/2).

    Algorithm (mirrors :func:`_build_lhs_pruned_plan`'s leaf discipline —
    per-leaf Enum alignment + row_index pre-prune — but does the final
    coefficient multiply in numpy on contiguous, key-aligned value
    buffers instead of carrying value columns through wide joins):

    1. Bake ``where_frames`` onto the Var seed and pre-prune the Var
       against ``row_index_lf``'s key set (semi-join), exactly as the
       prune helper.  Then re-apply ``where_frames`` after the seed is
       widened (no widening happens here — the seed keeps only var dims —
       but the pre-prune narrows it).
    2. Assemble ONE aligned accumulator by mirroring the polars
       prune-down chain's join sequence EXACTLY: starting from the pruned
       Var seed, **inner-join** each atomic Param's ``value`` (renamed
       uniquely ``__bc_val_{i}``) onto the accumulator on the shared key
       (after the same Enum-align + semi-join-narrow the prune helper
       does).  The inner join means a SPARSE Param drops the unmatched
       rows — identical to the polars chain — so this NEVER crashes on
       missing/sparse coefficient data.  The surviving row set + order are
       therefore identical to :func:`_build_lhs_pruned_plan`'s.
    3. Read each ``__bc_val_{i}`` column into a contiguous numpy buffer
       and multiply in the SAME left-to-right order as the chain rebuild
       (seed ``coef_scalar``, then ``*value`` for direction ``>= 0`` /
       ``/value`` for direction ``< 0``).  Same IEEE-double ops, same
       order, same row set ⇒ bit-identical to the polars path.  Only the
       final arithmetic moves from polars to numpy.
    4. Final inner-join to ``row_index_lf`` on ``on`` (Enum-aligned) to
       attach ``_rid``; return selecting ``_rid, col_id, coef``.

    Step 3 is vectorised over the whole aligned frame (one ufunc per
    factor) rather than looping Python over RLE blocks: because every
    value column rode the same join chain, the i-th row of every buffer
    refers to the same surviving cell, so a single elementwise
    ``coef *= value`` reproduces the per-cell chain product exactly.
    ``dense_spec`` records the block geometry for the profile emitter.

    Crash-free by construction: there is no left-join + raise-on-null any
    more.  A sparse Param simply contributes fewer surviving rows via its
    inner join — precisely what the polars prune-down chain does — so a
    missing coefficient is dropped, not fatal.
    """
    var_dims = list(var_source.dims)
    var_on = [d for d in var_dims if d in axis_cols]

    # --- Step 1: bake where_frames + pre-prune the Var seed.
    var_lf = var_source.frame.lazy()
    if where_frames is not None:
        var_lf = _apply_where_frames(var_lf, var_dims, where_frames)
    if var_on:
        ri_keys = row_index_lf.select(var_on).unique()
        var_lf_a, ri_keys_a = _align_enum_join_keys(var_lf, ri_keys, var_on)
        var_lf = var_lf_a.join(ri_keys_a, on=var_on, how="semi")
    # where_frames may share a dim that only enters via a Param; for a
    # non-Sum chain every Param dim is a subset of var.dims ∪ axis_cols
    # (classifier guarantee) and the seed already spans var.dims, so a
    # second bake against var_dims is the complete set — mirror the
    # prune helper's "re-apply after widening" defensively.
    if where_frames is not None:
        var_lf = _apply_where_frames(var_lf, var_dims, where_frames)

    seed = var_lf.select(*var_dims, "col_id")

    # --- Step 2: assemble ONE aligned frame by mirroring the polars
    # prune-down chain's join sequence EXACTLY, then read the value
    # columns into numpy.  The crucial parity property: the surviving row
    # set (and the per-factor value carried on each row) must be identical
    # to what ``_build_lhs_pruned_plan`` produces, so that the only thing
    # that moves from polars to numpy is the final arithmetic.
    #
    # The prune helper's chain is, per atomic ``(atomic, direction)``:
    #   * pre-prune the atomic by the accumulator's shared-key set (semi),
    #   * **inner-join** the atomic onto the accumulator on the shared key,
    #   * carry the Param's ``value``.
    # An INNER join is what makes a SPARSE Param drop the unmatched seed
    # rows (instead of crashing on a null).  We reproduce that here:
    # rather than left-join + raise-on-null, we inner-join each Param's
    # ``value`` (renamed uniquely as ``__bc_val_{i}``) onto a running
    # accumulator.  The accumulator's row set therefore shrinks exactly as
    # the polars chain's would, and never carries a null value column.
    #
    # We keep ``col_id`` on the accumulator throughout and never widen
    # past var dims (every Param dim ⊆ var.dims ∪ axis_cols by the
    # classifier guarantee, and the dense/shared dims that drive the join
    # are all ⊆ var.dims), so the accumulator stays ``(*var_dims, col_id,
    # __bc_val_0, …)``.
    dense_dims = list(dense_spec["dense_dims"])
    non_dense_dims = list(dense_spec["non_dense_dims"])
    sort_keys = non_dense_dims + dense_dims

    acc = seed
    val_cols: list[str] = []
    for i, (atomic, _direction) in enumerate(param_sources):
        shared = [d for d in var_dims if d in atomic.dims]
        val_col = f"__bc_val_{i}"
        atomic_lf = atomic.lazy.rename({"value": val_col})
        if shared:
            # Pre-prune the atomic Param by the accumulator's key set
            # (semi-join) then inner-join its value — mirror of the prune
            # helper's per-atomic discipline, with the SAME inner-join
            # drop semantics for sparse Params.
            acc_a, atomic_a = _align_enum_join_keys(acc, atomic_lf, shared)
            acc_keys = acc_a.select(shared).unique()
            acc_keys_a, atomic_a2 = _align_enum_join_keys(acc_keys, atomic_a, shared)
            atomic_pruned = atomic_a2.join(acc_keys_a, on=shared, how="semi")
            acc_b, atomic_pruned_b = _align_enum_join_keys(acc_a, atomic_pruned, shared)
            acc = acc_b.join(
                atomic_pruned_b,
                on=shared,
                how="inner",
                suffix="__bc_chain",
            )
        else:
            # No shared dim — the chain cross-joins the Param.  This is
            # bit-identical to a per-row broadcast of every Param value;
            # the classifier admits only foreign-free Params, but a
            # multi-row no-shared Param would cross-multiply the row set
            # (exactly as the polars chain's cross-join would).  We mirror
            # polars rather than special-casing: a cross-join carries the
            # value onto every accumulator row.
            acc = acc.join(atomic_lf, how="cross", suffix="__bc_chain")
        val_cols.append(val_col)

    # Canonical sort so the numpy multiply is deterministic and the
    # surviving-row order is pinned (independent of polars's join order).
    acc_sorted = acc.sort(sort_keys).collect()

    n = acc_sorted.height
    _emit_block_coo_path("joined")
    if n == 0:
        # Nothing survived the chain (empty seed or a sparse Param dropped
        # every row) — emit an empty (_rid, col_id, coef[, *keep_dims])
        # frame matching the prune helper's output schema.  keep_dims
        # columns adopt their dtypes from the (now-empty) sorted frame so
        # the warm tracker's downstream re-join keys align.
        return _empty_block_coo_frame(acc_sorted, keep_dims)

    # --- Step 3: numpy multiply in chain order on the aligned buffers.
    # Every value column lives on the SAME row as its ``col_id`` (they
    # rode the same join chain), so the i-th row of every buffer refers to
    # the same surviving cell — a single elementwise op per factor
    # reproduces the per-cell chain product, in the SAME left-to-right
    # order and with the SAME IEEE-double ops as the polars chain.
    coef = np.full(n, float(coef_scalar), dtype=np.float64)
    for (_atomic, direction), val_col in zip(param_sources, val_cols):
        vals = acc_sorted[val_col].to_numpy().astype(np.float64, copy=False)
        if direction >= 0:
            coef = coef * vals
        else:
            coef = coef / vals

    # --- Step 4: attach _rid via row_index inner-join on ``on``.
    result = acc_sorted.select(*var_dims, "col_id").with_columns(
        coef=pl.Series("coef", coef, dtype=pl.Float64)
    )
    ri_a, res_a = _align_enum_join_keys(row_index_lf, result.lazy(), on)
    joined_ri = ri_a.join(res_a, on=on, how="inner")
    return joined_ri.select("_rid", "col_id", "coef", *_block_coo_keep_cols(keep_dims)).collect()


def _build_block_coo_plan(
    row_index_lf: pl.LazyFrame,
    axis_cols: list[str],
    var_source: Var,
    param_sources: list[tuple[Param, int]],
    on: list[str],
    coef_scalar: float,
    where_frames: tuple[pl.LazyFrame, ...] | None,
    dense_spec: dict,
    keep_dims: tuple[str, ...] | None = None,
    *,
    dense_param_vectors: dict[int, np.ndarray] | None = None,
) -> pl.DataFrame:
    """Evaluate a non-Sum ``Var × Param-chain`` LHS term via *positional*
    per-block numpy slice-multiply on the already-sorted Var grid, with NO
    re-sort — falling back to :func:`_build_block_coo_plan_joined` (the
    always-correct, order-preserving join backstop) the instant positional
    alignment cannot be guaranteed.

    Same signature and return contract as the joined builder: an eager
    :class:`polars.DataFrame` with columns ``(_rid, col_id, coef)``, ready
    for the existing emission code in
    :meth:`Problem._build_canonical_matrix`.

    Why positional is faster
    -------------------------
    The joined builder is correct but pays an ``O(n log n)`` re-sort
    (``acc.sort(sort_keys)``) because polars joins do not preserve order,
    and materialises every Param's value at full grid resolution.  When the
    Var seed is dense-complete (see the completeness guard below) every
    leading-dim block holds the SAME complete, identically-ordered set of
    dense-axis tuples, so a dense-resolution factor aligns *positionally*
    across blocks: we slice/tile/repeat numpy buffers onto the pre-sorted
    seed and multiply in place — no join-induced reorder, no re-sort.

    Bit-identity
    ------------
    The coef is seeded with ``coef_scalar`` and multiplied STRICTLY in
    ``param_sources`` order (``*value`` for direction ``>= 0`` / ``/value``
    for ``< 0``) — the exact same IEEE-double op sequence
    :func:`_build_lhs_pruned_plan` and :func:`_build_block_coo_plan_joined`
    use.  Same row set (the seed, pre-pruned identically) + same per-row
    factor values + same multiply order ⇒ bit-identical coefficients.

    Completeness guard (positional vs fallback)
    --------------------------------------------
    Positional slicing is valid only when every leading-dim block contains
    the same complete, identically-ordered dense-axis tuple set.  Cheap
    sufficient test, given the caller has already verified the seed is
    lexicographically sorted by ``(non_dense_dims..., dense_dims...)``:

        n == n_lead * n_dense   AND   where_frames is None

    where ``n = seed.height``, ``n_lead`` = distinct ``non_dense_dims``
    tuples, ``n_dense`` = distinct ``dense_dims`` tuples.  If the seed is
    sorted and ``n == n_lead * n_dense`` then each of the ``n_lead`` blocks
    is exactly the full ``n_dense``-row dense set in the same order (a
    sorted frame with no row spillover can only be the full cartesian
    product).  Any ``where_frames`` (a deferred pure-filter) can carve the
    grid sparse/ragged, so its mere presence forces the fallback.

    Anything else — sparse seed, ragged blocks, a filter present, or any
    per-Param alignment that cannot be proven length-exact — returns
    ``_build_block_coo_plan_joined(...)``.  False fallbacks cost speed,
    never correctness.

    ``keep_dims`` (the warm-path Site-3 caller) appends those dim columns
    to the final ``.select(...)`` in addition to ``(_rid, col_id, coef)``
    so the warm param-tracking machinery can re-join each tracked Param on
    its dims.  For a block-evaluable term every ``keep_dims`` entry is a
    subset of ``var.dims`` (it rides on the seed and survives to the final
    join); we verify that and fall back to the joined builder otherwise.
    Default ``None`` ⇒ the unchanged ``(_rid, col_id, coef)`` shape.

    ``dense_param_vectors`` (perf hoist for the bounded coefficient walk) is
    an optional ``{id(atomic): sorted-dense-value-array}`` cache for the
    dense-only Param case: a dense-only Param's value vector (sorted by
    ``dense_dims``) is BATCH-INVARIANT (the same vector is tiled onto every
    leading block of every batch), so the walk collects it ONCE per term and
    threads it here, letting each batch tile from the cached numpy buffer
    instead of re-running ``atomic.lazy.collect().sort(dense_dims)``.  When a
    Param is absent from the cache (or the cache is ``None``) the builder
    collects it inline exactly as before — the cache is a pure speed hoist,
    bit-identical (same sorted vector either way).  Default ``None`` ⇒ the
    unchanged per-call collect (canonical-matrix callers).
    """
    var_dims = list(var_source.dims)
    var_on = [d for d in var_dims if d in axis_cols]
    dense_dims = list(dense_spec["dense_dims"])
    non_dense_dims = list(dense_spec["non_dense_dims"])

    _fallback_args = (
        row_index_lf,
        axis_cols,
        var_source,
        param_sources,
        on,
        coef_scalar,
        where_frames,
        dense_spec,
        keep_dims,
    )

    # ``keep_dims`` must ride on the Var seed (``*var_dims``) to survive to
    # the final select.  For a block-evaluable term ``term.dims ⊆ var.dims``
    # holds, but guard defensively: if any requested dim is absent, the
    # order-preserving joined builder (which carries the same dims) is the
    # safe backstop.
    if keep_dims and not set(keep_dims).issubset(var_dims):
        return _build_block_coo_plan_joined(*_fallback_args)

    # --- Step 1: bake where_frames + pre-prune the Var seed.  Filtering
    # (semi-join) preserves the seed's sort order, so the result stays
    # lexicographically sorted by (non_dense_dims..., dense_dims...) — just
    # possibly sparse.  Block-COO needs the seed eager.
    var_lf = var_source.frame.lazy()
    if where_frames is not None:
        var_lf = _apply_where_frames(var_lf, var_dims, where_frames)
    if var_on:
        ri_keys = row_index_lf.select(var_on).unique()
        var_lf_a, ri_keys_a = _align_enum_join_keys(var_lf, ri_keys, var_on)
        var_lf = var_lf_a.join(ri_keys_a, on=var_on, how="semi")
    seed = var_lf.select(*var_dims, "col_id").collect()

    n = seed.height
    if n == 0:
        # Empty seed — emit the empty (_rid, col_id, coef[, *keep_dims])
        # frame directly (cheaper than recursing; the joined builder
        # produces the same).  keep_dims dtypes come from the seed schema.
        _emit_block_coo_path("positional", reason="empty_seed")
        return _empty_block_coo_frame(seed, keep_dims)

    # --- Step 2: completeness guard.  A deferred filter (where_frames) can
    # leave the grid sparse/ragged ⇒ fall back.  Otherwise the seed is
    # sorted (caller-verified) and dense-complete iff n == n_lead * n_dense.
    if where_frames is not None:
        return _build_block_coo_plan_joined(*_fallback_args)

    n_dense = seed.select(dense_dims).n_unique()
    n_lead = seed.select(non_dense_dims).n_unique() if non_dense_dims else 1
    if n != n_lead * n_dense:
        # Sparse or ragged seed — blocks are not the full dense set in
        # identical order ⇒ positional slicing would mis-align.
        return _build_block_coo_plan_joined(*_fallback_args)

    # --- Step 3: positional path.  Block size = n_dense; there are n_lead
    # blocks laid end-to-end on the pre-sorted seed.  Build coef in numpy,
    # seeded with coef_scalar, multiplying each Param in chain order.
    coef = np.full(n, float(coef_scalar), dtype=np.float64)
    non_dense_set = set(non_dense_dims)
    dense_set = set(dense_dims)

    # Distinct lead-key table (one row per block, in seed block order).
    # Reused by the lead-only case.  Sorted seed ⇒ unique(maintain_order)
    # yields blocks in laid-out order.
    lead_table = None
    if non_dense_dims:
        lead_table = seed.select(non_dense_dims).unique(maintain_order=True)
        if lead_table.height != n_lead:
            # Defensive: n_unique disagreed with the materialised distinct
            # table (should never happen) ⇒ fall back.
            return _build_block_coo_plan_joined(*_fallback_args)

    for atomic, direction in param_sources:
        # Under the classifier every Param dim ⊆ var.dims, so the shared
        # dims (in var-dim order) are exactly the Param's dims, ordered.
        shared = [d for d in var_dims if d in atomic.dims]
        if not shared:
            # No shared dim at all (cross-join broadcast) — let the joined
            # builder reproduce polars's cross-join semantics.
            return _build_block_coo_plan_joined(*_fallback_args)

        shared_set = set(shared)
        has_lead = bool(shared_set & non_dense_set)
        has_dense = bool(shared_set & dense_set)

        if has_lead and not has_dense:
            # --- Case lead-only: Param constant within each block.  Align
            # it to the per-block lead keys WITHOUT grid broadcast: left
            # join the tiny (n_lead-row) distinct-lead table on `shared`
            # (a subset of non_dense_dims), preserving block order, then
            # np.repeat each block value over its n_dense rows.
            atomic_lf = atomic.lazy
            lt_a, at_a = _align_enum_join_keys(lead_table.lazy(), atomic_lf, shared)
            aligned = lt_a.join(at_a, on=shared, how="left", maintain_order="left").collect()
            if aligned.height != n_lead or aligned["value"].null_count() > 0:
                # Param duplicated a lead key (expansion) or missed one
                # (null) ⇒ positional repeat would be wrong ⇒ fall back.
                return _build_block_coo_plan_joined(*_fallback_args)
            block_vals = aligned["value"].to_numpy().astype(np.float64, copy=False)
            repeated = np.repeat(block_vals, n_dense)
            if direction >= 0:
                coef = coef * repeated
            else:
                coef = coef / repeated

        elif has_dense and not has_lead and shared == dense_dims:
            # --- Case dense-only: one dense vector shared by every block.
            # Sort the (small) Param by dense_dims, read its value array
            # (must be length n_dense — else it's sparse on the dense axis
            # ⇒ fall back), tile across all blocks, multiply.  The sorted
            # dense vector is batch-invariant, so the bounded walk may supply
            # it pre-collected via ``dense_param_vectors`` (keyed by id) — use
            # the cached buffer when present, else collect+sort inline.
            dense_vals = (
                dense_param_vectors.get(id(atomic)) if dense_param_vectors is not None else None
            )
            if dense_vals is None:
                atomic_df = atomic.lazy.collect().sort(dense_dims)
                if atomic_df.height != n_dense:
                    return _build_block_coo_plan_joined(*_fallback_args)
                dense_vals = atomic_df["value"].to_numpy().astype(np.float64, copy=False)
            elif dense_vals.size != n_dense:
                return _build_block_coo_plan_joined(*_fallback_args)
            tiled = np.tile(dense_vals, n_lead)
            if direction >= 0:
                coef = coef * tiled
            else:
                coef = coef / tiled

        elif has_dense:
            # --- Case lead-subset + dense (e.g. efficiency(p,d,t)): the
            # Param has a contiguous length-n_dense slice per distinct
            # shared-lead value.  Robust positional impl: left-join the
            # Param onto the seed on the FULL `shared` with
            # maintain_order="left" (verified to preserve seed row order in
            # polars 1.40.1), read the aligned value array (must be length
            # n with no nulls — else sparse/ragged ⇒ fall back), multiply.
            # This case may materialise the Param at grid resolution (it is
            # genuinely dense data) but incurs NO re-sort.
            atomic_lf = atomic.lazy
            seed_a, at_a = _align_enum_join_keys(seed.lazy(), atomic_lf, shared)
            aligned = seed_a.join(at_a, on=shared, how="left", maintain_order="left").collect()
            if aligned.height != n or aligned["value"].null_count() > 0:
                return _build_block_coo_plan_joined(*_fallback_args)
            vals = aligned["value"].to_numpy().astype(np.float64, copy=False)
            if direction >= 0:
                coef = coef * vals
            else:
                coef = coef / vals

        else:
            # Shared dims overlap neither lead nor dense — impossible under
            # the classifier (every Param dim ⊆ var.dims = non_dense ∪
            # dense), but fall back defensively rather than mis-align.
            return _build_block_coo_plan_joined(*_fallback_args)

    # --- Step 4: emit.  Attach coef to the pre-sorted seed (positional —
    # coef[i] is the product for seed row i), inner-join row_index on `on`.
    _emit_block_coo_path("positional")
    result = seed.select(*var_dims, "col_id").with_columns(
        coef=pl.Series("coef", coef, dtype=pl.Float64)
    )
    ri_a, res_a = _align_enum_join_keys(row_index_lf, result.lazy(), on)
    joined_ri = ri_a.join(res_a, on=on, how="inner")
    return joined_ri.select("_rid", "col_id", "coef", *_block_coo_keep_cols(keep_dims)).collect()


def _sum_block_coo_classify(
    term: _Term,
    axis_cols: list[str],
    on: list[str],
    dense_axes,
) -> dict | None:
    """Classify whether a ``Sum``-wrapped ``Var × Param-chain`` LHS term
    (Phase C-3a) can be rebuilt-and-reduced via block-COO from its
    captured :class:`SumBlockMeta` recipe.  Returns a small spec ``dict``
    if it can, else ``None`` (caller uses the reduced ``term.lazy`` path
    verbatim).

    Unlike :func:`_block_coo_classify` (the non-Sum arm) this fires on the
    POST-Sum term: ``term.var_source`` is ``None`` (Sum cleared it) and
    ``term.dims`` is the surviving ``keep`` set, but ``term.sum_block_meta``
    snapshots the FULL pre-Sum recipe (originating Var, the complete —
    un-survivor-filtered — Param chain, the deferred filter / map-effect
    Where frames, the Sum's ``over``, and ``keep``).  Block-COO rebuilds
    the unreduced ``Var × P1 × P2 …`` product on the pre-sorted Var grid,
    bakes the map-effect Where to introduce the map dims, then reduces over
    ``reduce_dims`` to ``keep`` — producing the reduced LP coefficients
    without polars' join + group_by.

    Fires iff ALL hold (else ``None`` → fall back, always safe):

    * ``term.sum_block_meta is not None`` and ``dense_axes`` non-empty.
    * The recipe's ``var_source`` dims END WITH ``dense_axes`` in declared
      order (the suffix contract — block-COO slices the dense suffix).
    * Every Param in the recipe's FULL ``param_sources`` is a real
      ``(Param, dir)`` pair with ``dir ∈ {±1}`` and dims ⊆
      ``var.dims ∪ map_extras ∪ axis_cols`` (no foreign dim the rebuild
      can't account for).
    * The post-map open dim set = ``var.dims ∪ map_extras``; ``keep`` (=
      ``term.dims``), ``reduce_dims`` (= ``over``) are both ⊆ that set, and
      ``on ⊆ keep``.

    Conservative by construction: any shape the rebuild+reduce loop cannot
    reproduce bit-equivalently must return ``None`` here.
    """
    if not dense_axes:
        return None
    meta = term.sum_block_meta
    if meta is None:
        return None
    dense_axes = list(dense_axes)

    var = meta.var_source
    if var is None:
        return None
    psrc = list(meta.param_sources)
    if len(psrc) == 0:
        return None

    var_dims = list(var.dims)
    var_dim_set = set(var_dims)

    # Suffix contract: the Var's dims must END WITH the declared dense_axes
    # in declared order, so block-COO can slice the dense suffix with no
    # re-sort under the client's sort promise.
    if len(dense_axes) > len(var_dims):
        return None
    if tuple(var_dims[-len(dense_axes) :]) != tuple(dense_axes):
        return None

    # Map-effect extras: dims the deferred map-Where introduces (e.g. ``n``
    # from ``flow_to_n``).  Collected in frame-column order per frame so the
    # rebuilt open-dim set is deterministic.  A pure map-effect frame whose
    # ``shared`` columns are not all ⊆ var.dims would inner-join on a column
    # the seed (Var dims only) cannot carry — fall back.
    map_extras: list[str] = []
    if meta.where_map_frames is not None:
        for mf, extras in meta.where_map_frames:
            mf_cols = mf.collect_schema().names()
            shared = [c for c in mf_cols if c in var_dim_set or c in map_extras]
            # The map frame's non-extra columns are its join keys; they must
            # already be present (var dims or a prior frame's extras).  A
            # frame keyed on a column we cannot supply ⇒ fall back.
            non_extra = [c for c in mf_cols if c not in extras]
            if any(c not in var_dim_set and c not in map_extras for c in non_extra):
                return None
            if not shared and extras:
                # Cross-join map frame (no shared key) — the reduce alignment
                # below relies on a deterministic join; decline rather than
                # risk a cross-product we cannot reduce cleanly.
                return None
            for c in mf_cols:
                if c in extras and c not in map_extras:
                    map_extras.append(c)

    # Post-map open dim universe.
    open_dims = var_dims + [d for d in map_extras if d not in var_dim_set]
    open_set = set(open_dims)
    axis_set = set(axis_cols)
    allowed = open_set | axis_set

    for entry in psrc:
        if not (isinstance(entry, tuple) and len(entry) == 2):
            return None
        atomic, direction = entry
        if not isinstance(atomic, Param):
            return None
        if direction not in (1, -1):
            return None
        if not set(atomic.dims).issubset(allowed):
            return None

    keep = list(meta.keep)
    reduce_dims = list(meta.reduce_dims)
    if not set(keep).issubset(open_set):
        return None
    if not set(reduce_dims).issubset(open_set):
        return None
    if not set(on).issubset(set(keep)):
        return None
    # ``keep`` must partition the open dims with ``reduce_dims`` — every open
    # dim is either kept or reduced (no dangling dim that would leave the
    # group ambiguous).  ``keep`` and ``reduce_dims`` must be disjoint.
    if set(keep) & set(reduce_dims):
        return None
    if open_set != (set(keep) | set(reduce_dims)):
        return None

    dense_dims = list(dense_axes)
    non_dense_dims = [d for d in var_dims if d not in set(dense_axes)]

    try:
        dense_card = int(var.frame.height)
    except Exception:
        dense_card = None
    if dense_card is not None and dense_card < _block_coo_min_dense():
        return None

    return {
        "var_dims": var_dims,
        "dense_dims": dense_dims,
        "non_dense_dims": non_dense_dims,
        "map_extras": map_extras,
        "open_dims": open_dims,
        "keep": keep,
        "reduce_dims": reduce_dims,
        "dense_card": dense_card if dense_card is not None else 0,
        "on": list(on),
    }


class _SumBlockCooFallback(Exception):
    """Raised inside :func:`_build_sum_block_coo_plan` when a shape cannot
    be reconstructed + reduced bit-equivalently, signalling the caller to
    use the reduced ``term.lazy`` emission verbatim."""


def _build_sum_block_coo_relabel(
    row_index_lf: pl.LazyFrame,
    axis_cols: list[str],
    meta: SumBlockMeta,
    on: list[str],
    dense_spec: dict,
    keep_dims: tuple[str, ...] | None = None,
    *,
    dense_param_vectors: dict[int, np.ndarray] | None = None,
) -> pl.DataFrame:
    """Relabel fast-path for a ``Sum``-wrapped ``Var × Param-chain`` term
    when ``reduce_dims ⊆ var.dims`` (Phase C-3b).

    The exploitable insight: when every reduced dim is already a Var dim,
    ``col_id`` is a function of the Var instance, so two unreduced rows
    sharing a ``col_id`` share all ``var.dims`` values, hence all
    ``reduce_dims`` values — they are the SAME row.  Every ``(*keep,
    col_id)`` reduction group is therefore SINGLE-ELEMENT: the Sum performs
    NO coefficient summation, only a RELABEL of each Var-grid row to its
    constraint row(s) via the map-join (1:1, or fan-out to several distinct
    rows — each still its own single-element group).

    So we skip the materialize-then-reduce builder's full-product
    sort + ``np.add.reduceat`` entirely and instead mirror
    :func:`_build_block_coo_plan`'s *positional* per-block slice-multiply on
    the pre-sorted Var seed (peak bounded by the Var grid + per-factor numpy
    buffers, NOT the full unreduced product after map / Param fan-out),
    apply the map-effect Where to introduce the kept map dims, then attach
    ``_rid`` by inner-joining ``row_index_lf`` on ``on`` — a direct emit,
    no group-by.

    Bit-identity
    ------------
    The coef is seeded with ``meta.coef_scalar`` and multiplied STRICTLY in
    ``meta.param_sources`` order (the FULL list, including summed-out
    factors such as ``p_unitsize``), with the SAME positional alignment
    cases (and the SAME IEEE-double op sequence) as
    :func:`_build_block_coo_plan`.  Same row set (the Var seed), same
    per-row factor values, same multiply order ⇒ bit-identical
    coefficients.  Because every reduce group is single-element there is no
    summation step to perturb — the result is bit-identical to the reduced
    ``term.lazy`` path, not merely bit-equivalent.

    Determinism
    -----------
    The seed is pre-sorted (caller-verified) and the positional multiply is
    a fixed left-to-right ufunc chain.  The map-join and the row_index
    inner-join are polars (Rust) joins whose output is a deterministic
    function of their inputs — independent of ``PYTHONHASHSEED`` (no Python
    set/dict iteration drives emission).  No hash-order-dependent step.

    Fallback contract
    -----------------
    Any shape this cannot reproduce bit-identically via the positional
    cases raises :class:`_SumBlockCooFallback`; the caller emits the reduced
    ``term.lazy`` verbatim.  Same contract as the combining path.
    """
    var_source = meta.var_source
    var_dims = list(var_source.dims)
    var_dim_set = set(var_dims)
    keep = list(dense_spec["keep"])
    dense_dims = list(dense_spec["dense_dims"])
    non_dense_dims = list(dense_spec["non_dense_dims"])
    coef_scalar = float(meta.coef_scalar)

    # --- Step 1: bake pure-filter Where frames + pre-prune the Var seed by
    # the row_index key set (semi-join), exactly as _build_block_coo_plan.
    # Filtering preserves the seed's sort order (just narrows it).
    var_on = [d for d in var_dims if d in axis_cols]
    var_lf = var_source.frame.lazy()
    if meta.where_frames is not None:
        var_lf = _apply_where_frames(var_lf, var_dims, meta.where_frames)
    if var_on:
        ri_keys = row_index_lf.select(var_on).unique()
        var_lf_a, ri_keys_a = _align_enum_join_keys(var_lf, ri_keys, var_on)
        var_lf = var_lf_a.join(ri_keys_a, on=var_on, how="semi")
    seed = var_lf.select(*var_dims, "col_id").collect()

    n = seed.height
    if n == 0:
        return _empty_block_coo_frame(seed, keep_dims)

    # --- Step 2: completeness guard.  Positional slicing is valid only when
    # every leading-dim block is the full, identically-ordered dense set; a
    # deferred filter can leave the grid sparse/ragged ⇒ fall back.  (Mirror
    # of _build_block_coo_plan's guard; here a failure raises the Sum
    # fallback sentinel rather than recursing into a joined backstop.)
    if meta.where_frames is not None:
        raise _SumBlockCooFallback("relabel: deferred filter present")
    n_dense = seed.select(dense_dims).n_unique()
    n_lead = seed.select(non_dense_dims).n_unique() if non_dense_dims else 1
    if n != n_lead * n_dense:
        raise _SumBlockCooFallback("relabel: sparse/ragged Var seed")

    # --- Step 3: positional coef chain (mirror of _build_block_coo_plan).
    coef = np.full(n, coef_scalar, dtype=np.float64)
    non_dense_set = set(non_dense_dims)
    dense_set = set(dense_dims)

    lead_table = None
    if non_dense_dims:
        lead_table = seed.select(non_dense_dims).unique(maintain_order=True)
        if lead_table.height != n_lead:
            raise _SumBlockCooFallback("relabel: lead-table cardinality")

    for atomic, direction in meta.param_sources:
        # Under the classifier a Param's dims ⊆ var.dims ∪ map_extras ∪
        # axis_cols.  When map frames are present, no Param references a
        # map_extra (such a multiply would have baked the map eagerly,
        # clearing where_map_frames) — so every relabel Param's dims that
        # matter are ⊆ var.dims.  A Param dim purely in axis_cols (not in
        # var.dims) yields an empty shared set ⇒ fall back.
        shared = [d for d in var_dims if d in atomic.dims]
        if not shared:
            raise _SumBlockCooFallback("relabel: no-shared Param")
        if not set(atomic.dims).issubset(var_dim_set):
            raise _SumBlockCooFallback("relabel: Param dim outside var.dims")

        shared_set = set(shared)
        has_lead = bool(shared_set & non_dense_set)
        has_dense = bool(shared_set & dense_set)

        if has_lead and not has_dense:
            # Lead-only: Param constant within each block.
            atomic_lf = atomic.lazy
            lt_a, at_a = _align_enum_join_keys(lead_table.lazy(), atomic_lf, shared)
            aligned = lt_a.join(at_a, on=shared, how="left", maintain_order="left").collect()
            if aligned.height != n_lead or aligned["value"].null_count() > 0:
                raise _SumBlockCooFallback("relabel: lead-only mis-align")
            block_vals = aligned["value"].to_numpy().astype(np.float64, copy=False)
            repeated = np.repeat(block_vals, n_dense)
            if direction >= 0:
                coef = coef * repeated
            else:
                coef = coef / repeated

        elif has_dense and not has_lead and shared == dense_dims:
            # Dense-only: one dense vector shared by every block.  The sorted
            # dense vector is batch-invariant, so the bounded walk may supply
            # it pre-collected via ``dense_param_vectors`` (keyed by id) — use
            # the cached buffer when present, else collect+sort inline.
            dense_vals = (
                dense_param_vectors.get(id(atomic)) if dense_param_vectors is not None else None
            )
            if dense_vals is None:
                atomic_df = atomic.lazy.collect().sort(dense_dims)
                if atomic_df.height != n_dense:
                    raise _SumBlockCooFallback("relabel: dense-only sparse")
                dense_vals = atomic_df["value"].to_numpy().astype(np.float64, copy=False)
            elif dense_vals.size != n_dense:
                raise _SumBlockCooFallback("relabel: dense-only sparse")
            tiled = np.tile(dense_vals, n_lead)
            if direction >= 0:
                coef = coef * tiled
            else:
                coef = coef / tiled

        elif has_dense:
            # Lead-subset + dense: positional left-join on the FULL shared
            # with maintain_order="left" (order-preserving in polars 1.40.1).
            atomic_lf = atomic.lazy
            seed_a, at_a = _align_enum_join_keys(seed.lazy(), atomic_lf, shared)
            aligned = seed_a.join(at_a, on=shared, how="left", maintain_order="left").collect()
            if aligned.height != n or aligned["value"].null_count() > 0:
                raise _SumBlockCooFallback("relabel: lead+dense mis-align")
            vals = aligned["value"].to_numpy().astype(np.float64, copy=False)
            if direction >= 0:
                coef = coef * vals
            else:
                coef = coef / vals

        else:
            raise _SumBlockCooFallback("relabel: Param overlaps neither axis")

    # --- Step 4: attach coef to the pre-sorted seed (positional — coef[i]
    # is the product for seed row i).  Carry only the var dims + col_id.
    result = seed.select(*var_dims, "col_id").with_columns(
        coef=pl.Series("coef", coef, dtype=pl.Float64)
    )

    # --- Step 5: apply the map-effect Where frames (e.g. flow_to_n) to
    # introduce the kept map dims (n).  This is an inner-join on the map
    # frame's shared keys (var dims); it may relabel 1:1 or fan a row out to
    # several DISTINCT (keep, col_id) rows — each still a single-element
    # group, so NO summation either way.  The map frame is small; the result
    # stays at relabel-output resolution, never the full param×map product.
    res_lf, res_dims = _apply_where_map_frames(
        result.lazy(), tuple(var_dims) + ("col_id", "coef"), meta.where_map_frames
    )

    # --- Step 6: project to (*keep, col_id, coef) and attach _rid by
    # inner-joining row_index on ``on`` (⊆ keep).  No group-by, no reduceat:
    # single-element groups mean the relabel is the final coefficient.
    reduced = res_lf.select(*keep, "col_id", "coef")
    ri_a, res_a = _align_enum_join_keys(row_index_lf, reduced, on)
    joined_ri = ri_a.join(res_a, on=on, how="inner")
    return joined_ri.select("_rid", "col_id", "coef", *_block_coo_keep_cols(keep_dims)).collect()


def _build_sum_block_coo_plan(
    row_index_lf: pl.LazyFrame,
    axis_cols: list[str],
    meta: SumBlockMeta,
    on: list[str],
    dense_spec: dict,
    keep_dims: tuple[str, ...] | None = None,
    *,
    dense_param_vectors: dict[int, np.ndarray] | None = None,
) -> pl.DataFrame:
    """Evaluate a ``Sum``-wrapped ``Var × Param-chain`` LHS term (Phase
    C-3a) by rebuilding the unreduced product from the
    :class:`SumBlockMeta` recipe and reducing it in-block, WITHOUT polars'
    join + group_by.

    Returns an eager :class:`polars.DataFrame` with columns
    ``(_rid, col_id, coef[, *keep_dims])`` — the same emission contract as
    :func:`_build_block_coo_plan`, so the canonical builder consumes it
    unchanged.

    Branch (relabel vs combining)
    -----------------------------
    When ``reduce_dims ⊆ var.dims`` (e.g. ``nodeBalance_eq``,
    ``over=("p","source","sink") ⊆ v_flow.dims``) every ``(*keep, col_id)``
    reduce group is SINGLE-ELEMENT — the Sum is a pure RELABEL, no
    coefficient summation — so we delegate to
    :func:`_build_sum_block_coo_relabel`, which mirrors
    :func:`_build_block_coo_plan`'s bounded *positional* per-block
    slice-multiply and emits directly (peak bounded by the Var grid + numpy
    buffers, NOT the full unreduced product).  Otherwise (genuine coef
    combining: a reduced dim is NOT a Var dim, e.g. a map-introduced ``h``
    fanned out and summed) we take the materialize-then-reduce path below,
    which is correct but peaks at the full unreduced product.

    Materialize-then-reduce algorithm (combining path)
    --------------------------------------------------
    1. **Seed** = ``meta.var_source.frame`` (PRE-SORTED by ``(non_dense…,
       dense…)`` per the dense_axes contract — verified by the caller via
       :func:`_verify_dense_sorted`).  Carry ``col_id`` + the Var dims.
    2. **Bake ``meta.where_frames``** (deferred pure-filter, semi-join,
       order-preserving) onto the seed.
    3. **Apply ``meta.where_map_frames``** via :func:`_apply_where_map_frames`
       to introduce the map dims (e.g. ``n``).  This is an inner-join that
       can reorder, so we re-establish a deterministic order in step 5.
    4. **Multiply the FULL ``meta.param_sources``** in chain order (seed
       ``coef_scalar``, ``*value`` for ``dir >= 0`` / ``/value`` for
       ``dir < 0``).  Each Param is left-joined onto the accumulator on its
       shared dims with ``maintain_order="left"`` (verified order-preserving
       in polars 1.40.1) — the SAME IEEE-double op sequence as the polars
       prune-down chain.  A sparse Param (a join that introduces a null
       ``value``) means the recipe's unreduced product is not dense over
       this accumulator ⇒ raise the fallback sentinel.
    5. **Reduce** over ``reduce_dims`` to ``keep``: sort the unreduced frame
       by ``(*keep, col_id)`` (a FIXED canonical order) and sum ``coef``
       per ``(*keep, col_id)`` group via ``np.add.reduceat`` over the group
       boundaries.  The fixed sort order guarantees run-to-run determinism.
       Where each group is a single row (e.g. nodeBalance — each flow is a
       distinct ``col_id`` mapping to one node) the sum is a 1-element sum
       ⇒ BIT-IDENTICAL to polars' group_by.  Multi-row groups (true coef
       combining) are bit-EQUIVALENT (a different summation order than
       polars' hash-group).
    6. **Attach ``_rid``** by inner-joining ``row_index_lf`` on ``on``
       (Enum-aligned).  Return selecting ``_rid, col_id, coef[, *keep]``.

    Fallback contract
    -----------------
    On ANY shape this cannot reconstruct + reduce bit-equivalently it
    raises :class:`_SumBlockCooFallback`; the caller catches it and uses
    the reduced ``term.lazy`` emission verbatim (byte-identical to today).
    """
    var_source = meta.var_source
    var_dims = list(var_source.dims)

    # --- Relabel fast-path: reduce_dims ⊆ var.dims ⇒ single-element groups
    # ⇒ no summation ⇒ skip the full-product sort + reduceat.  Bit-identical.
    if set(dense_spec["reduce_dims"]).issubset(set(var_dims)):
        if os.environ.get("POLAR_HIGH_BLOCK_COO_PROFILE") == "1":
            sys.stderr.write("[block_coo profile]\tkind=sum\tpath=relabel\n")
            sys.stderr.flush()
        return _build_sum_block_coo_relabel(
            row_index_lf,
            axis_cols,
            meta,
            on,
            dense_spec,
            keep_dims,
            dense_param_vectors=dense_param_vectors,
        )
    if os.environ.get("POLAR_HIGH_BLOCK_COO_PROFILE") == "1":
        sys.stderr.write("[block_coo profile]\tkind=sum\tpath=combining\n")
        sys.stderr.flush()

    keep = list(dense_spec["keep"])
    # ``reduce_dims`` (= dense_spec["reduce_dims"]) is summed out implicitly:
    # the group_by on (*keep, col_id) below collapses every open dim not in
    # ``keep``, which by the classifier's partition guarantee is exactly the
    # reduce-dims set.
    coef_scalar = float(meta.coef_scalar)

    # --- Step 1: seed = pre-sorted Var grid (col_id + var dims).
    seed_df = var_source.frame.select(*var_dims, "col_id")
    acc_lf = seed_df.lazy()
    acc_dims = list(var_dims)

    # --- Step 2: bake deferred pure-filter Where frames (semi-join).
    acc_lf = _apply_where_frames(acc_lf, acc_dims, meta.where_frames)

    # --- Step 3: bake map-effect Where frames (inner-join, dim-extending).
    acc_lf, acc_dims_t = _apply_where_map_frames(acc_lf, acc_dims, meta.where_map_frames)
    acc_dims = list(acc_dims_t)

    # --- Step 4: multiply the FULL Param chain in order.  Use a left-join
    # per atomic with maintain_order="left" (order-preserving in polars
    # 1.40.1) so we can read each value column into a numpy buffer aligned
    # to the accumulator and multiply in the SAME order as the polars chain.
    # We carry ``value`` columns one at a time (rename uniquely) so the
    # accumulator stays narrow.
    acc = acc_lf.collect()
    n = acc.height
    if n == 0:
        # Empty after filter — emit the empty (_rid, col_id, coef[, *keep])
        # frame.  keep_dims (warm site) dtypes come from the seed schema.
        return _empty_block_coo_frame(seed_df, keep_dims)

    coef = np.full(n, coef_scalar, dtype=np.float64)
    for atomic, direction in meta.param_sources:
        shared = [d for d in acc_dims if d in atomic.dims]
        atomic_lf = atomic.lazy.rename({"value": "__sb_val"})
        if shared:
            acc_a, at_a = _align_enum_join_keys(acc.lazy(), atomic_lf, shared)
            aligned = acc_a.join(
                at_a.select(*shared, "__sb_val"),
                on=shared,
                how="left",
                maintain_order="left",
            ).collect()
        else:
            # Param with no shared dim ⇒ a scalar broadcast (single row) or
            # a cross-product we won't reduce cleanly.  Only a single-row
            # Param is safe (broadcast); anything else falls back.
            atomic_df = atomic.lazy.collect()
            if atomic_df.height != 1:
                raise _SumBlockCooFallback("no-shared multi-row Param")
            aligned = acc.with_columns(pl.lit(float(atomic_df["value"][0])).alias("__sb_val"))
        if aligned.height != n:
            # A left join that changed the row count means a Param key
            # expansion (duplicate keys) — the recipe's product is no longer
            # 1:1 with the seed rows ⇒ fall back.
            raise _SumBlockCooFallback("Param key expansion")
        if "__sb_val" not in aligned.columns:
            raise _SumBlockCooFallback("Param value column missing")
        if aligned["__sb_val"].null_count() > 0:
            # Sparse Param: the recipe's unreduced product is not dense over
            # the accumulator (a polars inner-join chain would DROP these
            # rows, not null them).  Reproducing the drop here positionally
            # is fragile; fall back to the guaranteed-correct reduced path.
            raise _SumBlockCooFallback("sparse Param (null after left join)")
        vals = aligned["__sb_val"].to_numpy().astype(np.float64, copy=False)
        if direction >= 0:
            coef = coef * vals
        else:
            coef = coef / vals
        # ``aligned`` preserves the accumulator's row order (maintain_order
        # ="left"); keep ``acc`` as the order-of-record for the next factor.
        acc = aligned.drop("__sb_val")

    # --- Step 5: deterministic reduce over reduce_dims to keep.  Attach the
    # unreduced coef to the accumulator, sort by (*keep, col_id) (a FIXED
    # canonical order), and sum per (*keep, col_id) group with
    # np.add.reduceat over the group boundaries.
    unreduced = acc.select(*keep, "col_id").with_columns(
        coef=pl.Series("coef", coef, dtype=pl.Float64)
    )
    group_keys = keep + ["col_id"]
    unreduced = unreduced.sort(group_keys)
    m = unreduced.height
    if m == 0:
        return _empty_block_coo_frame(seed_df, keep_dims)
    coef_sorted = unreduced["coef"].to_numpy().astype(np.float64, copy=False)
    # Group boundaries: first row of each distinct (*keep, col_id) tuple.
    # Build a boolean "is new group" mask via a struct equality shift.  The
    # very first row's shift(1) is null ⇒ the inequality is null; fill it
    # True (a new group always starts at row 0) and cast to a dense bool
    # array so np.flatnonzero / np.add.reduceat operate on contiguous data.
    is_new = (
        unreduced.select(
            (pl.struct(group_keys) != pl.struct(group_keys).shift(1)).fill_null(True).alias("__new")
        )["__new"]
        .to_numpy()
        .astype(bool, copy=False)
    )
    starts = np.flatnonzero(is_new)
    reduced_coef = np.add.reduceat(coef_sorted, starts)
    reduced = (
        unreduced[starts]
        .select(*keep, "col_id")
        .with_columns(coef=pl.Series("coef", reduced_coef, dtype=pl.Float64))
    )

    # --- Step 6: attach _rid via row_index inner-join on ``on``.
    ri_a, res_a = _align_enum_join_keys(row_index_lf, reduced.lazy(), on)
    joined_ri = ri_a.join(res_a, on=on, how="inner")
    return joined_ri.select("_rid", "col_id", "coef", *_block_coo_keep_cols(keep_dims)).collect()


def _where_pushdown_disabled() -> bool:
    """Return True when ``POLAR_HIGH_DISABLE_WHERE_PUSHDOWN=1``.

    Mirror of :func:`_prune_down_disabled`.  When set, :func:`Where`
    falls back to its original behaviour — eager inner-join into
    ``term.lazy`` and ``var_source`` / ``coef_scalar`` / ``where_frames``
    cleared — so users have a safety hatch if the deferred-filter path
    drifts.  Sum / Lag still bake any pre-existing ``where_frames`` for
    correctness; the env-var only controls the pushdown record itself.
    """
    return os.environ.get("POLAR_HIGH_DISABLE_WHERE_PUSHDOWN") == "1"


def _merge_param_sources(
    a: list[tuple[Param, int]] | None,
    b: list[tuple[Param, int]] | None,
    *,
    flip_other: bool,
) -> list[tuple[Param, int]] | None:
    """Combine two source lists for ``Param * Param`` (flip_other=False)
    or ``Param / Param`` (flip_other=True).  Returns None when both
    inputs are None (no tracking needed).
    """
    if a is None and b is None:
        return None
    out: list[tuple[Param, int]] = []
    if a is not None:
        out.extend(a)
    if b is not None:
        if flip_other:
            out.extend((p, -s) for p, s in b)
        else:
            out.extend(b)
    return out


class Var:
    """A variable family.  ``frame`` carries columns ``*dims, col_id``.

    ``Var.frame`` stays an eager polars DataFrame — it's small (one row
    per LP column), produced once in :meth:`Problem.add_var`, and
    consumed by both flextool integration (``v.frame["col_id"].unique()``)
    and ``Problem.solve`` (col_id → bound/name lookups).  Algebra ops on
    Var lazify on the fly so the resulting ``_Term`` is lazy."""

    __slots__ = ("name", "dims", "frame", "lower", "upper", "integer")

    def __init__(
        self,
        name: str,
        dims: tuple[str, ...],
        frame: pl.DataFrame,
        lower: float = 0.0,
        upper: float = float("inf"),
        integer: bool = False,
    ):
        self.name = name
        self.dims = tuple(dims)
        # Var.frame is intentionally eager — see class docstring.
        if isinstance(frame, pl.LazyFrame):
            frame = frame.collect()
        self.frame = frame
        self.lower = lower
        self.upper = upper
        self.integer = integer

    def to_expr(self) -> Expr:
        f = self.frame.lazy().with_columns(coef=pl.lit(1.0)).select(*self.dims, "col_id", "coef")
        return Expr([_Term(f, self.dims, var_source=self)])

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            f = (
                self.frame.lazy()
                .with_columns(coef=pl.lit(float(other)))
                .select(*self.dims, "col_id", "coef")
            )
            return Expr([_Term(f, self.dims, var_source=self, coef_scalar=float(other))])
        if isinstance(other, Param):
            shared = [d for d in self.dims if d in other.dims]
            new_dims = tuple(dict.fromkeys(self.dims + other.dims))
            lf = self.frame.lazy()
            if shared:
                lf, right_lf = _align_enum_join_keys(lf, other.lazy, shared)
                j = lf.join(right_lf, on=shared, how="inner")
            else:
                j = lf.join(other.lazy, how="cross")
            j = j.rename({"value": "coef"}).select(*new_dims, "col_id", "coef")
            psrc = other._sources_for_propagation()
            return Expr(
                [
                    _Term(
                        j,
                        new_dims,
                        param_sources=psrc,
                        var_source=self,
                        coef_scalar=other._value_scalar,
                    )
                ]
            )
        return NotImplemented

    __rmul__ = __mul__

    def __neg__(self):
        return self.to_expr() * -1.0

    def __add__(self, o):
        return self.to_expr() + o

    def __sub__(self, o):
        return self.to_expr() - o

    def __radd__(self, o):
        return self.to_expr() + o

    def __rsub__(self, o):
        return _to_expr(o) - self.to_expr()


@dataclass(frozen=True)
class SumBlockMeta:
    """Inert reconstruction recipe captured at ``Sum``-time (Phase C-2).

    When a block-eligible term (``var_source`` present, non-empty
    ``param_sources`` list, non-empty ``over``) is reduced by
    :func:`Sum`, the aggregation clears ``var_source`` and survivor-
    filters ``param_sources`` on the returned term, discarding the
    information needed to rebuild the pre-Sum ``row_index → Var → P1 →
    P2 …`` chain and reduce it in-block.  :class:`SumBlockMeta` snapshots
    that pre-Sum state so a future block-COO classifier (Phase C-3) can
    evaluate the Sum-wrapped chain by rebuilding from leaves and reducing
    within the block.

    All fields are captured from the PRE-Sum term, *before* any clearing
    or survivor-filtering:

    * ``var_source`` — the originating :class:`Var`.
    * ``param_sources`` — the FULL pre-Sum ``list[(Param, direction)]``,
      NOT survivor-filtered: block-COO needs every factor, including
      Params whose dims are summed out (e.g. ``p_unitsize`` over ``p``).
    * ``coef_scalar`` — the cumulative constant scalar folded into coef.
    * ``where_frames`` — the pre-Sum deferred pure-filter frames.
    * ``where_map_frames`` — the pre-Sum deferred map-effect frames.
    * ``reduce_dims`` — the dims summed out (the Sum's ``over``).
    * ``keep`` — the post-Sum open dims (the returned term's dims).

    This class is currently INERT: nothing reads it (confirmed by grep).
    It is set ONLY at the point of reduction in :func:`Sum`, never
    propagated through later :class:`Expr` ops or a nested / re-reduced
    :func:`Sum` (those set it to ``None``), so a stale recipe can never
    attach to an already-reduced term.
    """

    var_source: Var
    param_sources: tuple[tuple[Param, int], ...]
    coef_scalar: float
    where_frames: tuple[pl.LazyFrame, ...] | None
    where_map_frames: tuple[tuple[pl.LazyFrame, frozenset[str]], ...] | None
    reduce_dims: tuple[str, ...]
    keep: tuple[str, ...]


def _forward_sum_block_meta_param_mul(
    meta: SumBlockMeta,
    q: Param,
    *,
    flip: bool,
) -> SumBlockMeta | None:
    """D1: forward a ``SumBlockMeta`` recipe through a post-Sum
    ``Expr * Param`` (``flip=False``) or ``Expr / Param`` (``flip=True``).

    The captured recipe rebuilds the pre-Sum ``Var × P1 × P2 …`` chain on
    the Var grid and reduces it in-block, so a Param multiplied AFTER the
    Sum can only ride along when its rows align with the rows the recipe
    already produces — i.e. its dims sit inside the surviving ``keep`` set
    and do NOT touch a summed-out ``reduce_dims`` (which would re-introduce
    a collapsed axis the recipe can no longer broadcast over).

    SAFE — ``set(q.dims) ⊆ set(meta.keep)`` AND
    ``set(q.dims) ∩ set(meta.reduce_dims) == ∅``: append ``q``'s atomic
    ``(Param, direction)`` constituents to ``param_sources`` (flipping
    direction for division, mirroring :func:`_merge_param_sources`) and
    fold ``q``'s folded constant ``_value_scalar`` into ``coef_scalar``
    exactly as the live Param arm folds it into ``_Term.coef_scalar``.
    Returns the forwarded recipe.

    DECLINE (option b) — ``q`` introduces a new dim (``⊄ keep``) or
    re-introduces a summed dim (``∩ reduce_dims``): the recipe cannot be
    reconstructed bit-equivalently, so return ``None`` (the term takes the
    bounded fallback) AND emit a loud, always-on marker naming the
    offending shape for a future "option a" dim-extending builder.
    """
    q_dims = set(q.dims)
    keep_set = set(meta.keep)
    reduce_set = set(meta.reduce_dims)
    if q_dims.issubset(keep_set) and not (q_dims & reduce_set):
        psrc_other = q._sources_for_propagation()
        merged = _merge_param_sources(list(meta.param_sources), psrc_other, flip_other=flip)
        new_scalar = (
            meta.coef_scalar / q._value_scalar if flip else meta.coef_scalar * q._value_scalar
        )
        return replace(
            meta,
            param_sources=tuple(merged) if merged is not None else (),
            coef_scalar=new_scalar,
        )
    # DECLINE — emit the always-on marker, then drop the recipe.
    import warnings

    op = "Expr / Param" if flip else "Expr * Param"
    new_dims = sorted(q_dims - keep_set)
    reintroduced = sorted(q_dims & reduce_set)
    qname = q.name if q.name is not None else "<anonymous>"
    warnings.warn(
        "[sum-block-meta DECLINE] post-Sum "
        f"{op} drops the block-COO reconstruction recipe: Param "
        f"{qname!r} dims={tuple(q.dims)!r} "
        f"introduces new dim(s)={new_dims!r} (not in keep) "
        f"and/or re-introduces summed dim(s)={reintroduced!r} "
        f"(in reduce_dims); term keep={meta.keep!r} "
        f"reduce_dims={meta.reduce_dims!r}. The term reverts to the "
        "bounded fallback. To carry it, extend the block builder with a "
        'dim-extending branch ("option a") for this Param shape.',
        UserWarning,
        stacklevel=2,
    )
    return None


class _Term:
    """One additive term inside an Expr.

    ``dims`` are the *open* dims of the term — the dims that have not
    been collapsed by ``Sum``.  When this term is bound to a
    constraint's ``over=`` index, ``dims`` must all appear in that
    index (or be empty).

    ``lazy`` is a :class:`polars.LazyFrame` carrying columns
    ``*dims, col_id, coef`` (or just ``col_id, coef`` for collapsed
    terms).  Stored lazy so a chain of algebra ops (mul / sub / Sum /
    Where / Lag) builds up a single fused query that polars
    materializes once at constraint emission.  The ``.frame`` property
    collects on demand.

    ``param_sources`` — opt-in metadata for :class:`WarmProblem`'s
    Param-tracked auto-update.  Lists ``(Param, direction)`` tuples for
    every named :class:`Param` that contributed to this term's ``coef``
    column (direction +1 numerator, -1 denominator).  ``None`` when no
    tracking is requested — terms flowing through unnamed Params pay no
    overhead.  The Param object is held by reference so :class:`WarmProblem`
    can read its current value frame at build time.

    ``var_source`` — opt-in metadata used by the LHS Param-chain prune-
    down path (mirror of the RHS prune-down in
    ``_build_canonical_matrix``).  Holds the originating :class:`Var`
    (``Var.frame`` is the col_id source) and the eager Param chain
    multiplicand list, so the canonical / streaming / warm builders can
    rebuild a ``row_index → Var → P1 → P2 …`` join chain that prunes one
    atomic at a time instead of materialising the fully-merged Var×Param×
    Param… intermediate inside ``term.lazy``.  Set by :meth:`Var.__mul__`
    / :meth:`Expr.__mul__` and preserved through ``Expr.__sub__`` /
    ``__neg__`` / scalar multiplies / pure-filter :func:`Where`.
    Cleared (set to ``None``) by operations that change the term's row
    identity — :func:`Sum`, :func:`Lag`, and the "map effect" branch of
    :func:`Where` (when the filter frame contributes extra dims) —
    because the rebuilt chain would no longer mirror the
    post-aggregation/extension row set.

    ``where_frames`` — opt-in tuple of deferred filter frames recorded by
    pure-filter :func:`Where` calls (shared cols only, no map effect).
    Each entry is a :class:`polars.LazyFrame`; the filter is *not* baked
    into ``lazy`` so downstream prune-down can stay leaf-level.  The
    canonical / streaming / warm LHS builders bake these against the
    rebuilt ``row_index → Var → P1 → P2 …`` chain via
    :func:`_apply_where_frames`.  Stored as a tuple so accidental shared
    mutation between sibling terms is impossible.  Cleared by
    :func:`Sum` / :func:`Lag` / map-effect :func:`Where` (after baking)
    and by fallback paths that materialise ``lazy`` directly.

    ``where_map_frames`` — opt-in tuple of deferred *map-effect*
    :func:`Where` frames.  Each entry is ``(frame_lf, extras_frozenset)``
    where ``extras_frozenset`` is the set of frame columns NOT in the
    term's dims at the time of the Where call (the new open dims the
    frame introduces, e.g. ``flow_to_n`` mapping ``(p, source, sink) →
    n``).  Like ``where_frames``, the inner-join is *not* baked into
    ``lazy`` — instead the term's dims are extended with the extras at
    the Where call while ``var_source`` / ``param_sources`` /
    ``coef_scalar`` / ``where_frames`` are PRESERVED, so the LHS
    prune-down (and, later, block-COO) can still reach the leaves.  The
    dim-extending inner-join is baked via :func:`_apply_where_map_frames`
    at leaf-rebuild time (``_build_lhs_pruned_plan``) or at Sum / Lag /
    consumer-fallback.  Stored as a tuple of immutable
    ``(LazyFrame, frozenset)`` pairs.

    ``sum_block_meta`` — opt-in :class:`SumBlockMeta` snapshot of the
    pre-Sum reconstruction recipe, set ONLY on the term returned by a
    block-eligible :func:`Sum` reduction (Phase C-2).  Captures the
    pre-Sum ``var_source`` / FULL (un-filtered) ``param_sources`` /
    ``coef_scalar`` / ``where_frames`` / ``where_map_frames`` plus the
    Sum's ``over`` (``reduce_dims``) and the surviving open dims
    (``keep``), so a future block-COO classifier can rebuild and reduce
    the chain in-block.  ``None`` for every other term, including
    non-block-eligible Sums and nested / re-reduced Sums (never
    propagated through later ops).  Currently INERT — read nowhere.
    """

    __slots__ = (
        "lazy",
        "dims",
        "param_sources",
        "var_source",
        "coef_scalar",
        "where_frames",
        "where_map_frames",
        "sum_block_meta",
    )

    def __init__(
        self,
        lazy: pl.LazyFrame | pl.DataFrame,
        dims: tuple[str, ...],
        param_sources: list[tuple[Param, int]] | None = None,
        var_source: Var | None = None,
        coef_scalar: float = 1.0,
        where_frames: tuple[pl.LazyFrame, ...] | None = None,
        where_map_frames: tuple[tuple[pl.LazyFrame, frozenset[str]], ...] | None = None,
        sum_block_meta: SumBlockMeta | None = None,
    ):
        if isinstance(lazy, pl.DataFrame):
            lazy = lazy.lazy()
        self.lazy = lazy
        self.dims = tuple(dims)
        self.param_sources = param_sources
        self.var_source = var_source
        # ``coef_scalar`` records the cumulative constant scalar (negation,
        # ``Expr * float``, ``Var * float``, ``Expr / float``) folded into
        # ``coef`` outside the Param-chain factorisation.  The LHS prune-
        # down (``_build_lhs_pruned_plan``) starts from
        # ``coef=coef_scalar`` so the rebuilt
        # ``row_index → Var → P1 → P2 …`` chain reproduces ``term.lazy``'s
        # signed coefficient.  Param/Var multiplies leave it at 1.0
        # because their factor is the Param's value column (tracked
        # separately in ``param_sources``), not a constant scalar.  Sum/
        # Where/Lag null out ``var_source`` so prune-down doesn't fire on
        # those terms — ``coef_scalar`` is irrelevant there.
        self.coef_scalar = float(coef_scalar)
        # Normalise to None or a tuple (immutable).
        if where_frames is None or len(where_frames) == 0:
            self.where_frames = None
        else:
            self.where_frames = tuple(where_frames)
        if where_map_frames is None or len(where_map_frames) == 0:
            self.where_map_frames = None
        else:
            self.where_map_frames = tuple(where_map_frames)
        # Inert Phase C-2 reconstruction recipe; read nowhere yet.
        self.sum_block_meta = sum_block_meta

    @property
    def frame(self) -> pl.DataFrame:
        """Collected eager view.  Each access re-collects — terms are
        usually consumed once at constraint emission, so we don't
        cache."""
        return self.lazy.collect()


def _to_expr(x) -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, Var):
        return x.to_expr()
    raise TypeError(f"cannot convert {type(x).__name__} to an Expr")


class Expr:
    """A sum of terms (decision-variable contributions).

    The terms can have different open-dim sets — they're concatenated,
    not broadcast.  Broadcasting happens once, at constraint emission,
    via a join to the constraint's ``over=`` row index."""

    __slots__ = ("terms",)

    def __init__(self, terms: list[_Term]):
        self.terms = terms

    def __add__(self, other):
        return Expr(self.terms + _to_expr(other).terms)

    def __sub__(self, other):
        # D1: route through ``self + (-other)`` so the subtracted operand's
        # terms inherit the scalar-``__mul__`` (×-1) recipe forwarding —
        # negating ``coef`` AND ``coef_scalar`` AND the recipe's
        # ``coef_scalar`` — instead of rebuilding the terms here and
        # dropping ``sum_block_meta``.  ``__add__`` concatenates verbatim,
        # so the ``self`` side is untouched.  Behaviourally identical to the
        # prior open-coded negate (``coef``/``coef_scalar`` negated, every
        # other field preserved).
        return self + (-_to_expr(other))

    def __radd__(self, other):
        return _to_expr(other) + self

    def __rsub__(self, other):
        return _to_expr(other) - self

    def __neg__(self):
        return self * -1.0

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            s = float(scalar)
            return Expr(
                [
                    _Term(
                        t.lazy.with_columns(coef=pl.col("coef") * s),
                        t.dims,
                        param_sources=t.param_sources,
                        var_source=t.var_source,
                        coef_scalar=t.coef_scalar * s,
                        where_frames=t.where_frames,
                        where_map_frames=t.where_map_frames,
                        # D1: a constant scalar folds into the recipe's
                        # ``coef_scalar`` exactly as it folds into
                        # ``_Term.coef_scalar`` (the walk seeds
                        # ``coef = coef_scalar``), so the rebuilt chain stays
                        # byte-identical — forward the recipe.
                        sum_block_meta=(
                            replace(
                                t.sum_block_meta,
                                coef_scalar=t.sum_block_meta.coef_scalar * s,
                            )
                            if t.sum_block_meta is not None
                            else None
                        ),
                    )
                    for t in self.terms
                ]
            )
        if isinstance(scalar, Param):
            psrc_other = scalar._sources_for_propagation()
            new = []
            for t in self.terms:
                use_lazy, use_dims, out_where_map_frames = _bake_map_before_mul(t, scalar.dims)
                # ``use_lazy`` may not physically carry every entry of
                # ``use_dims`` when extras are still deferred — compute
                # ``shared`` and the final ``.select`` against the
                # physical column set so we never touch a missing column.
                lazy_cols = set(use_lazy.collect_schema().names())
                shared = [d for d in use_dims if d in scalar.dims and d in lazy_cols]
                new_dims = tuple(dict.fromkeys(tuple(use_dims) + scalar.dims))
                if shared:
                    left_lf, right_lf = _align_enum_join_keys(use_lazy, scalar.lazy, shared)
                    j = left_lf.join(right_lf, on=shared, how="inner")
                else:
                    j = use_lazy.join(scalar.lazy, how="cross")
                joined_cols = set(j.collect_schema().names())
                select_cols = [d for d in new_dims if d in joined_cols]
                j = j.with_columns(coef=pl.col("coef") * pl.col("value")).select(
                    *select_cols, "col_id", "coef"
                )
                merged = _merge_param_sources(t.param_sources, psrc_other, flip_other=False)
                # D1: forward the block-COO reconstruction recipe through
                # the post-Sum Param multiply when ``scalar`` rides existing
                # kept rows (SAFE); else DECLINE (drop the recipe + emit the
                # loud marker).
                fwd_meta = (
                    _forward_sum_block_meta_param_mul(t.sum_block_meta, scalar, flip=False)
                    if t.sum_block_meta is not None
                    else None
                )
                new.append(
                    _Term(
                        j,
                        new_dims,
                        param_sources=merged,
                        var_source=t.var_source,
                        coef_scalar=t.coef_scalar * scalar._value_scalar,
                        where_frames=t.where_frames,
                        where_map_frames=out_where_map_frames,
                        sum_block_meta=fwd_meta,
                    )
                )
            return Expr(new)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self * (1.0 / float(other))
        if isinstance(other, Param):
            psrc_other = other._sources_for_propagation()
            new = []
            for t in self.terms:
                use_lazy, use_dims, out_where_map_frames = _bake_map_before_mul(t, other.dims)
                lazy_cols = set(use_lazy.collect_schema().names())
                shared = [d for d in use_dims if d in other.dims and d in lazy_cols]
                new_dims = tuple(dict.fromkeys(tuple(use_dims) + other.dims))
                if shared:
                    left_lf, right_lf = _align_enum_join_keys(use_lazy, other.lazy, shared)
                    j = left_lf.join(right_lf, on=shared, how="inner")
                else:
                    j = use_lazy.join(other.lazy, how="cross")
                joined_cols = set(j.collect_schema().names())
                select_cols = [d for d in new_dims if d in joined_cols]
                j = j.with_columns(coef=pl.col("coef") / pl.col("value")).select(
                    *select_cols, "col_id", "coef"
                )
                merged = _merge_param_sources(t.param_sources, psrc_other, flip_other=True)
                # D1: forward the block-COO recipe through the post-Sum
                # Param divide (direction flipped) when SAFE; else DECLINE.
                fwd_meta = (
                    _forward_sum_block_meta_param_mul(t.sum_block_meta, other, flip=True)
                    if t.sum_block_meta is not None
                    else None
                )
                new.append(
                    _Term(
                        j,
                        new_dims,
                        param_sources=merged,
                        var_source=t.var_source,
                        coef_scalar=t.coef_scalar / other._value_scalar,
                        where_frames=t.where_frames,
                        where_map_frames=out_where_map_frames,
                        sum_block_meta=fwd_meta,
                    )
                )
            return Expr(new)
        return NotImplemented


class _CstrProto:
    __slots__ = ("expr", "sense", "rhs")

    def __init__(self, expr: Expr, sense: str, rhs):
        self.expr = expr
        self.sense = sense
        self.rhs = rhs


class CstrRecord:
    """Read-only metadata for a registered constraint family.

    Returned by :meth:`Problem.cstrs_named` for emission-introspection
    tests.  ``proto`` carries the LHS ``Expr``, ``sense`` and ``rhs``
    structures; most callers only need ``over`` (whose ``height`` is the
    row count) and ``name``."""

    __slots__ = ("name", "over", "proto")

    def __init__(self, name: str, over, proto: _CstrProto):
        self.name = name
        self.over = over
        self.proto = proto

    def __repr__(self) -> str:
        n = "scalar" if self.over is None else self.over.height
        return f"CstrRecord(name={self.name!r}, rows={n})"


class _CanonicalMatrix:
    """Canonical CSC LP storage (Stage B1).

    Built once by :meth:`Problem.canonicalise`; consumers (currently only
    :meth:`Problem.write_mps`, more under B2/B3) walk the arrays
    read-only.  ``val`` / ``col_obj`` / ``row_lb`` / ``row_ub`` carry
    the POST-Layer-2 scaled coefficients — the side vectors have
    already been baked in.  Row indices are 0-based over CONSTRAINT
    rows only (no objective row); column indices are 0-based over the
    full ``Problem._next_col`` range.
    """

    __slots__ = (
        "n_rows",
        "n_cols",
        "nnz",
        "col_ptr",
        "row_idx",
        "val",
        "row_lb",
        "row_ub",
        "sense_char",
        "col_obj",
        "col_lb",
        "col_ub",
        "col_int",
        "col_names",
        "row_names",
    )

    def __init__(
        self,
        *,
        n_rows: int,
        n_cols: int,
        nnz: int,
        col_ptr: np.ndarray,
        row_idx: np.ndarray,
        val: np.ndarray,
        row_lb: np.ndarray,
        row_ub: np.ndarray,
        sense_char: np.ndarray,
        col_obj: np.ndarray,
        col_lb: np.ndarray,
        col_ub: np.ndarray,
        col_int: np.ndarray,
        col_names: list[str],
        row_names: list[str],
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.nnz = nnz
        self.col_ptr = col_ptr
        self.row_idx = row_idx
        self.val = val
        self.row_lb = row_lb
        self.row_ub = row_ub
        self.sense_char = sense_char
        self.col_obj = col_obj
        self.col_lb = col_lb
        self.col_ub = col_ub
        self.col_int = col_int
        self.col_names = col_names
        self.row_names = row_names


def _split_terms(terms: dict, side: str) -> tuple[Expr | None, Param | float]:
    """Sort {label: term} into (variable Expr, constant Param-or-float).
    Variable terms are summed into a single Expr; constant terms are
    summed into a single Param (or scalar if no Params)."""
    var_acc: Expr | None = None
    const_acc: Param | float = 0.0
    for label, t in terms.items():
        if isinstance(t, (Var, Expr)):
            e = t.to_expr() if isinstance(t, Var) else t
            var_acc = e if var_acc is None else var_acc + e
        elif isinstance(t, Param):
            const_acc = (
                t if isinstance(const_acc, (int, float)) and const_acc == 0 else const_acc + t
            )
        elif isinstance(t, (int, float)):
            const_acc = (
                const_acc + float(t)
                if isinstance(const_acc, (int, float))
                else const_acc + float(t)
            )
        else:
            raise TypeError(f"{side}_terms[{label!r}]: unsupported term type {type(t).__name__}")
    return var_acc, const_acc


def _sub_const(a, b):
    """Compute a − b where each is Param-or-scalar."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) - float(b)
    if isinstance(b, (int, float)) and b == 0.0:
        return a
    if isinstance(a, (int, float)) and a == 0.0:
        return -b
    return a - b


# ---------------------------------------------------------------------------
# Sum / Where


def Lag(var, lag_frame: pl.DataFrame, time_dim: str, lag_col: str) -> Expr:
    """Return an Expr that, for each (carry_dims, ``time_dim``) in
    ``lag_frame``, references ``var`` at (carry_dims, ``lag_col``).

    Used for shifting variables in time, e.g. for storage state-change:

        v_state[n, d, t]  -  v_state[n, d, t_prev]

      = v_state - Lag(v_state, dtttdt, "t", "t_prev_within_timeset")

    ``lag_frame`` carries the (d, t, t_prev) lookup; ``carry_dims`` are
    the columns shared between ``var`` and ``lag_frame`` other than the
    time dim itself (typically ``d``).
    """
    if isinstance(var, Var):
        var = var.to_expr()
    # accept either eager or lazy lag_frame; need column names up-front
    if isinstance(lag_frame, pl.LazyFrame):
        lag_lf = lag_frame
        lag_cols = lag_lf.collect_schema().names()
    else:
        lag_lf = lag_frame.lazy()
        lag_cols = lag_frame.columns
    new_terms: list[_Term] = []
    for t in var.terms:
        # We need to know if time_dim is among the term's columns.  The
        # term's lazy schema mirrors its dims + (col_id, coef), so
        # checking t.dims is enough.
        if time_dim not in t.dims:
            new_terms.append(t)
            continue
        # Bake any deferred Where filters before the rename — Lag
        # changes row identity (time-shift) so the filter must be
        # applied first.  Both pure-filter and map-effect frames bake
        # here; the map-effect bake extends the term's dims with extras.
        t_lazy = _apply_where_frames(t.lazy, t.dims, t.where_frames)
        t_lazy, term_dims = _apply_where_map_frames(t_lazy, t.dims, t.where_map_frames)
        carry = [c for c in lag_cols if c in term_dims and c != time_dim and c != lag_col]
        lagged = t_lazy.rename({time_dim: "_lag_src"})
        # Align Enum dtypes on the carry columns (symmetric keys) and on
        # the asymmetric (lag_col, _lag_src) pair.  The asymmetric pair
        # is handled by temporarily renaming so we can reuse the
        # subset-aware helper, then rename back.
        lag_lf_a, lagged_a = _align_enum_join_keys(lag_lf, lagged, carry)
        lag_keyed = lag_lf_a.rename({lag_col: "__lag_join_key"})
        lagged_keyed = lagged_a.rename({"_lag_src": "__lag_join_key"})
        lag_keyed, lagged_keyed = _align_enum_join_keys(lag_keyed, lagged_keyed, ["__lag_join_key"])
        lag_lf_a = lag_keyed.rename({"__lag_join_key": lag_col})
        lagged_a = lagged_keyed.rename({"__lag_join_key": "_lag_src"})
        j = lag_lf_a.join(
            lagged_a, left_on=carry + [lag_col], right_on=carry + ["_lag_src"], how="inner"
        )
        new_terms.append(
            _Term(
                j.select(*[d for d in term_dims if d != time_dim], time_dim, "col_id", "coef"),
                term_dims,
                param_sources=t.param_sources,
            )
        )
    return Expr(new_terms)


def Where(expr, frame: pl.DataFrame) -> Expr:
    """Inner-join an Expr against ``frame``.  Two effects in one op:

    * Filter — rows of the term whose shared-column values don't
      appear in ``frame`` are dropped (e.g. ``Where(v_flow, wind_only)``
      keeps only the wind rows).
    * Map — any columns of ``frame`` that the term doesn't already
      carry become *new open dims* of the resulting term (e.g.
      ``Where(v_flow, flow_to_n)`` where ``flow_to_n`` has columns
      ``(p, source, sink, n)`` adds ``n`` so the term can be bound
      to a constraint indexed by ``(n, t)``).

    The pure-filter case (no extras) does *not* bake the join into
    ``t.lazy`` — it records ``frame`` into ``_Term.where_frames`` so
    the LHS prune-down (``_build_lhs_pruned_plan``) can apply it at the
    leaf-rebuild step (where the row count is bounded by the smaller of
    ``Var.frame`` and the row_index key set).  ``var_source`` / Param
    chain / ``coef_scalar`` are preserved.

    The map-effect case (extras non-empty) ALSO defers: ``frame`` is
    recorded into ``_Term.where_map_frames`` as ``(frame_lf,
    frozenset(extras))``, the term's dims are extended with the extras,
    and ``var_source`` / Param chain / ``coef_scalar`` / ``where_frames``
    are preserved.  The deferred inner-join (which produces the extras
    dim columns) is baked at leaf-rebuild time (or at Sum / Lag /
    consumer fallback) via :func:`_apply_where_map_frames`.  Param
    multiplies AFTER the map-effect Where bake first iff their dims
    overlap any pending extras (correctness); otherwise the deferral
    propagates through.

    Set ``POLAR_HIGH_DISABLE_WHERE_PUSHDOWN=1`` to recover today's
    behaviour (eager join + clear metadata) verbatim for BOTH branches.
    """
    if isinstance(expr, Var):
        expr = expr.to_expr()
    if isinstance(frame, pl.LazyFrame):
        frame_lf = frame
        frame_cols = frame_lf.collect_schema().names()
    else:
        frame_lf = frame.lazy()
        frame_cols = frame.columns
    disabled = _where_pushdown_disabled()
    new: list[_Term] = []
    for t in expr.terms:
        # Term schema = dims + (col_id, coef).  Only dims overlap with
        # ``frame``'s join keys (col_id is internal).
        term_cols = set(t.dims) | {"col_id", "coef"}
        shared = [c for c in frame_cols if c in term_cols]
        extra = tuple(c for c in frame_cols if c not in term_cols)
        if disabled:
            # Safety fallback: today's verbatim behaviour — eager
            # inner-join + clear all leaf-rebuild metadata.  Latent
            # ``shared==[] and extras!=()`` bug preserved verbatim too
            # (no cross-join), since this branch's job is to recover
            # exact prior semantics.
            f = t.lazy
            if shared:
                f, frame_lf_a = _align_enum_join_keys(f, frame_lf, shared)
                f = f.join(frame_lf_a, on=shared, how="inner")
            new.append(_Term(f, t.dims + extra, param_sources=t.param_sources))
            continue
        if extra:
            # Map effect — the frame introduces new open dims (extras).
            # DEFER (pushdown enabled): record ``(frame_lf,
            # frozenset(extras))`` into ``where_map_frames``, extend the
            # term's dims with the extras, and PRESERVE
            # ``var_source`` / ``param_sources`` / ``coef_scalar`` /
            # ``where_frames`` so the LHS prune-down (and, later,
            # block-COO) can still reach the leaves.  The dim-extending
            # inner-join is baked at leaf-rebuild / Sum / Lag / fallback
            # via :func:`_apply_where_map_frames`.  Param multiplies bake
            # first iff their dims overlap any pending extras (see
            # :func:`_bake_map_before_mul`).
            prev_map = t.where_map_frames or ()
            # D1: forward the block-COO recipe through a post-Sum map-effect
            # Where (the nodeBalance ``n``-introduction is supported).
            # Record the same ``(frame_lf, frozenset(extras))`` the
            # classifier consumes and grow ``keep`` by the extras so the
            # rebuilt open-dim set still partitions into keep ∪ reduce_dims.
            fwd_meta = None
            if t.sum_block_meta is not None:
                m = t.sum_block_meta
                fwd_meta = replace(
                    m,
                    where_map_frames=(m.where_map_frames or ()) + ((frame_lf, frozenset(extra)),),
                    keep=tuple(m.keep) + tuple(d for d in extra if d not in m.keep),
                )
            new.append(
                _Term(
                    t.lazy,
                    t.dims + extra,
                    param_sources=t.param_sources,
                    var_source=t.var_source,
                    coef_scalar=t.coef_scalar,
                    where_frames=t.where_frames,
                    where_map_frames=prev_map + ((frame_lf, frozenset(extra)),),
                    sum_block_meta=fwd_meta,
                )
            )
            continue
        # Pure-filter — defer.  When ``shared`` is empty the filter is a
        # no-op for this term's row set (no shared keys to constrain
        # on); pass the term through unchanged.  Otherwise record the
        # frame into ``where_frames`` so prune-down / fallback can apply
        # it leaf-level.
        if not shared:
            # D1: no-op filter for this term's row set — forward the recipe
            # verbatim (the term itself is passed through unchanged).
            new.append(
                _Term(
                    t.lazy,
                    t.dims,
                    param_sources=t.param_sources,
                    var_source=t.var_source,
                    coef_scalar=t.coef_scalar,
                    where_frames=t.where_frames,
                    where_map_frames=t.where_map_frames,
                    sum_block_meta=t.sum_block_meta,
                )
            )
            continue
        prev = t.where_frames or ()
        # D1: forward the block-COO recipe through a post-Sum pure-filter
        # Where, recording the filter into the recipe's ``where_frames`` so
        # the in-block rebuild applies it leaf-level (no dim change).
        fwd_meta = None
        if t.sum_block_meta is not None:
            m = t.sum_block_meta
            fwd_meta = replace(m, where_frames=(m.where_frames or ()) + (frame_lf,))
        new.append(
            _Term(
                t.lazy,
                t.dims,
                param_sources=t.param_sources,
                var_source=t.var_source,
                coef_scalar=t.coef_scalar,
                where_frames=prev + (frame_lf,),
                where_map_frames=t.where_map_frames,
                sum_block_meta=fwd_meta,
            )
        )
    return Expr(new)


def Sum(expr, over: tuple[str, ...] | str | None = None, where: pl.DataFrame | None = None) -> Expr:
    """Aggregate an Expr.  ``over`` lists the dims to sum out; the
    remaining dims become the term's open dims.  ``where`` is an index
    frame that pre-filters the term frames (inner join on shared
    columns) before the group-by-sum.

    ``Sum(expr)`` with ``over=None`` collapses every open dim — useful
    for a scalar (objective term, single-row constraint).

    Any deferred :func:`Where` filters recorded on each term
    (``t.where_frames``) are baked into ``t.lazy`` *before* the
    aggregation — Sum collapses dims so the filter could not be
    applied at the leaf level downstream.  The result clears
    ``var_source`` / ``where_frames`` (today's behaviour); the
    per-term ``where`` kwarg is applied on top, unchanged.
    """
    if isinstance(expr, Var):
        expr = expr.to_expr()
    if over is None:
        # collapse every open dim that appears in any term
        all_dims = []
        for t in expr.terms:
            for d in t.dims:
                if d not in all_dims:
                    all_dims.append(d)
        over = tuple(all_dims)
    elif isinstance(over, str):
        over = (over,)
    else:
        over = tuple(over)

    where_lf = None
    where_cols: list[str] = []
    if where is not None:
        if isinstance(where, pl.LazyFrame):
            where_lf = where
            where_cols = where_lf.collect_schema().names()
        else:
            where_lf = where.lazy()
            where_cols = where.columns

    new_terms: list[_Term] = []
    for t in expr.terms:
        # Bake any deferred Where filters before the per-term ``where``
        # kwarg join — Sum collapses dims, so the filter has to land
        # at the leaf level.  Pure-filter frames first (semi-join), then
        # map-effect frames (inner-join, dim-extending) — the latter
        # produces the extras columns that ``term_dims`` claims.
        f = _apply_where_frames(t.lazy, t.dims, t.where_frames)
        f, term_dims = _apply_where_map_frames(f, t.dims, t.where_map_frames)
        if where_lf is not None:
            shared = [c for c in where_cols if c in term_dims]
            if shared:
                where_sub = where_lf.select(shared).unique()
                f, where_sub = _align_enum_join_keys(f, where_sub, shared)
                f = f.join(where_sub, on=shared, how="inner")
        keep = tuple(d for d in term_dims if d not in over)
        # Phase C-2 (INERT): capture the pre-Sum reconstruction recipe
        # BEFORE the survivor filter below mutates ``psrc``.  Set ONLY at
        # the point of a block-eligible reduction — ``var_source`` present,
        # a non-empty ``param_sources`` list, and a non-empty ``over``.
        # ``param_sources`` is captured FULL (NOT survivor-filtered) so
        # block-COO sees every factor, including Params whose dims are
        # summed out (e.g. ``p_unitsize`` over ``p``).  A nested / re-
        # reduced Sum (``t`` already carries a recipe) sets the new term's
        # to None so a stale recipe never propagates.  Nothing reads this
        # field yet, so the capture is behaviorally inert.
        block_meta: SumBlockMeta | None = None
        if (
            t.var_source is not None
            and isinstance(t.param_sources, list)
            and len(t.param_sources) > 0
            and over
            and t.sum_block_meta is None
        ):
            block_meta = SumBlockMeta(
                var_source=t.var_source,
                param_sources=tuple(t.param_sources),
                coef_scalar=t.coef_scalar,
                where_frames=t.where_frames,
                where_map_frames=t.where_map_frames,
                reduce_dims=tuple(over),
                keep=keep,
            )
        # If the Sum collapses any of the source-Param's dim columns,
        # multiple cells (with different param values) get merged into
        # one — we can no longer recover the per-cell param contribution
        # from the aggregated coef.  In that case we must drop the
        # tracking metadata.  Tracking survives only when every source
        # Param's dim_signature is contained in ``keep``.
        psrc = t.param_sources
        if psrc is not None and over:
            over_set = set(over)
            survivors = [(p, s) for (p, s) in psrc if not any(d in over_set for d in p.dims)]
            psrc = survivors if survivors else None
        if keep:
            f = (
                f.group_by(list(keep) + ["col_id"])
                .agg(pl.col("coef").sum())
                .select(*keep, "col_id", "coef")
            )
        else:
            f = f.group_by("col_id").agg(pl.col("coef").sum())
        # D1: collapse-all preserve.  When the resolved ``over`` is EMPTY
        # this Sum is a no-op relabel over an already-collapsed term (e.g.
        # ``set_objective``'s outer ``Sum(expr, over=None)`` over already-
        # scalar objective terms) — forward the incoming recipe verbatim
        # instead of dropping it.  A re-reducing outer Sum (non-empty
        # ``over``) over a meta-bearing term still drops the recipe: the
        # capture guard above blocks re-capture, and this forward clause is
        # gated on ``not over`` so it fires ONLY for the no-op collapse.
        out_meta = (
            block_meta if block_meta is not None else (t.sum_block_meta if not over else None)
        )
        new_terms.append(_Term(f, keep, param_sources=psrc, sum_block_meta=out_meta))
    return Expr(new_terms)


# ---------------------------------------------------------------------------
# Problem container


class Problem:
    """LP container.  Generic — no flextool-specific knowledge."""

    def __init__(self, dense_axes: tuple[str, ...] | None = None) -> None:
        """Construct an empty LP container.

        Pure polar-high is a generic LP kernel; scaling decisions are
        left to the caller.  See :mod:`polar_high.autoscale` for the
        opt-in autoscaler (Layer 1 detect + Layer 3 recommendation)
        that callers (e.g. FlexTool) use to drive
        ``user_bound_scale`` / ``user_objective_scale`` automatically.

        ``dense_axes`` — the explicit client contract for the block-COO
        LHS arm.  When the client (e.g. FlexTool) declares the dense
        trailing axes once here (e.g. ``Problem(dense_axes=("d", "t"))``),
        it makes a binding PROMISE about every frame it passes that
        contains those columns:

            the frame is globally lexicographically sorted by
            ``(other_dims_in_declared_order..., *dense_axes)`` — i.e. the
            declared dense axes are the trailing sort keys, in the given
            order, and the leading dims form a sorted prefix.

        This lets block-COO slice the dense suffix of each Var with NO
        re-sort (a re-sort would cost more than the multiply itself).
        polar-high VERIFIES this promise cheaply (a single-pass monotonic
        scan — see :func:`_verify_dense_sorted`) on every Var that the
        block-COO arm classifies + fires on, and RAISES a clear
        ``ValueError`` naming the Var if the client breaks it.  Frames
        that do not contain the dense axes (e.g. an investment Var
        ``("p", "d")`` when ``dense_axes=("d", "t")``) simply do not fire
        block-COO and are unaffected.  ``None`` (default) leaves the
        block-COO arm dormant — it only fires once dense axes are declared
        (here or via :meth:`declare_dense_axes`).
        """
        self._dense_axes: tuple[str, ...] | None = None
        self.declare_dense_axes(dense_axes)
        self._vars: dict[str, Var] = {}
        self._cstrs: list[tuple[str, _CstrProto, pl.DataFrame | None]] = []
        self._next_col = 0
        self._obj_terms: list[_Term] = []
        self._obj_sense = "min"
        self._obj_offset: float = 0.0
        # Generic small-coefficient cutoff.  When > 0.0, any constraint-
        # matrix coefficient OR row-bound (RHS) term whose absolute value
        # is strictly LESS THAN this threshold is floored to exactly
        # ``0.0`` at LP-assembly time, just before the coefficients are
        # handed to HiGHS.  Values exactly equal to the threshold are
        # kept; ``±inf`` row-bound sentinels are never affected
        # (``abs(inf) < thr`` is false).  Applied in BOTH build paths:
        # :meth:`_solve_streaming` (per-family ``addRows``) and
        # :meth:`_build_canonical_matrix` (the CSC ``val`` + ``row_lb`` /
        # ``row_ub`` arrays consumed by the non-streaming ``passModel``
        # path and by :class:`WarmProblem`).  polar-high knows nothing
        # about why a caller wants this; it is a pure numeric floor.
        # Default ``0.0`` ⇒ OFF — behaviour is byte-identical to code
        # that never sets it.  The floor REPLACES the value with 0.0; it
        # never drops a matrix entry, so matrix structure/determinism is
        # preserved.
        self.coef_zero_threshold: float = 0.0
        # HiGHS option-name → value applied via setOptionValue at solve()
        # time.  Populated by ``set_solver_options`` (or by flextool's
        # ``build_flextool`` when ``FlexData.solver_options`` is set).
        # An explicit ``options`` kwarg on ``solve()`` overrides this.
        self._solver_options: dict | None = None
        # ``save_memory=True`` on :meth:`solve` flips this flag once
        # the Python-side LP source-of-truth has been dropped (see
        # :meth:`_release_python_lp_inputs`).  Subsequent ``solve()``
        # calls then raise — there is no LP left to re-emit.
        self._released: bool = False
        # Layer 2 side-vector scaling — see :mod:`flextool.engine_polars.
        # autoscale._layer2`.  When set (by ``apply_layer2``), the four
        # LP consumers (``write_mps``, ``_build_lp_arrays``,
        # ``_solve_streaming``, ``WarmProblem._initial_build``) multiply
        # each collected LHS / objective / RHS coefficient by these
        # power-of-two factors at emit time instead of mutating the
        # underlying lazy plans.  ``None`` = no scaling (default;
        # consumers behave identically to pre-Layer-2 code).
        #
        # ``_layer2_col_factor[col_id]``: applied to LHS and cost terms
        # (cost = ``col_factor[col_id]`` only — there is no objective
        # row in the row-factor array).  Column bounds are NOT scaled
        # here; ``apply_layer2`` mutates ``Var.lower`` / ``Var.upper``
        # in place for that.
        # ``_layer2_row_factor[row_id]``: applied to LHS and RHS;
        # ``row_id`` is the 0-based index over constraint rows in the
        # order constraints appear in ``self._cstrs`` (NOT including
        # the objective row, which has no row factor).
        # ``_layer2_locked``: when True, ``add_cstr`` raises — the side
        # vectors are sized for the constraints that existed at
        # ``apply_layer2`` time, so adding more rows would silently
        # leave them un-scaled.  Set by a later commit's
        # ``apply_layer2`` rewrite; this commit only adds the attr.
        self._layer2_col_factor: np.ndarray | None = None
        self._layer2_row_factor: np.ndarray | None = None
        self._layer2_locked: bool = False
        # Stage B1 — canonical CSC store.  Populated lazily by
        # :meth:`canonicalise`; consumed (currently) only by
        # :meth:`write_mps`.  ``_canonical_dirty`` flips to ``True`` on
        # ``add_var`` / ``add_cstr`` (and should be flipped by future
        # ``apply_layer2`` reruns) so the next ``canonicalise`` call
        # rebuilds.  Side vectors (``_layer2_col_factor`` /
        # ``_layer2_row_factor``) are baked into ``_matrix.val`` /
        # ``col_obj`` / ``row_lb`` / ``row_ub`` at build time per
        # orchestrator decision D8.
        self._matrix: _CanonicalMatrix | None = None
        self._canonical_dirty: bool = True

    def declare_dense_axes(self, axes: tuple[str, ...] | None) -> None:
        """Declare the dense trailing axes for block-COO (see __init__ contract).

        Equivalent to passing ``dense_axes=`` to the constructor; provided so
        callers that receive an already-constructed Problem (e.g. FlexTool's
        ``build_flextool`` step, which builds the Problem first and populates
        it afterwards) can declare them.  Pass ``None`` to clear.
        """
        if axes is None:
            self._dense_axes = None
            return
        if not isinstance(axes, (tuple, list)) or not all(isinstance(a, str) for a in axes):
            raise TypeError(
                f"declare_dense_axes expects a tuple/list of str (or None); got {axes!r}"
            )
        self._dense_axes = tuple(axes) if axes else None

    def set_solver_options(self, options: dict | None) -> None:
        """Store HiGHS options to be applied in ``solve()``.  Pass ``None``
        to clear.  Keys are HiGHS canonical option names (``presolve``,
        ``solver``, ``parallel``, ``time_limit`` etc); values must be
        already coerced to the type HiGHS expects (str/int/float/bool).
        Unknown keys are tolerated (a warning is emitted at solve time)."""
        self._solver_options = dict(options) if options else None

    def set_solver_option(self, name: str, value) -> None:
        """Set a single HiGHS option, leaving the rest untouched.

        Convenience for callers that want to add one knob without
        re-passing the whole dict (e.g. ``user_bound_scale`` set by
        the autoscaler).  Equivalent to a dict merge plus
        :meth:`set_solver_options`.
        """
        opts = dict(self._solver_options) if self._solver_options else {}
        opts[name] = value
        self._solver_options = opts

    def get_solver_option(self, name: str):
        """Return the caller-set value of ``name`` (or ``None`` if unset).

        Reads ``self._solver_options`` only — does NOT consult HiGHS
        (the option may not have been pushed to a live ``Highs``
        instance yet).  Returns ``None`` for unset options so callers
        can use ``if get_solver_option(...) is not None`` to test
        explicit setting.
        """
        if not self._solver_options:
            return None
        return self._solver_options.get(name)

    # -- variables -------------------------------------------------------

    def add_var(
        self,
        name: str,
        dims: tuple[str, ...] | str,
        index: pl.DataFrame,
        lower: float = 0.0,
        upper: float = float("inf"),
        integer: bool = False,
    ) -> Var:
        if self._layer2_locked:
            raise RuntimeError(
                "Problem.add_var called after apply_layer2 — adding "
                "variables after Layer 2 is not supported (would need "
                "to extend the col_factor side vector)."
            )
        if isinstance(dims, str):
            dims = (dims,)
        dims = tuple(dims)
        for d in dims:
            if d not in index.columns:
                raise ValueError(f"index frame {index.columns} missing dim {d!r}")
        if name in self._vars:
            raise ValueError(f"variable {name!r} already declared")

        n = index.height
        col_ids = np.arange(self._next_col, self._next_col + n, dtype=np.int64)
        self._next_col += n
        frame = index.select(*dims).with_columns(col_id=pl.Series(col_ids))
        v = Var(name, dims, frame, lower, upper, integer)
        self._vars[name] = v
        self._canonical_dirty = True
        return v

    # -- constraints -----------------------------------------------------

    def add_cstr(
        self,
        name: str,
        *,
        over: pl.DataFrame | None = None,
        sense: str,
        lhs_terms: dict[str, Var | Expr | Param | int | float],
        rhs_terms: dict[str, Var | Expr | Param | int | float] | None = None,
    ) -> None:
        """Add a constraint of the form  ``Σ lhs_terms  sense  Σ rhs_terms``.

        Each term entry is either:
          * a ``Var`` or ``Expr``   — variable contribution, or
          * a ``Param``, ``int`` or ``float`` — constant contribution.

        The engine sorts variables and constants out per side, builds
        ``(lhs_var − rhs_var) sense (rhs_const − lhs_const)``, and adds
        the row to highspy at solve time.  Labels (the dict keys) are
        used in row names and diagnostics.
        """
        if self._layer2_locked:
            raise RuntimeError(
                "Problem.add_cstr called after apply_layer2 — adding "
                "constraints after Layer 2 is not supported (would need "
                "to extend the row_factor side vector)."
            )
        if rhs_terms is None:
            rhs_terms = {}
        if not isinstance(lhs_terms, dict) or not isinstance(rhs_terms, dict):
            raise TypeError("lhs_terms and rhs_terms must be dicts {label: term}")
        if sense not in ("<=", ">=", "=="):
            raise ValueError(f"sense must be '<=', '>=' or '=='; got {sense!r}")
        if name in {n for n, _, _ in self._cstrs}:
            raise ValueError(f"constraint {name!r} already declared")

        lhs_var, lhs_const = _split_terms(lhs_terms, "lhs")
        rhs_var, rhs_const = _split_terms(rhs_terms, "rhs")

        # combine into canonical (var_expr, sense, rhs_const) form
        var_expr = lhs_var if lhs_var is not None else Expr([])
        if rhs_var is not None:
            var_expr = var_expr - rhs_var

        # net constant on the RHS:  rhs_const − lhs_const
        const = _sub_const(rhs_const, lhs_const)

        proto = _CstrProto(var_expr, sense, const)
        self._cstrs.append((name, proto, over))
        self._canonical_dirty = True

    # -- introspection ---------------------------------------------------

    def cstr_names(self) -> list[str]:
        """All constraint family names currently registered, in declaration
        order.  Useful for emission audits and debugging."""
        return [n for n, _, _ in self._cstrs]

    def cstrs_named(self, name: str) -> list[CstrRecord]:
        """Return constraint metadata records matching ``name``.

        An exact-name match returns the single record; otherwise a prefix
        match returns every record whose name starts with ``name + "_"``
        (so passing ``"minimum_uptime"`` returns both
        ``minimum_uptime_linear`` and ``minimum_uptime_integer``).

        Each :class:`CstrRecord` carries:
          * ``name``: full registered name of the constraint family;
          * ``over``: the polars DataFrame of axis tuples (``len(over)``
            is the row count);
          * ``proto``: the underlying ``_CstrProto`` (``expr``, ``sense``,
            ``rhs``) for advanced introspection.
        """
        # Exact match first.
        for n, proto, over in self._cstrs:
            if n == name:
                return [CstrRecord(name=n, over=over, proto=proto)]
        prefix = name + "_"
        return [
            CstrRecord(name=n, over=over, proto=proto)
            for n, proto, over in self._cstrs
            if n.startswith(prefix)
        ]

    def cstr_row_count(self, name: str) -> int:
        """Total LP-row count across all constraint families matching
        ``name`` (exact or prefix; see :meth:`cstrs_named`).  Returns 0
        when no families match — letting callers distinguish "absent"
        from "empty" without exception handling.  A scalar constraint
        (``over=None``) counts as one row."""
        total = 0
        for rec in self.cstrs_named(name):
            total += 1 if rec.over is None else int(rec.over.height)
        return total

    # -- objective -------------------------------------------------------

    def set_objective(self, expr: Expr | Var, sense: str = "min") -> None:
        if isinstance(expr, Var):
            expr = expr.to_expr()
        scalar = Sum(expr, over=None)  # collapse every dim
        for t in scalar.terms:
            if t.dims:
                raise RuntimeError(f"objective term still has open dims {t.dims}")
        self._obj_terms = scalar.terms
        if sense not in ("min", "max"):
            raise ValueError(f"sense must be 'min' or 'max'; got {sense!r}")
        self._obj_sense = sense

    def add_obj_constant(self, value: float) -> None:
        """Accumulate a constant into the objective offset.  HiGHS adds
        this to the reported ``getObjectiveValue()`` after solve, so it
        shows up in ``Solution.obj`` even though no decision variable
        carries it.  Used for pure-Param objective terms like the §8.1
        existing-entity fixed cost."""
        self._obj_offset += float(value)

    # -- release ---------------------------------------------------------

    def _release_python_lp_inputs(self) -> None:
        """Drop the Python-side LP source-of-truth.

        Called from :meth:`_solve_streaming` after all columns + rows
        have been emitted to HiGHS, when ``solve(save_memory=True)``
        was requested.  Why this exists: ``HiGHS.run()`` no longer needs
        any of the polars LazyFrames, numpy COO buffers, or rhs Param
        frames that polar-high accumulated while building the model —
        HiGHS owns its own copy.  Releasing the polar side here makes
        steady-state RSS during ``run()`` comparable to solvers that
        serialise the LP to disk and free their Python representation
        (the comparison this exists to make fair).

        What stays alive: ``self._vars`` and each :class:`Var.frame`
        (its ``col_id`` column maps HiGHS column indices back to user-
        space dim tuples, which :meth:`Solution.value` reads on demand).
        ~288 MB at N=3000 dense — acceptable; without it the Solution
        is unusable.

        What goes: every ``_Term.lazy`` plan (objective and constraint
        LHS), every ``_CstrProto.rhs`` reference (which may pin a Param
        frame), every ``_Term.sum_block_meta`` recipe (whose FULL,
        un-survivor-filtered ``param_sources`` pins the summed-out dense
        Params for the Problem's lifetime — see below), and the
        constraint-family list itself.  Sets :attr:`_released` so
        :meth:`solve` refuses to run again — the Problem is no longer
        re-emittable.

        Why ``sum_block_meta`` must go too: a ``Sum``-reduced term clears
        its own ``var_source`` and *survivor-filters* ``param_sources``
        (dropping the summed-out factors), but the captured
        :class:`SumBlockMeta` recipe snapshots the FULL pre-Sum chain —
        including those summed-out ``(d,t)`` Params (e.g. a profile ×
        availability product collapsed by a ``Sum``).  The matrix build
        is the recipe's last reader; once HiGHS owns the assembled LP the
        recipe is dead weight that would otherwise keep every snapshotted
        dense Param (and its eager source frame) alive past release,
        defeating the save_memory contract.  Nulling ``t.lazy`` /
        ``t.param_sources`` alone does NOT release them — the recipe is a
        separate, independent reference.
        """
        # Objective terms: drop lazy plans first so any Param objects
        # referenced via ``param_sources`` aren't extended past the
        # constraint walk below.  Also drop the SumBlockMeta recipe, whose
        # FULL param chain independently pins the summed-out dense Params.
        for t in self._obj_terms:
            t.lazy = None  # type: ignore[assignment]
            t.param_sources = None
            t.sum_block_meta = None
        self._obj_terms = []

        # Constraint families: clear each Expr's term list and drop the
        # rhs reference (which may be a Param holding a sizeable eager
        # frame).  Clearing the term list drops each term object whole,
        # so its ``sum_block_meta`` recipe (and the dense Params that
        # recipe pins) goes with it — no separate null needed here.  We
        # don't touch ``over`` — it's typically the row-index DataFrame,
        # already small compared to the LHS plans we just dropped, and
        # stripping it would complicate any future diagnostic that wants
        # to report which family came last.
        for _name, proto, _over in self._cstrs:
            proto.expr.terms = []
            proto.rhs = None
        self._cstrs = []

        # Stage B1 — drop the canonical store too.  It's the same data
        # in a different shape; carrying it past release would defeat
        # the save_memory contract.  The next ``canonicalise`` call
        # would have nothing to read from anyway (``_cstrs`` and
        # ``_obj_terms`` are empty).
        self._matrix = None
        self._canonical_dirty = True

        self._released = True

    # -- solve -----------------------------------------------------------

    def solve(
        self,
        *,
        options: dict | None = None,
        keep_solver: bool = False,
        streaming: bool = True,
        save_memory: bool = False,
        tmp_dir: str | os.PathLike | None = None,
    ) -> Solution:
        """Solve the LP and return a :class:`Solution`.

        Parameters
        ----------
        options
            Per-call HiGHS options dict (overrides ``set_solver_options``).
        keep_solver
            When ``True``, the live HiGHS instance is kept on the returned
            :class:`Solution` so callers can inspect it post-solve (e.g.
            ``sol.highs.writeModel("model.mps")``).  Default ``False`` —
            the C-side LP storage is released as soon as primal/dual/
            objective have been extracted.
        streaming
            When ``True`` (default), columns are added once via ``addCols``
            and each constraint family is emitted to HiGHS via ``addRows``
            immediately after its COO triples are built; the family's local
            arrays then go out of scope before the next family is processed.
            This caps peak memory at one family's COO + the running HiGHS
            LP.  When ``False``, the entire model is assembled into a single
            :class:`highspy.HighsLp` and loaded via ``passModel`` —
            numerically identical results either way; ``False`` is mostly
            useful for benchmarking the legacy path.
        save_memory
            Single one-shot knob that trades wall time for peak RSS.  When
            ``True``, two things happen right before ``HiGHS.run()``:

            1. polar-high's polars/numpy LP source-of-truth (constraint
               family ``Expr.terms`` lists, ``_CstrProto.rhs`` Param
               frames, objective ``_Term.lazy`` plans, the caller-side
               ``col_names`` / ``row_names`` lists) is dropped.  Only the
               per-:class:`Var` ``col_id`` frames survive — they are
               needed by :meth:`Solution.value` to map column indices
               back to user-space dim tuples.
            2. The HiGHS instance is round-tripped through disk: the LP
               is written to a temp MPS file, the original ``Highs()`` is
               cleared and discarded, ``malloc_trim(0)`` is called (best
               effort, Linux only), a fresh ``Highs()`` is created, the
               same solver options are re-applied, and the MPS file is
               read back.  This resets the HiGHS allocator's high-water
               mark — the C++ side accumulates ~5 GB of slack from the
               incremental ``addRows`` loading path that ``readModel``
               avoids by sizing once up front.

            Cost: ~+90 s of MPS file I/O at N=3000 dense.  Benefit: peak
            RSS drops by ~5 GB on the same problem.  Intended for one-
            shot single-solve benchmarks where warm-start / re-solve
            isn't needed.  After the call returns, the :class:`Problem`
            is in a "released" state and any further :meth:`solve` call
            raises ``RuntimeError`` (the polar-side source AND the
            original HiGHS instance with its basis have both been
            discarded, so neither a fresh re-solve nor a WarmProblem-
            style update is possible).  Honoured only by the streaming
            path; on ``streaming=False`` a warning is emitted and the
            flag is ignored.  Default ``False``.
        tmp_dir
            Directory for the temporary MPS file written by the
            ``save_memory=True`` round-trip.  ``None`` (default) uses
            the system temp dir (``$TMPDIR`` / ``/tmp``).  Set when the
            caller wants the spill file on a specific volume (e.g. the
            same filesystem as its workspace, or a per-job scratch
            directory) instead of the system default.  Ignored when
            ``save_memory=False``.
        """
        # Guard against re-entry on a Problem whose LP source has
        # already been dropped — see ``save_memory`` above.  We can
        # neither rebuild the matrix from ``self._cstrs`` (cleared) nor
        # recompute the objective from ``self._obj_terms`` (cleared);
        # the caller must construct a fresh :class:`Problem`.
        if getattr(self, "_released", False):
            raise RuntimeError(
                "Problem.solve(save_memory=True) was called previously; "
                "the LP source has been released and the Problem cannot be "
                "solved again.  Rebuild from scratch, or use the default "
                "save_memory=False for re-solvable Problems."
            )

        if streaming:
            # Column arrays are built inside ``_solve_streaming`` so they
            # die at the end of that frame rather than persisting through
            # ``HiGHS.run()`` in this caller's locals.  The non-streaming
            # path below recomputes its own column arrays inside the
            # HiGHS adapter (``LpView.from_problem``), so we don't build
            # them here either.
            return self._solve_streaming(
                options=options,
                keep_solver=keep_solver,
                save_memory=save_memory,
                tmp_dir=tmp_dir,
            )

        if save_memory:
            # The non-streaming path delegates to the HiGHS adapter,
            # which builds a full :class:`HighsLp` via ``passModel``
            # before ``h.run()`` — there is no in-tree hook where the
            # polar-side source could be dropped or the HiGHS allocator
            # round-tripped between matrix emission and solve.  Warn and
            # proceed without releasing rather than hard-failing, so
            # callers that pair the two flags don't get surprised when
            # toggling ``streaming``.
            import warnings

            warnings.warn(
                "Problem.solve(save_memory=True) is honoured only by "
                "the streaming path; ignoring on streaming=False.",
                stacklevel=2,
            )

        # Non-streaming path now delegates to the HiGHS adapter behind
        # :mod:`polar_high.solvers`.  The adapter recomputes the column +
        # LP arrays from ``self``, runs HiGHS via ``passModel``, and
        # returns a :class:`SolverResult`.  We then convert that back
        # into the legacy :class:`Solution` shape so callers see no
        # breaking change.  The adapter stashes the raw numpy arrays as
        # private attributes on the result for zero-copy round-trip; the
        # public dict fields exist for cross-solver consumers.
        #
        # NOTE: the streaming path above is HiGHS-only and intentionally
        # bypasses :mod:`polar_high.solvers`.  See ``_highs.py``'s module
        # docstring for the locked rationale.
        from .solvers._base import SolverStatus
        from .solvers._highs import run as _highs_run
        from .solvers._lp_view import LpView

        _sp_emit_ns, _sp_on_ns = _make_solve_profile_emitter()
        if _sp_on_ns:
            _sp_emit_ns(
                "nonstreaming_enter",
                n_vars=len(self._vars),
                n_cstrs=len(self._cstrs),
            )

        view = LpView.from_problem(self)
        if _sp_on_ns:
            _sp_emit_ns("nonstreaming_lpview_built")
        # Per-call ``options`` wins over what was stored on the Problem.
        opts = options if options is not None else self._solver_options
        result = _highs_run(
            view,
            options=opts,
            keep_solver=keep_solver,
        )
        if _sp_on_ns:
            _sp_emit_ns("nonstreaming_highs_run_done")

        # SolverResult → Solution round-trip.  All of these private
        # attributes are populated unconditionally by ``_highs.run``;
        # cross-solver code paths read the public ``primal`` / ``dual``
        # dicts instead.
        col_value: np.ndarray = result._col_value
        row_dual: np.ndarray = result._row_dual
        col_dual: np.ndarray = result._col_dual
        col_names_out: list[str] = result._col_names
        row_names_out: list[str] = result._row_names
        sol_highs: highspy.Highs | None = result._highs_instance

        # Pre-Phase-2 behaviour: ``Solution.obj`` carried the HiGHS-reported
        # objective regardless of model status.  SolverResult zeros the
        # ``objective`` field for non-optimal solves, so we read the raw
        # HiGHS objective off the private ``_objective_raw`` field for
        # bit-identical round-trip.
        return Solution(
            optimal=result.status == SolverStatus.OPTIMAL,
            obj=result._objective_raw,
            col_value=col_value,
            row_dual=row_dual,
            col_dual=col_dual,
            col_names=col_names_out,
            row_names=row_names_out,
            vars=dict(self._vars),
            highs=sol_highs,
        )

    def build_only(
        self,
        mps_path: str,
        *,
        options: dict | None = None,
    ) -> None:
        """Build the LP into HiGHS, write to ``mps_path``, release everything.

        For callers that want to drive the actual solve out-of-process —
        typically a subprocess HiGHS reading the MPS file in a clean
        address space.  After this call:

        * ``mps_path`` exists and contains the LP in MPS format.
        * polar-side LP source-of-truth has been dropped
          (:meth:`_release_python_lp_inputs`).
        * The live :class:`highspy.Highs` instance has been torn down
          and the glibc allocator trimmed (Linux best-effort).
        * ``self._released`` is True — further calls to :meth:`solve`
          will raise.

        What stays alive: ``self._vars`` and each :class:`Var.frame`,
        so the caller can later construct a :class:`Solution` from
        externally-produced ``col_value`` / ``row_dual`` arrays via
        ``Solution(..., vars=dict(self._vars), ...)``.

        Honoured only on the streaming path; the non-streaming
        (passModel) path is not supported here.  Raises
        ``RuntimeError`` if called on an already-released Problem.

        Parameters
        ----------
        mps_path
            Where to write the MPS file.  Caller owns the file —
            polar-high will not delete it.
        options
            HiGHS solver options to apply during the build (mainly
            ``presolve`` / ``solver`` / ``simplex_scale_strategy`` —
            options that affect what ``writeModel`` serialises).
            ``None`` uses :attr:`_solver_options`.
        """
        if getattr(self, "_released", False):
            raise RuntimeError(
                "Problem.build_only() called on an already-released "
                "Problem.  Construct a fresh Problem.",
            )
        self._solve_streaming(
            options=options,
            keep_solver=False,
            save_memory=True,
            _mps_out_path=mps_path,
            _build_only=True,
        )

    # ------------------------------------------------------------------
    # Canonical CSC store (Stage B1)
    #
    # GLPK-style: build the LP matrix once, then have every consumer
    # walk read-only arrays.  Currently only :meth:`write_mps` reads
    # ``_matrix``; :meth:`_build_lp_arrays`, :meth:`_solve_streaming`,
    # :class:`WarmProblem`, and :class:`LpView` keep their Stage A
    # multiply-at-emit code until B2/B3.
    # ------------------------------------------------------------------
    def canonicalise(self) -> _CanonicalMatrix:
        """Build (or return cached) the canonical CSC matrix + metadata.

        Idempotent: returns the cached ``_matrix`` unless
        ``_canonical_dirty`` is set (which ``add_var`` / ``add_cstr``
        flip).  Side vectors (``_layer2_col_factor`` /
        ``_layer2_row_factor``) are baked into the returned arrays at
        build time per orchestrator decision D8.
        """
        if self._matrix is not None and not self._canonical_dirty:
            return self._matrix
        self._matrix = self._build_canonical_matrix()
        self._canonical_dirty = False
        return self._matrix

    def _build_canonical_matrix(self) -> _CanonicalMatrix:
        """One-shot LP build: walks ``_cstrs`` + ``_obj_terms`` once,
        applies Layer-2 side vectors, global-dedups, and returns a
        canonical CSC ``_CanonicalMatrix``.  See :meth:`canonicalise`.
        """
        # Per-family profile gate — same env var as write_mps so the
        # canonicalise step appears in the same `[write_mps profile]`
        # stream.  Emits the per-family checkpoints that used to live
        # inline in write_mps (D8 moved the family walk here).
        _profile = os.environ.get("POLAR_HIGH_WRITE_MPS_PROFILE") == "1"
        _cm_emit = None  # type: ignore[assignment]
        if _profile:
            try:
                import psutil as _ps_cm

                _cm_proc = _ps_cm.Process()
                _cm_t0 = time.monotonic()

                def _cm_emit(phase: str, **extras) -> None:  # noqa: E306
                    rss = _cm_proc.memory_info().rss / (1024**3)
                    wall = time.monotonic() - _cm_t0
                    extras_str = "\t".join(f"{k}={v}" for k, v in extras.items())
                    sys.stderr.write(
                        f"[write_mps profile]\tphase={phase}"
                        f"\trss_gb={rss:.2f}\twall_s={wall:.2f}"
                        + (f"\t{extras_str}" if extras_str else "")
                        + "\n"
                    )
                    sys.stderr.flush()

                _cm_emit("canonicalise_enter", n_cstrs=len(self._cstrs))
            except ImportError:
                _profile = False

        n_cols = int(self._next_col)

        # Side vectors (BAKE site per D8).  After this method returns,
        # ``_matrix.val`` / ``col_obj`` / ``row_lb`` / ``row_ub`` carry
        # post-Layer-2 coefficients and consumers do NOT re-multiply.
        _rf = self._layer2_row_factor
        _cf = self._layer2_col_factor

        # ---- Pass 1: walk families to build row metadata + LHS triples.
        triple_rows: list[np.ndarray] = []
        triple_cols: list[np.ndarray] = []
        triple_vals: list[np.ndarray] = []
        rows_lb_chunks: list[np.ndarray] = []
        rows_ub_chunks: list[np.ndarray] = []
        sense_chunks: list[np.ndarray] = []
        row_names: list[str] = []
        next_row = 0  # 0-based over constraint rows (no objective row).

        for _fam_idx, (cname, proto, over) in enumerate(self._cstrs):
            expr, sense, rhs = proto.expr, proto.sense, proto.rhs

            if over is None:
                row_count = 1
                row_index = pl.DataFrame({"_rid": [0]})
                axis_cols: list[str] = []
            else:
                row_count = int(over.height)
                axis_cols = list(over.columns)
                row_index = over.with_columns(_rid=pl.int_range(0, over.height, dtype=pl.Int64))

            if _profile:
                _cm_emit(
                    "family_start",
                    family=cname,
                    family_idx=_fam_idx,
                    row_count=row_count,
                    term_count=len(expr.terms),
                )

            base_row = next_row
            next_row += row_count

            # ---- RHS vector (Param / scalar / Var-on-RHS fold).
            rhs_vec = np.zeros(row_count, dtype=np.float64)
            if isinstance(rhs, (int, float)):
                rhs_vec[:] = float(rhs)
            elif isinstance(rhs, Param):
                missing = [d for d in rhs.dims if d not in axis_cols]
                if missing:
                    raise ValueError(
                        f"constraint {cname!r}: rhs Param has dim {missing} not in over={axis_cols}"
                    )
                on = list(rhs.dims)
                # Prefer the prune-down path when the composite RHS Param
                # tracks its atomic constituents via ``_sources`` (length
                # >= 2).  Walking the chain one atomic at a time and
                # semi-joining each atomic to the running accumulator's
                # key projection keeps the intermediate bounded to
                # ``row_count`` rows — avoiding the wide Cartesian-on-shared-
                # dim intermediates that ``Param.__mul__``'s nested inner
                # joins otherwise materialise (e.g. DES
                # ``profile_flow_upper_limit`` RHS = profile_value *
                # process_existing_count * process_availability, where the
                # first inner join on ``d`` alone exploded to many hundred
                # MB of (f, p, d, t)-keyed rows).
                #
                # Single-Param RHS (``_sources is None`` or len <= 1) and
                # anonymous chains (``_sources is None``) fall back to the
                # original merged-lazy + semi-join-prune path verbatim so
                # we don't perturb existing parity.
                sources = rhs._sources if isinstance(rhs._sources, list) else None
                use_prune_down = (
                    on and sources is not None and len(sources) >= 2 and not _prune_down_disabled()
                )
                if use_prune_down:
                    # Start the accumulator from row_index with value=1.0.
                    # Each atomic contributes either as a left-joined value
                    # column (multiplied / divided in) or, for scalar
                    # atomics, as a literal scalar factor.
                    # Seed the accumulator with rhs._value_scalar so the
                    # rebuilt chain honours scalar multiplies on the
                    # composite Param (``Param.__mul__`` / ``__truediv__``
                    # / ``__neg__`` with int/float).  Atomic Params keep
                    # ``_value_scalar=1.0`` so this is the identity for
                    # the common no-scalar chain.
                    acc = row_index.lazy().with_columns(
                        value=pl.lit(float(rhs._value_scalar), dtype=pl.Float64)
                    )
                    for atomic, direction in sources:
                        atomic_on = [d for d in atomic.dims if d in axis_cols]
                        if atomic_on:
                            # Pre-prune the atomic to only rows whose keys
                            # exist in the accumulator's projection.  Each
                            # join is wrapped in _align_enum_join_keys so
                            # Enum dtype mismatches between row_index keys
                            # and atomic keys don't silently drop rows.
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
                            # Scalar atomic — fold the constant into the
                            # running value column directly.
                            scalar_val = float(atomic.frame["value"][0])
                            if direction >= 0:
                                acc = acc.with_columns(value=pl.col("value") * scalar_val)
                            else:
                                acc = acc.with_columns(value=pl.col("value") / scalar_val)
                    if _profile:
                        _cm_emit(
                            "family_rhs_pruned_down",
                            family=cname,
                            family_idx=_fam_idx,
                            n_atomics=len(sources),
                        )
                    _plan = acc.select("_rid", "value")
                    j = _collect_streaming(_plan)
                    if j.height != row_count:
                        raise ValueError(
                            f"constraint {cname!r}: rhs Param chain produced "
                            f"{j.height} rows from row_index (rows={row_count})"
                            " — likely duplicate keys in one of the atomic "
                            f"Params {[a.name for a, _ in sources]!r}."
                        )
                    rhs_vec = (
                        j.sort("_rid")["value"]
                        .fill_null(0.0)
                        .to_numpy()
                        .astype(np.float64, copy=False)
                    )
                elif on:
                    ri_a, rf_a = _align_enum_join_keys(
                        row_index.lazy(),
                        rhs.lazy,
                        on,
                    )
                    keys_lazy = ri_a.select(on).unique()
                    rf_pruned = rf_a.join(keys_lazy, on=on, how="semi")
                    _plan = ri_a.join(rf_pruned, on=on, how="left")
                    try:
                        j = _plan.collect(engine="streaming")
                    except TypeError:
                        j = _plan.collect(streaming=True)
                    if j.height != row_count:
                        raise ValueError(
                            f"constraint {cname!r}: rhs Param has duplicate "
                            f"keys on {on!r} — left join from row_index "
                            f"(rows={row_count}) produced {j.height} rows."
                        )
                    rhs_vec = (
                        j.sort("_rid")["value"]
                        .fill_null(0.0)
                        .to_numpy()
                        .astype(np.float64, copy=False)
                    )
                else:
                    rhs_vec[:] = float(rhs.frame["value"][0])
            else:
                raise TypeError(f"constraint {cname!r}: unsupported rhs type {type(rhs).__name__}")

            if _profile:
                _cm_emit(
                    "family_rhs_evaluated",
                    family=cname,
                    family_idx=_fam_idx,
                )

            # ---- Layer 2 row-factor on RHS (BAKE).
            if _rf is not None and row_count:
                rhs_vec = rhs_vec * _rf[base_row : base_row + row_count]

            if _profile:
                _cm_emit(
                    "family_rhs_l2baked",
                    family=cname,
                    family_idx=_fam_idx,
                )

            # ---- sense → lb/ub vectors.
            if sense == "<=":
                sc = "L"
                lb_vec = np.full(row_count, -np.inf, dtype=np.float64)
                ub_vec = rhs_vec
            elif sense == ">=":
                sc = "G"
                lb_vec = rhs_vec
                ub_vec = np.full(row_count, np.inf, dtype=np.float64)
            elif sense == "==":
                sc = "E"
                lb_vec = rhs_vec
                ub_vec = rhs_vec
            else:
                raise ValueError(
                    f"constraint {cname!r}: sense must be '<=', '>=' or '=='; got {sense!r}"
                )
            rows_lb_chunks.append(lb_vec)
            rows_ub_chunks.append(ub_vec)
            sense_chunks.append(np.full(row_count, ord(sc), dtype=np.uint8))

            if _profile:
                _cm_emit(
                    "family_senses_built",
                    family=cname,
                    family_idx=_fam_idx,
                )

            # ---- row names.
            if over is None:
                row_names.append(cname)
            else:
                row_names.extend(
                    over.select(
                        pl.format(
                            "{}[{}]",
                            pl.lit(cname),
                            pl.concat_str(
                                [pl.col(d).cast(pl.String) for d in axis_cols],
                                separator=",",
                            ),
                        ).alias("__rn")
                    )["__rn"].to_list()
                )

            if _profile:
                _cm_emit(
                    "family_rownames_built",
                    family=cname,
                    family_idx=_fam_idx,
                )

            # ---- LHS term plans (same semi-join + streaming pattern as
            # write_mps' Stage A code).
            row_index_lf = row_index.lazy()
            # Each entry: (kind, plan, on_dims) — on_dims is the shared
            # join keys for "dim" terms, or [] for "scalar" terms.
            term_plans: list[tuple] = []
            for _term_idx, term in enumerate(expr.terms):
                if term.dims:
                    missing = [d for d in term.dims if d not in axis_cols]
                    if missing:
                        raise ValueError(
                            f"constraint {cname!r}: term has open dims "
                            f"{term.dims}, but constraint axes are "
                            f"{axis_cols}; aggregate {missing} via Sum() "
                            f"before adding."
                        )
                    on = [d for d in term.dims if d in axis_cols]
                    # Prefer LHS prune-down when the term carries both a
                    # surviving Var reference and a Param-chain with >= 2
                    # atomics — mirror of the RHS prune-down above.  Same
                    # rationale: avoid the wide ``Var ⋈ P1 ⋈ P2 …``
                    # intermediate that ``term.lazy`` materialises before
                    # the row_index semi-join can prune it.  Single-Param
                    # / no-Param / Sum-collapsed terms (var_source is
                    # None after Sum/Where/Lag) fall through to the
                    # original semi-join path verbatim.
                    _lhs_psrc = term.param_sources if isinstance(term.param_sources, list) else None
                    _use_lhs_prune = (
                        term.var_source is not None
                        and _lhs_psrc is not None
                        and len(_lhs_psrc) >= 2
                        and not _prune_down_disabled()
                    )
                    # Block-COO is a sibling arm for dense-axis non-Sum
                    # Var×Param chains — it key-aligns factors and does the
                    # final multiply in numpy on contiguous value buffers
                    # while staying bit-identical (same factor order, same
                    # row set, same IEEE-double ops).  It is default ON;
                    # POLAR_HIGH_DISABLE_BLOCK_COO=1 is the off switch (see
                    # DECISIONS D5).  It fires ONLY when the Problem declares
                    # dense_axes AND the term matches the suffix contract, so
                    # default-on is inert for callers that don't declare
                    # dense_axes.  Conservatively gated by the classifier;
                    # any shape it can't reproduce exactly returns None →
                    # fall through to prune-down / fallback unchanged.
                    # (Sites 2/3 — streaming / warm — are left on the
                    # prune-down path; a separate task wires them.)
                    _block_spec = None
                    if not _block_coo_disabled():
                        _block_spec = _block_coo_classify(term, axis_cols, on, self._dense_axes)
                    # A deferred map-effect Where (non-None
                    # where_map_frames) introduces extras dims that the
                    # block-COO seed (Var dims only) cannot carry.  The
                    # classifier already declines such terms (``on`` then
                    # includes an extra ∉ var.dims), but guard explicitly:
                    # bake-before-block keeps the path byte-identical and
                    # routes the term to the prune / fallback arms which
                    # handle where_map_frames.  Phase C-3 will teach
                    # block-COO to assemble the map-join itself.
                    if term.where_map_frames is not None:
                        _block_spec = None
                    _use_block_coo = _block_spec is not None
                    # Sum-block-COO (Phase C-3a) is a SIBLING arm for
                    # ``Sum``-wrapped Var×Param chains (var_source cleared by
                    # Sum, but a SumBlockMeta recipe captured).  It rebuilds
                    # the unreduced product from the recipe on the pre-sorted
                    # Var grid, bakes the map-effect Where, then reduces over
                    # ``over`` to ``keep`` — without polars' join + group_by.
                    # Same off-switch as the non-Sum arm; fires only when the
                    # Problem declares dense_axes AND the recipe matches the
                    # suffix contract.  Conservatively gated by
                    # ``_sum_block_coo_classify``; anything it can't
                    # reproduce bit-equivalently returns None → the term
                    # reads its reduced ``term.lazy`` exactly as today.
                    # Mutually exclusive with the non-Sum arm (a Sum term has
                    # var_source=None so _block_spec is already None here).
                    _sum_block_spec = None
                    if not _use_block_coo and not _block_coo_disabled():
                        _sum_block_spec = _sum_block_coo_classify(
                            term, axis_cols, on, self._dense_axes
                        )
                    _use_sum_block_coo = _sum_block_spec is not None
                    if _use_block_coo:
                        # Verify the client's dense_axes sort contract on
                        # the Var BEFORE building — a mis-ordered frame
                        # would silently corrupt the dense-suffix slice, so
                        # raise a clear, actionable error instead.
                        _verify_dense_sorted(
                            term.var_source.frame,
                            _block_spec["non_dense_dims"],
                            _block_spec["dense_dims"],
                            getattr(term.var_source, "name", None),
                        )
                        _t_blk0 = time.monotonic()
                        # Block-COO collects eagerly inside the helper; wrap
                        # the result lazy so the shared per-term collect loop
                        # (_collect_one) consumes it like every other plan.
                        plan = _build_block_coo_plan(
                            row_index_lf,
                            axis_cols,
                            term.var_source,
                            _lhs_psrc,
                            on,
                            term.coef_scalar,
                            term.where_frames,
                            _block_spec,
                        )
                        if os.environ.get("POLAR_HIGH_BLOCK_COO_PROFILE") == "1":
                            _blk_wall = time.monotonic() - _t_blk0
                            _n_rows = int(plan.height)
                            _dense = _block_spec["dense_dims"]
                            _nb = _block_spec["dense_card"]
                            _avg = (_n_rows / _nb) if _nb else 0.0
                            sys.stderr.write(
                                f"[block_coo profile]\tphase=block_coo_term"
                                f"\tfamily={cname}\tfamily_idx={_fam_idx}"
                                f"\tterm_idx={_term_idx}"
                                f"\tdense_dims={','.join(_dense)}"
                                f"\tn_blocks={_nb}"
                                f"\tavg_block_size={_avg:.2f}"
                                f"\twall_s={_blk_wall:.4f}\n"
                            )
                            sys.stderr.flush()
                        plan = plan.lazy()
                    elif _use_sum_block_coo:
                        # Verify the dense_axes sort contract on the RECIPE's
                        # Var (the seed block-COO slices) before building.
                        _sm = term.sum_block_meta
                        _verify_dense_sorted(
                            _sm.var_source.frame,
                            _sum_block_spec["non_dense_dims"],
                            _sum_block_spec["dense_dims"],
                            getattr(_sm.var_source, "name", None),
                        )
                        _t_blk0 = time.monotonic()
                        try:
                            plan = _build_sum_block_coo_plan(
                                row_index_lf,
                                axis_cols,
                                _sm,
                                on,
                                _sum_block_spec,
                            )
                            _sum_block_fired = True
                        except _SumBlockCooFallback:
                            # Shape the rebuild can't reduce bit-equivalently
                            # ⇒ use the reduced ``term.lazy`` path verbatim
                            # (byte-identical to the block-COO-off run).
                            _sum_block_fired = False
                            plan = None
                        if _sum_block_fired:
                            if os.environ.get("POLAR_HIGH_BLOCK_COO_PROFILE") == "1":
                                _blk_wall = time.monotonic() - _t_blk0
                                _n_rows = int(plan.height)
                                _dense = _sum_block_spec["dense_dims"]
                                _nb = _sum_block_spec["dense_card"]
                                _avg = (_n_rows / _nb) if _nb else 0.0
                                sys.stderr.write(
                                    f"[block_coo profile]\tphase=block_coo_term"
                                    f"\tkind=sum\tfamily={cname}"
                                    f"\tfamily_idx={_fam_idx}"
                                    f"\tterm_idx={_term_idx}"
                                    f"\tdense_dims={','.join(_dense)}"
                                    f"\tn_blocks={_nb}"
                                    f"\tavg_block_size={_avg:.2f}"
                                    f"\twall_s={_blk_wall:.4f}\n"
                                )
                                sys.stderr.flush()
                            plan = plan.lazy()
                        else:
                            # Reduced-``term.lazy`` fallback (identical to the
                            # final else arm): bake any deferred filters, then
                            # the row_index semi-join + inner-join.
                            term_lazy_filtered = _apply_where_frames(
                                term.lazy, term.dims, term.where_frames
                            )
                            term_lazy_filtered, _ = _apply_where_map_frames(
                                term_lazy_filtered,
                                term.dims,
                                term.where_map_frames,
                            )
                            rl_a, tl_a = _align_enum_join_keys(row_index_lf, term_lazy_filtered, on)
                            keys_lazy = rl_a.select(on).unique()
                            tl_pruned = tl_a.join(keys_lazy, on=on, how="semi")
                            plan = rl_a.join(tl_pruned, on=on, how="inner").select(
                                "_rid", "col_id", "coef"
                            )
                    elif _use_lhs_prune:
                        plan = _build_lhs_pruned_plan(
                            row_index_lf,
                            axis_cols,
                            term.var_source,
                            _lhs_psrc,
                            on,
                            coef_scalar=term.coef_scalar,
                            where_frames=term.where_frames,
                            where_map_frames=term.where_map_frames,
                        ).select("_rid", "col_id", "coef")
                        if _profile:
                            _cm_emit(
                                "family_term_pruned_down",
                                family=cname,
                                family_idx=_fam_idx,
                                term_idx=_term_idx,
                                n_atomics=len(_lhs_psrc),
                            )
                    else:
                        # Bake any deferred Where filters before the
                        # semi-join — fallback path must apply them
                        # since prune-down isn't firing here.  Pure-filter
                        # first, then map-effect (dim-extending) so the
                        # plan carries the extras columns ``on`` may need.
                        term_lazy_filtered = _apply_where_frames(
                            term.lazy, term.dims, term.where_frames
                        )
                        term_lazy_filtered, _ = _apply_where_map_frames(
                            term_lazy_filtered, term.dims, term.where_map_frames
                        )
                        rl_a, tl_a = _align_enum_join_keys(row_index_lf, term_lazy_filtered, on)
                        keys_lazy = rl_a.select(on).unique()
                        tl_pruned = tl_a.join(keys_lazy, on=on, how="semi")
                        plan = rl_a.join(tl_pruned, on=on, how="inner").select(
                            "_rid", "col_id", "coef"
                        )
                    term_plans.append(("dim", plan, list(on)))
                else:
                    term_plans.append(("scalar", term.lazy.select("col_id", "coef"), []))

            if _profile:
                _cm_emit(
                    "family_term_plans_built",
                    family=cname,
                    family_idx=_fam_idx,
                    lhs_term_count=len(term_plans),
                )
                # Optional .explain() dump per family-term (cheap I/O,
                # gated on env var).  Skipped silently on any failure
                # so it can't introduce new failure modes.
                try:
                    import os as _os_pl

                    _plans_dir = "/tmp/polar_high_canonicalise_plans"
                    _os_pl.makedirs(_plans_dir, exist_ok=True)
                    _safe_cname = "".join(
                        c if c.isalnum() or c in "._-" else "_" for c in str(cname)
                    )
                    for _i, (_kind, _p, _on) in enumerate(term_plans):
                        _fname = f"{_plans_dir}/{_fam_idx:04d}_{_safe_cname}_term{_i}.txt"
                        try:
                            _txt = _p.explain(optimized=True)
                        except Exception:
                            try:
                                _txt = _p.explain(optimized=False)
                            except Exception:
                                _txt = "<explain unavailable>"
                        try:
                            with open(_fname, "w") as _fh:
                                _fh.write(
                                    f"# family={cname} family_idx={_fam_idx}\n"
                                    f"# term_idx={_i} term_kind={_kind} "
                                    f"on={_on}\n"
                                )
                                _fh.write(_txt)
                        except Exception:
                            pass
                except Exception:
                    pass

            if not term_plans:
                if _profile:
                    _cm_emit(
                        "family_collected",
                        family=cname,
                        family_idx=_fam_idx,
                        fam_nnz=0,
                        lhs_term_count=0,
                    )
                continue

            # Collect one-at-a-time to bound peak memory; same fallback
            # chain (streaming engine → streaming kwarg → plain collect)
            # write_mps' Stage A code uses.
            def _collect_one(p: pl.LazyFrame) -> pl.DataFrame:
                try:
                    return p.collect(engine="streaming")
                except TypeError:
                    try:
                        return p.collect(streaming=True)
                    except TypeError:
                        return p.collect()
                except Exception:
                    return p.collect()

            # Explicit loop (was a list comprehension) so we can emit
            # per-term collect_start/collected checkpoints without
            # changing collection order or semantics.
            collected: list[pl.DataFrame] = []
            for _i, (_kind, _p, _on) in enumerate(term_plans):
                if _profile:
                    _cm_emit(
                        "family_term_collect_start",
                        family=cname,
                        family_idx=_fam_idx,
                        term_idx=_i,
                        term_kind=_kind,
                        on=_on,
                    )
                _j = _collect_one(_p)
                collected.append(_j)
                if _profile:
                    _cm_emit(
                        "family_term_collected",
                        family=cname,
                        family_idx=_fam_idx,
                        term_idx=_i,
                        term_kind=_kind,
                        term_rows=int(_j.height),
                    )
            fam_nnz = 0
            for (kind, _, _), j in zip(term_plans, collected):
                if kind == "dim":
                    if j.height == 0:
                        continue
                    rids_local = j["_rid"].to_numpy().astype(np.int64, copy=False)
                    abs_rows = (base_row + rids_local).astype(np.int64, copy=False)
                    cids = j["col_id"].to_numpy().astype(np.int64, copy=False)
                    vals = j["coef"].to_numpy().astype(np.float64, copy=False)
                    # BAKE side vectors into LHS values.
                    if _rf is not None:
                        vals = vals * _rf[abs_rows]
                    if _cf is not None:
                        vals = vals * _cf[cids]
                    triple_rows.append(abs_rows)
                    triple_cols.append(cids)
                    triple_vals.append(vals)
                    fam_nnz += abs_rows.size
                else:
                    cids = j["col_id"].to_numpy().astype(np.int64, copy=False)
                    vals = j["coef"].to_numpy().astype(np.float64, copy=False)
                    if cids.size == 0:
                        continue
                    rs = np.repeat(
                        np.arange(base_row, base_row + row_count, dtype=np.int64),
                        cids.size,
                    )
                    tiled_cols = np.tile(cids, row_count)
                    tiled_vals = np.tile(vals, row_count)
                    if _rf is not None:
                        tiled_vals = tiled_vals * _rf[rs]
                    if _cf is not None:
                        tiled_vals = tiled_vals * _cf[tiled_cols]
                    triple_rows.append(rs)
                    triple_cols.append(tiled_cols)
                    triple_vals.append(tiled_vals)
                    fam_nnz += rs.size
            del collected

            if _profile:
                _cm_emit(
                    "family_lhs_scattered",
                    family=cname,
                    family_idx=_fam_idx,
                )
                _cm_emit(
                    "family_collected",
                    family=cname,
                    family_idx=_fam_idx,
                    fam_nnz=fam_nnz,
                    lhs_term_count=len(term_plans),
                )

        n_constraint_rows = next_row

        # ---- Pass 2: global dedup-sum on (col, row) keys (D9).  Same
        # pattern as ``_build_lp_arrays`` post-Stage-A: polars group_by.
        if triple_rows:
            tr = np.concatenate(triple_rows)
            tc = np.concatenate(triple_cols)
            tv = np.concatenate(triple_vals)
            del triple_rows, triple_cols, triple_vals
            dedup = (
                pl.DataFrame({"r": tr, "c": tc, "v": tv})
                .group_by(["r", "c"])
                .agg(pl.col("v").sum())
            )
            tr = dedup["r"].to_numpy().astype(np.int64, copy=False)
            tc = dedup["c"].to_numpy().astype(np.int64, copy=False)
            tv = dedup["v"].to_numpy().astype(np.float64, copy=False)
            del dedup
        else:
            tr = np.zeros(0, dtype=np.int64)
            tc = np.zeros(0, dtype=np.int64)
            tv = np.zeros(0, dtype=np.float64)

        if _profile:
            _cm_emit("global_deduped", nnz=int(tr.size))

        # ---- Pass 3: CSC sort + col_ptr.  Mirrors ``_build_lp_arrays``
        # at engine.py:~2538 (idx_dtype choice and lexsort + np.add.at).
        nnz = int(tr.size)
        idx_dtype = np.int32 if nnz < (1 << 31) else np.int64
        if nnz:
            order = np.lexsort((tr, tc))  # primary: col, secondary: row
            sorted_r = tr[order].astype(idx_dtype)
            sorted_c = tc[order].astype(idx_dtype)
            sorted_v = tv[order].astype(np.float64)
        else:
            sorted_r = np.zeros(0, dtype=idx_dtype)
            sorted_c = np.zeros(0, dtype=idx_dtype)
            sorted_v = np.zeros(0, dtype=np.float64)
        del tr, tc, tv

        col_ptr = np.zeros(n_cols + 1, dtype=idx_dtype)
        if sorted_c.size:
            np.add.at(col_ptr[1:], sorted_c, 1)
        col_ptr = np.cumsum(col_ptr).astype(idx_dtype)

        if _profile:
            _cm_emit("csc_built", nnz=nnz)

        # ---- Pass 4: objective.  BAKE col_factor (no row_factor —
        # the cost row is NOT in the row_factor vector per GLPK
        # convention; orchestrator principle #4).
        col_obj = np.zeros(n_cols, dtype=np.float64)
        for t in self._obj_terms:
            # Bake any deferred Where pushdown frames before the collect.
            # Objective terms flow through Sum(over=None), which already
            # bakes both slots, so this is a defensive no-op on the
            # production path — but it keeps the bake-before-consume
            # invariant uniform across every ``term.lazy`` consumer.
            _obj_lazy = _apply_where_frames(t.lazy, t.dims, t.where_frames)
            _obj_lazy, _ = _apply_where_map_frames(_obj_lazy, t.dims, t.where_map_frames)
            f = _obj_lazy.collect()
            if f.height == 0:
                del f
                continue
            cids = f["col_id"].to_numpy().astype(np.int64, copy=False)
            vals = f["coef"].to_numpy().astype(np.float64, copy=False)
            if _cf is not None:
                vals = vals * _cf[cids]
            # np.add.at handles duplicate col_ids across obj terms.
            np.add.at(col_obj, cids, vals)
            del f

        # ---- Pass 5: col metadata.  ``Var.lower``/``upper`` are
        # already mutated in place by ``apply_layer2``; no further
        # scaling needed here (BOUNDS section gets these verbatim).
        col_lb = np.zeros(n_cols, dtype=np.float64)
        col_ub = np.full(n_cols, np.inf, dtype=np.float64)
        col_int = np.zeros(n_cols, dtype=np.int8)
        # ---- Pass 6: col names.  Used by write_mps + (future)
        # write_lp / diagnostic emitters.  Not gated on emit_names —
        # the canonical store always carries them; write_mps can
        # override with generic R/C names at emit time.  Merged with
        # the bounds/integrality scatter above so each var.frame's
        # col_id materialisation runs once instead of twice.
        col_names: list[str] = [""] * n_cols
        for v in self._vars.values():
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
                                [pl.col(d).cast(pl.String) for d in v.dims],
                                separator=",",
                            ),
                        ).alias("__name")
                    )
                )["__name"].to_list()
                for cid, nm in zip(ids.tolist(), tagged):
                    col_names[cid] = nm
            else:
                col_names[int(ids[0])] = v.name

        # Concat row metadata into final arrays.
        if rows_lb_chunks:
            row_lb = np.concatenate(rows_lb_chunks)
            row_ub = np.concatenate(rows_ub_chunks)
            sense_char = np.concatenate(sense_chunks)
        else:
            row_lb = np.zeros(0, dtype=np.float64)
            row_ub = np.zeros(0, dtype=np.float64)
            sense_char = np.zeros(0, dtype=np.uint8)

        # Small-coefficient cutoff (0.0 ⇒ OFF).  Floors the CSC matrix
        # coefficients and the row bounds (RHS) so this canonical matrix —
        # consumed by the non-streaming ``passModel`` path AND by
        # :class:`WarmProblem` — carries the same floored values the
        # streaming path produces.  Replaces |value| < threshold with 0.0;
        # ±inf row-bound sentinels survive (abs(inf) < thr is False).
        _coef_zero_thr = float(getattr(self, "coef_zero_threshold", 0.0) or 0.0)
        if _coef_zero_thr > 0.0:
            sorted_v = _floor_small_coefs(sorted_v, _coef_zero_thr)
            row_lb = _floor_small_coefs(row_lb, _coef_zero_thr)
            row_ub = _floor_small_coefs(row_ub, _coef_zero_thr)

        if _profile:
            _cm_emit(
                "canonicalise_exit",
                n_rows=n_constraint_rows,
                n_cols=n_cols,
                nnz=nnz,
            )

        return _CanonicalMatrix(
            n_rows=n_constraint_rows,
            n_cols=n_cols,
            nnz=nnz,
            col_ptr=col_ptr,
            row_idx=sorted_r,
            val=sorted_v,
            row_lb=row_lb,
            row_ub=row_ub,
            sense_char=sense_char,
            col_obj=col_obj,
            col_lb=col_lb,
            col_ub=col_ub,
            col_int=col_int,
            col_names=col_names,
            row_names=row_names,
        )

    # ------------------------------------------------------------------
    # Direct polars → MPS writer
    #
    # Independent of the HiGHS-backed ``build_only`` path: walks the
    # polar-side LP source-of-truth (``_vars``, ``_cstrs``, ``_obj_terms``)
    # and streams an MPS file out without ever instantiating
    # :class:`highspy.Highs` or :class:`LpView`.  Intended for the very
    # large LPs where HiGHS' own ``writeModel`` serialiser hits its memory
    # ceiling (see ``specs/`` for the motivating profile).
    # ------------------------------------------------------------------
    def write_mps(
        self,
        path: str | os.PathLike,
        *,
        free_format: bool = True,
        column_order_strict: bool = True,
        emit_names: bool = True,
        release: bool = False,
        name: str = "POLAR_HIGH",
    ) -> None:
        """Stream the LP/MIP to ``path`` as a free-format MPS file.

        This is an in-house writer that consumes the polar-side LP
        source-of-truth (``_vars``, ``_cstrs``, ``_obj_terms``) directly
        and emits MPS without ever instantiating :class:`highspy.Highs`
        or building an :class:`LpView`.  The motivation is peak-memory:
        HiGHS' own ``writeModel`` serialiser materialises ~20× more
        transient state than the in-process LP itself, which OOMs on
        very large LPs (~10 M rows × 5 M cols × 20 M nonzeros).  This
        writer targets ~2-3 GB peak on that same LP.

        Parameters
        ----------
        path
            Output MPS file path.  Caller owns the file; this writer
            does not delete on failure.
        free_format
            Reserved for future use.  Currently always free-format MPS
            (the modern default — HiGHS, Gurobi, CPLEX, Xpress all
            accept it).  Fixed 8-char-column MPS is not yet implemented.
        column_order_strict
            When ``True`` (default), all nonzeros for a single column
            are emitted contiguously in the COLUMNS section, sorted by
            ``(col_id, row_id)``.  This is the spec-compliant ordering
            every MPS reader accepts.  ``False`` is currently not
            implemented — see :ref:`Open follow-ups` in the module docs.
        emit_names
            When ``True`` (default), row and column names come from the
            variable family name + dim tuple and the constraint family
            name + dim tuple (matching the format used by
            :meth:`Problem.solve` for ``Solution.col_names`` /
            ``Solution.row_names``).  When ``False``, generic
            ``C0000001`` / ``R0000001`` names are emitted instead —
            saves ~30-50% file size at the cost of losing the
            dim-tuple → column mapping in the MPS itself.  Callers
            using ``emit_names=False`` must use index-based solution
            readback (column 0 in the MPS is column 0 in the polars
            ``Var.frame`` ordering, etc.).  The objective row name is
            always ``cost`` regardless — it's a reserved unique name.
        release
            When ``True``, calls :meth:`_release_python_lp_inputs`
            after the write completes.  Mirrors the
            ``solve(save_memory=True)`` semantics — drops the polars
            LazyFrames, constraint family list, and rhs Param frames
            while keeping ``self._vars`` (so solutions read back from an
            external solver can still be mapped to user-space dim
            tuples via the surviving ``Var.frame["col_id"]`` columns).
            After release the :class:`Problem` is no longer solvable —
            :meth:`solve` will raise.
        name
            MPS ``NAME`` header; default ``"POLAR_HIGH"``.

        Raises
        ------
        RuntimeError
            If called on a :class:`Problem` whose source has already
            been released (by an earlier ``solve(save_memory=True)``,
            ``build_only(...)``, or ``write_mps(..., release=True)``).
        ValueError
            If any coefficient is NaN or infinite (silent filtering
            would hide model bugs).  The error message identifies at
            least one offending ``(col_id, row_id)`` pair.
        OSError
            File I/O failures (passed through from ``open()``).

        Notes
        -----
        - The MPS format has no portable encoding for an objective
          constant offset.  If ``self._obj_offset != 0.0`` this method
          emits a :class:`UserWarning` and writes the LP without the
          offset.  Solutions from any downstream solver will then be
          off by ``self._obj_offset`` — the caller must add it back
          manually if needed.
        - Integer columns are bracketed with ``MARKER 'INTORG'`` /
          ``MARKER 'INTEND'`` in the COLUMNS section, per MPS
          convention.
        - The writer is single-pass per family and uses one global
          polars sort to put the COLUMNS section into column-major
          order.  Streaming-engine fallback to disk activates
          automatically for triple counts that don't fit in RAM.
        """
        import warnings

        if getattr(self, "_released", False):
            raise RuntimeError(
                "Problem.write_mps() called on an already-released "
                "Problem.  Construct a fresh Problem.",
            )

        # ------------------------------------------------------------------
        # Optional memory-profiling instrumentation.
        #
        # Activated by ``POLAR_HIGH_WRITE_MPS_PROFILE=1``.  When inactive
        # (the default) this entire block adds one env-var read and
        # leaves ``profile`` False — no psutil import, no time.monotonic
        # calls inside the hot loop, no stderr output.  Call sites below
        # are guarded by ``if profile:`` so the closure is never invoked
        # in the off-path.
        # ------------------------------------------------------------------
        profile = os.environ.get("POLAR_HIGH_WRITE_MPS_PROFILE") == "1"
        _emit = None  # type: ignore[assignment]
        if profile:
            try:
                import time as _time

                import psutil as _psutil
            except ImportError:
                sys.stderr.write(
                    "[write_mps profile] psutil not installed — "
                    "profiling disabled (install psutil to enable).\n"
                )
                sys.stderr.flush()
                profile = False
            else:
                _proc = _psutil.Process()
                _t0 = _time.monotonic()
                _state = {"prev_rss_gb": _proc.memory_info().rss / (1024**3)}

                def _emit(  # noqa: E306 — closure local to write_mps
                    phase: str,
                    *,
                    family: str = "-",
                    family_idx: object = "-",
                    **extra: object,
                ) -> None:
                    rss_gb = _proc.memory_info().rss / (1024**3)
                    delta_gb = rss_gb - _state["prev_rss_gb"]
                    _state["prev_rss_gb"] = rss_gb
                    wall_s = _time.monotonic() - _t0
                    parts = [
                        "[write_mps profile]",
                        f"phase={phase}",
                        f"family={family}",
                        f"family_idx={family_idx}",
                        f"rss_gb={rss_gb:.2f}",
                        f"delta_gb={delta_gb:+.2f}",
                        f"wall_s={wall_s:.2f}",
                    ]
                    for k, v in extra.items():
                        parts.append(f"{k}={v}")
                    sys.stderr.write("\t".join(parts) + "\n")
                    sys.stderr.flush()

                # Reset wall clock & rss baseline so the enter checkpoint
                # reports wall_s=0.0 and delta_gb=+0.00.
                _state["prev_rss_gb"] = _proc.memory_info().rss / (1024**3)
                _t0 = _time.monotonic()
                _emit("enter")

        if not column_order_strict:
            # The user-doc kwarg is kept on the signature so a future
            # implementation doesn't break callers; for now strict order
            # is the only supported mode (HiGHS' own writer also emits
            # strict order, so portability matches the existing path).
            raise NotImplementedError(
                "write_mps(column_order_strict=False) is not implemented; "
                "use the default strict ordering.",
            )

        if self._obj_offset:
            warnings.warn(
                f"Problem.write_mps: dropping non-zero objective offset "
                f"{self._obj_offset!r} — MPS has no portable encoding for "
                f"the offset.  Solutions from the downstream solver will be "
                f"off by this amount; add it back manually if needed.",
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # Stage B1 — read from the canonical CSC matrix.  Per D8, side
        # vectors are already baked into ``m.val`` / ``m.col_obj`` /
        # ``m.row_lb`` / ``m.row_ub`` so this method does NOT multiply
        # by ``_layer2_*_factor`` itself.
        # ------------------------------------------------------------------
        if profile:
            _emit("canonicalise_start")
        m = self.canonicalise()
        if profile:
            _emit("canonicalise_done", nnz=m.nnz, n_rows=m.n_rows, n_cols=m.n_cols)

        n_cols = m.n_cols
        n_constraint_rows = m.n_rows

        # Hard-error on NaN / inf coefficients (silent filtering would
        # hide real model bugs; the MPS spec has no representation).
        # Checks both the matrix values and the objective row — both
        # are written to the COLUMNS section.  (RHS rows skip non-finite
        # entries below, matching the prior emit behaviour where
        # ``inf`` rhs on an L/G row collapses to no entry.)
        if m.nnz:
            bad_mask = ~np.isfinite(m.val)
            if bad_mask.any():
                bad_k = int(np.nonzero(bad_mask)[0][0])
                raise ValueError(
                    f"Problem.write_mps: {int(bad_mask.sum())} matrix "
                    f"coefficient(s) are NaN or infinite — refusing to "
                    f"write a corrupt MPS. First offender (row, col, coef): "
                    f"({int(m.row_idx[bad_k])}, ?, {float(m.val[bad_k])})"
                )
        if n_cols:
            obj_bad = ~np.isfinite(m.col_obj)
            if obj_bad.any():
                bad_j = int(np.nonzero(obj_bad)[0][0])
                raise ValueError(
                    f"Problem.write_mps: {int(obj_bad.sum())} objective "
                    f"coefficient(s) are NaN or infinite — refusing to "
                    f"write a corrupt MPS. First offender (col, coef): "
                    f"({bad_j}, {float(m.col_obj[bad_j])})"
                )

        # ---- Names: row 0 is "cost", indices 1.. are constraint rows.
        # MPS row index space (used in RHS emit below) keeps the cost
        # at absolute row 0, so we prepend "cost" to the constraint
        # row_names from the canonical matrix.
        if emit_names:
            row_names: list[str] = ["cost"] + list(m.row_names)
            col_names: list[str] = list(m.col_names)
        else:
            row_names = ["cost"] + [f"R{i + 2:07d}" for i in range(n_constraint_rows)]
            col_names = [f"C{j + 1:07d}" for j in range(n_cols)]
            # Note: cost row externally appears as "cost"; constraint
            # rows are R0000002, R0000003, ... so the generic numbering
            # gives 1-indexed sequential row ids that match how a user
            # would count rows in the file.

        # ---- Integer-col set from m.col_int (1 bit per column).
        integer_cols: set[int] = set(int(c) for c in np.nonzero(m.col_int)[0].tolist())

        # Sense characters per constraint row, decoded from m.sense_char.
        sense_chars = m.sense_char.tobytes().decode("ascii") if n_constraint_rows else ""

        # MPS coefficient formatting: %.17g preserves round-trip for
        # float64 across solvers.  Names emit verbatim — the caller is
        # responsible for whitespace-free names.
        def _fmt(v: float) -> str:
            return f"{float(v):.17g}"

        # Open in text mode with utf-8; a moderately large buffer cuts
        # syscall overhead on the very-large LP target.
        with open(path, "w", encoding="utf-8", buffering=1 << 20) as f:
            # NAME header.
            f.write(f"NAME          {name}\n")

            # OBJSENSE.
            f.write("OBJSENSE\n")
            f.write("    MAX\n" if self._obj_sense == "max" else "    MIN\n")

            # ROWS section.
            f.write("ROWS\n")
            f.write(" N  cost\n")
            for rid in range(n_constraint_rows):
                f.write(f" {sense_chars[rid]}  {row_names[rid + 1]}\n")

            # COLUMNS section — walk CSC column-by-column.  For each
            # column j, emit the objective entry (if present) first then
            # the constraint coefficients in row order.
            f.write("COLUMNS\n")
            in_integer = False
            int_marker_id = 0
            col_ptr = m.col_ptr
            row_idx = m.row_idx
            val = m.val
            col_obj = m.col_obj
            for j in range(n_cols):
                start = int(col_ptr[j])
                end = int(col_ptr[j + 1])
                obj_v = float(col_obj[j])
                obj_nz = obj_v != 0.0 and math.isfinite(obj_v)
                if start == end and not obj_nz:
                    continue
                # Integer-marker flip at column boundary.
                is_int = j in integer_cols
                if is_int and not in_integer:
                    int_marker_id += 1
                    f.write("    MARKER                 'MARKER'                 'INTORG'\n")
                    in_integer = True
                elif (not is_int) and in_integer:
                    f.write("    MARKER                 'MARKER'                 'INTEND'\n")
                    in_integer = False
                cname = col_names[j]
                if obj_nz:
                    f.write(f"    {cname}  cost  {_fmt(obj_v)}\n")
                # Matrix entries — row_idx is 0-based over constraint
                # rows; MPS row name is row_names[rid + 1].
                for k in range(start, end):
                    r = int(row_idx[k])
                    f.write(f"    {cname}  {row_names[r + 1]}  {_fmt(val[k])}\n")
            if in_integer:
                f.write("    MARKER                 'MARKER'                 'INTEND'\n")
                in_integer = False

            if profile:
                _emit("columns_emitted")

            # RHS section — one entry per constraint row that has a
            # finite non-zero rhs.  For L sense the rhs lives in ub;
            # for G in lb; for E both are equal so either works.
            f.write("RHS\n")
            if n_constraint_rows:
                # Pick the finite side as rhs.  np.where short-circuits
                # the inf masking on the two halves.
                lb = m.row_lb
                ub = m.row_ub
                lb_fin = np.isfinite(lb)
                # Prefer ub for L (lb=-inf), lb for G (ub=+inf), and lb
                # for E (lb==ub).  Equivalent: pick lb if finite, else ub.
                rhs_arr = np.where(lb_fin, lb, ub)
                nz = np.nonzero(np.isfinite(rhs_arr) & (rhs_arr != 0.0))[0]
                for rid in nz.tolist():
                    f.write(f"    rhs  {row_names[rid + 1]}  {_fmt(rhs_arr[rid])}\n")

            if profile:
                _emit("rhs_emitted")

            # BOUNDS section — per-variable-family scalar bounds.  Emit
            # only when the column deviates from the MPS default
            # ``[0, +inf]``.  Walks ``self._vars`` (not ``m.col_lb`` /
            # ``m.col_ub``) to preserve the per-family bound-shape
            # classification — every column in a family shares the
            # same lo/hi, so per-family is the natural grouping.  The
            # values come from ``Var.lower`` / ``Var.upper`` which
            # ``apply_layer2`` mutates in place; ``m.col_lb`` /
            # ``m.col_ub`` are built from those same values (see
            # _build_canonical_matrix pass 5), so the BOUNDS section is
            # bit-for-bit identical to the pre-B1 output.
            f.write("BOUNDS\n")
            for v in self._vars.values():
                lo = float(v.lower)
                hi = float(v.upper)
                if math.isfinite(lo) and lo == 0.0 and math.isinf(hi) and hi > 0:
                    continue
                ids = v.frame["col_id"].to_numpy()
                if math.isinf(lo) and lo < 0 and math.isinf(hi) and hi > 0:
                    for cid in ids.tolist():
                        f.write(f" FR bnd  {col_names[int(cid)]}\n")
                elif math.isinf(lo) and lo < 0 and math.isfinite(hi):
                    for cid in ids.tolist():
                        nm = col_names[int(cid)]
                        f.write(f" MI bnd  {nm}\n")
                        f.write(f" UP bnd  {nm}  {_fmt(hi)}\n")
                elif math.isfinite(lo) and math.isinf(hi) and hi > 0:
                    if lo != 0.0:
                        for cid in ids.tolist():
                            f.write(f" LO bnd  {col_names[int(cid)]}  {_fmt(lo)}\n")
                elif math.isfinite(lo) and math.isfinite(hi):
                    for cid in ids.tolist():
                        nm = col_names[int(cid)]
                        f.write(f" LO bnd  {nm}  {_fmt(lo)}\n")
                        f.write(f" UP bnd  {nm}  {_fmt(hi)}\n")
                else:
                    for cid in ids.tolist():
                        nm = col_names[int(cid)]
                        if math.isfinite(lo):
                            f.write(f" LO bnd  {nm}  {_fmt(lo)}\n")
                        if math.isfinite(hi):
                            f.write(f" UP bnd  {nm}  {_fmt(hi)}\n")

            f.write("ENDATA\n")

        if profile:
            _emit("bounds_emitted")

        if release:
            self._release_python_lp_inputs()

        if profile:
            _emit("exit")

    # ------------------------------------------------------------------
    # Shared non-streaming LP-array builder
    #
    # Historically also fed :meth:`peek_lp_ranges` (removed in 1.3.0;
    # callers now read ``Solution.streamed_lp_ranges`` populated during
    # the streaming solve).  Retained for the non-streaming
    # ``solve(streaming=False)`` benchmark path through
    # :mod:`polar_high.solvers`.
    def _build_lp_arrays(
        self,
    ) -> tuple[
        np.ndarray,  # col_lb_h
        np.ndarray,  # col_ub_h
        np.ndarray,  # row_lb_h
        np.ndarray,  # row_ub_h
        np.ndarray,  # sorted_v (matrix values, CSC order)
        np.ndarray,  # sorted_r (row indices, CSC order)
        np.ndarray,  # starts (CSC column starts)
        list[str],  # row_names
        int,  # n_rows
    ]:
        """Build the per-row + matrix arrays for the non-streaming path.

        Stage B2: this function now delegates to :meth:`canonicalise` for
        the family-walk + dedup + CSC build (which bakes in Layer 2 side
        vectors per D8).  The pre-B2 body — per-family LHS collect, RHS
        Param resolve, Stage A multiply-at-emit, global dedup + sort —
        has been moved into :meth:`_build_canonical_matrix`.

        Returns
        -------
        tuple of numpy arrays + list[str] + int
            ``(col_lb_h, col_ub_h, row_lb_h, row_ub_h, sorted_v, sorted_r,
            starts, row_names, n_rows)``.  The bound arrays use HiGHS'
            ``kHighsInf`` sentinel in place of ``np.inf`` so they're
            ready for ``HighsLp.row_lower_`` / ``row_upper_``.
        """
        m = self.canonicalise()

        # The canonical matrix stores ±inf for portability across solver
        # adapters; HiGHS specifically wants ``kHighsInf``, so substitute
        # at the boundary.
        inf = highspy.kHighsInf
        col_lb_h = np.where(m.col_lb == -np.inf, -inf, m.col_lb).astype(np.float64, copy=False)
        col_ub_h = np.where(m.col_ub == np.inf, inf, m.col_ub).astype(np.float64, copy=False)
        row_lb_h = np.where(m.row_lb == -np.inf, -inf, m.row_lb).astype(np.float64, copy=False)
        row_ub_h = np.where(m.row_ub == np.inf, inf, m.row_ub).astype(np.float64, copy=False)

        return (
            col_lb_h,
            col_ub_h,
            row_lb_h,
            row_ub_h,
            m.val,
            m.row_idx,
            m.col_ptr,
            m.row_names,
            m.n_rows,
        )

    # ------------------------------------------------------------------
    # Streaming variant: opt-in alternative to the passModel fast path.
    #
    # Sets up columns once, then for each constraint family in
    # ``self._cstrs`` builds that family's COO contribution, converts to
    # row-CSR, and feeds it to HiGHS via ``addRows``.  The family's local
    # arrays go out of scope at the end of each iteration so peak
    # memory is bounded by one family's COO + the running HiGHS LP.
    #
    # Numerically identical to the passModel path: same primal, same
    # duals (HiGHS row indexing continues monotonically across
    # ``addRows`` calls so per-family ``base_row`` is preserved), same
    # objective value.  The lazy plans on each ``_Term`` survive
    # untouched so re-solves and ``WarmProblem`` stay valid.
    def _solve_streaming(
        self,
        *,
        options: dict | None,
        keep_solver: bool,
        save_memory: bool = False,
        _mps_out_path: str | None = None,
        _build_only: bool = False,
        tmp_dir: str | os.PathLike | None = None,
    ) -> Solution | None:
        _sp_emit, _sp_on = _make_solve_profile_emitter()
        # Small-coefficient cutoff threshold (0.0 ⇒ OFF).  Read once here
        # so every per-family floor below shares the same value.
        _coef_zero_thr = float(getattr(self, "coef_zero_threshold", 0.0) or 0.0)
        if _sp_on:
            _sp_emit(
                "enter",
                n_vars=len(self._vars),
                n_cstrs=len(self._cstrs),
                n_obj_terms=len(self._obj_terms),
                save_memory=int(save_memory),
                keep_solver=int(keep_solver),
            )
        # Build the per-column arrays inside this frame (rather than in
        # the public ``solve`` caller) so they die at the end of the
        # streaming solve rather than living through ``HiGHS.run()`` on
        # the outer stack.  The contents are identical to what
        # :class:`LpView` materialises for the non-streaming path; the
        # two builders diverge only on which downstream sink they feed
        # (``addCols`` vs ``HighsLp`` slots).
        n_cols = self._next_col
        col_lb = np.zeros(n_cols, dtype=np.float64)
        col_ub = np.full(n_cols, np.inf, dtype=np.float64)
        col_obj = np.zeros(n_cols, dtype=np.float64)
        col_int = np.zeros(n_cols, dtype=np.int8)  # 1 = integer column
        col_names: list[str] = [None] * n_cols  # type: ignore[list-item]
        if _sp_on:
            _sp_emit("col_arrays_alloc", n_cols=n_cols)

        for v in self._vars.values():
            ids = v.frame["col_id"].to_numpy()
            col_lb[ids] = float(v.lower)
            col_ub[ids] = float(v.upper)
            if v.integer:
                col_int[ids] = 1
            if v.dims:
                # Build the formatted "name[tag]" strings inside polars in
                # one expression, then a single .to_list() — avoids the
                # per-row Python f-string formatting loop.
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
                # ids is a 1-D int64 numpy array — Python int conversion is
                # cheap; loop body has zero arithmetic now.
                ids_list = ids.tolist()
                for cid, nm in zip(ids_list, tagged):
                    col_names[cid] = nm
            else:
                cid0 = int(ids[0])
                col_names[cid0] = v.name

        if _sp_on:
            _sp_emit("var_loop_done", n_cols=n_cols)

        # objective — scatter-add via np.add.at (one numpy op per term,
        # no per-nonzero Python iteration).  np.add.at handles the rare
        # case where a term frame contains duplicate col_ids correctly.
        # NOTE: materialize each term into a *local* DataFrame and let it
        # drop at the end of the iteration — we deliberately do NOT
        # populate any cache on _Term, so the eager frame is released
        # once the COO contribution is built.  WarmProblem and any
        # subsequent solve re-collect from the surviving lazy plan.
        # Layer 2 col-factor (off ⇒ no-op).  Objective gets col_factor
        # only — there is no row_factor entry for the cost row.
        _cf_obj = self._layer2_col_factor
        for t in self._obj_terms:
            # Defensive bake-before-consume (see _build_canonical_matrix
            # objective loop) — Sum(over=None) already bakes both slots.
            _obj_lazy = _apply_where_frames(t.lazy, t.dims, t.where_frames)
            _obj_lazy, _ = _apply_where_map_frames(_obj_lazy, t.dims, t.where_map_frames)
            f = _obj_lazy.collect()
            cids = f["col_id"].to_numpy()
            vals = f["coef"].to_numpy()
            if _cf_obj is not None:
                vals = vals * _cf_obj[cids]
            np.add.at(col_obj, cids, vals)
            del f

        if _sp_on:
            _sp_emit("obj_terms_collected", n_obj_terms=len(self._obj_terms))

        inf = highspy.kHighsInf

        # Translate +/-inf in the column bounds to HiGHS's sentinel.
        col_lb_h = np.where(col_lb == -np.inf, -inf, col_lb).astype(np.float64, copy=False)
        col_ub_h = np.where(col_ub == np.inf, inf, col_ub).astype(np.float64, copy=False)
        col_obj_h = col_obj.astype(np.float64, copy=False)

        if _sp_on:
            _sp_emit("col_bounds_translated", n_cols=n_cols)

        h = highspy.Highs()
        if _sp_on:
            _sp_emit("highs_constructed")

        # Apply solver options BEFORE any model state is established —
        # mirrors the non-streaming path (``presolve`` and friends must
        # be set before HiGHS sees the LP to take effect on run()).
        opts = options if options is not None else self._solver_options
        # HiGHS lazily initialises a process-global thread scheduler on
        # the first Highs construction with whatever ``threads`` default
        # is in force (typically ``getRunningHardwareThreads()`` =
        # however many cores the box has).  Setting ``threads`` to a
        # different value AFTER that point fails with "global scheduler
        # has already been initialized to use N threads".  If the
        # caller has passed an explicit ``threads`` (or ``parallel`` —
        # the same scheduler is consulted for parallel-on dispatch),
        # tear the global down BEFORE the option loop so the new value
        # takes effect.  ``resetGlobalScheduler(False)`` is a safe no-op
        # when no scheduler is active.
        _wants_threads = opts is not None and ("threads" in opts or "parallel" in opts)
        if _wants_threads:
            try:
                h.resetGlobalScheduler(False)
            except Exception:
                pass
        if opts:
            import warnings

            ok_status = getattr(highspy.HighsStatus, "kOk", None)
            for key, val in opts.items():
                try:
                    status = h.setOptionValue(key, val)
                except Exception as exc:
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

        if _sp_on:
            _sp_emit(
                "options_applied",
                n_opts=(len(opts) if opts else 0),
            )

        # Route HiGHS' log through Python ``sys.stdout`` now — BEFORE any
        # model op (the version banner is emitted on HiGHS' first log call,
        # which can precede ``run()``), so the whole log is captured under
        # Jupyter / Spine-Toolbox on Windows where fd 1 is not forwarded.
        route_highs_log_to_stdout(h)

        # Objective sense + offset up front; column data comes next.
        h.changeObjectiveSense(
            highspy.ObjSense.kMaximize if self._obj_sense == "max" else highspy.ObjSense.kMinimize
        )
        if self._obj_offset:
            h.changeObjectiveOffset(float(self._obj_offset))

        if _sp_on:
            _sp_emit("obj_meta_set")

        # Add all columns in one shot — no nonzeros yet (rows arrive
        # afterwards via addRows).  highspy expects int32 for
        # start/index even when num_new_nz == 0.
        empty_i32 = np.zeros(0, dtype=np.int32)
        empty_f64 = np.zeros(0, dtype=np.float64)
        h.addCols(
            int(n_cols),
            col_obj_h,
            col_lb_h,
            col_ub_h,
            0,
            empty_i32,
            empty_i32,
            empty_f64,
        )

        if _sp_on:
            _sp_emit("add_cols_done", n_cols=n_cols)

        # Integrality — same vectorized HighsVarType lookup as the
        # passModel path; only set when at least one column is integer.
        if col_int.any():
            kCont = int(highspy.HighsVarType.kContinuous)
            kInt = int(highspy.HighsVarType.kInteger)
            integ_arr = np.where(col_int, kInt, kCont).astype(np.uint8)
            all_idx = np.arange(n_cols, dtype=np.int32)
            h.changeColsIntegrality(int(n_cols), all_idx, integ_arr)

        if _sp_on:
            _sp_emit("integrality_set", n_int_cols=int(col_int.sum()))

        # Column names — passColName is per-item but cheap.
        for i, nm in enumerate(col_names):
            if nm is not None:
                h.passColName(i, nm)

        if _sp_on:
            _sp_emit("col_names_passed", n_cols=n_cols)

        # Under save_memory, HiGHS now owns the canonical name
        # storage, so we can drop the Python-side list.  At N=3000 dense
        # this is 18 M PyUnicode objects (~1.1 GB).  Cost: the returned
        # Solution.col_names becomes empty and Solution.constraint_dual()
        # won't find named columns — same trade as releasing the LP
        # source-of-truth.
        if save_memory:
            col_names = []

        # Stream-time LP-range accumulation.  Cost is a handful of O(n)
        # numpy scans on arrays we already build (per-family for matrix
        # / row_bound, once here for cost / col_bound), so peak memory
        # is unchanged.  Exposed on the returned :class:`Solution` as
        # ``streamed_lp_ranges`` for caller diagnostics — and consumed
        # by :mod:`polar_high.autoscale` to drive Layer 1 detection and
        # Layer 3 ``user_*_scale`` recommendation.
        _r_matrix_lo, _r_matrix_hi = math.inf, 0.0
        _r_row_lo, _r_row_hi = math.inf, 0.0
        _r_cost_lo, _r_cost_hi = _running_finite_nonzero_min_max(col_obj_h, math.inf, 0.0)
        # col bounds: drop the HiGHS infinity sentinel as well as zeros
        # (zeros aren't bounds in the scaling-decision sense).  Process
        # col_lb_h and col_ub_h separately — concatenating into a single
        # 2·n_cols array doubles the transient working set for no reason.
        _r_cb_lo, _r_cb_hi = _running_finite_nonzero_min_max(col_lb_h, math.inf, 0.0)
        _r_cb_lo, _r_cb_hi = _running_finite_nonzero_min_max(col_ub_h, _r_cb_lo, _r_cb_hi)

        # HiGHS has its own internal copies of the column bound / cost
        # arrays after addCols; we don't reference these locals again.
        # Releasing them now (~6·n_cols·8 bytes = ~864 MB at N=3000
        # dense for col_lb, col_ub, col_obj, col_lb_h, col_ub_h, col_obj_h
        # combined; ``col_obj_h`` may alias ``col_obj`` and ``col_lb_h``
        # / ``col_ub_h`` are fresh ``np.where`` outputs) lets the
        # steady-state RSS during the family loop and ``h.run()`` come
        # down accordingly.  Done now (rather than at the end of solve)
        # because we no longer hold these arrays on the outer caller
        # frame — the column build moved inside this method.
        del col_lb_h, col_ub_h, col_obj_h
        del col_lb, col_ub, col_obj, col_int

        if _sp_on:
            _sp_emit("col_arrays_dropped")

        # Walk the constraint families one at a time.  For each family
        # we collect that family's term plans, build local COO arrays,
        # convert to row-CSR, and call addRows.  All locals fall out of
        # scope at iteration end.
        row_names: list[str] = []
        next_row = 0

        for _fam_idx, (name, proto, over) in enumerate(self._cstrs):
            expr, sense, rhs = proto.expr, proto.sense, proto.rhs

            if over is None:
                row_count = 1
                row_index = pl.DataFrame({"_rid": [0]})
                axis_cols: list[str] = []
            else:
                row_count = over.height
                axis_cols = list(over.columns)
                row_index = over.with_columns(_rid=pl.int_range(0, over.height, dtype=pl.Int64))

            if _sp_on:
                _sp_emit(
                    "fam_start",
                    family=name,
                    family_idx=_fam_idx,
                    row_count=row_count,
                    term_count=len(expr.terms),
                )

            # base_row for this family is ``next_row``; HiGHS appends
            # rows in monotonic order via addRows, so per-family row
            # ranges are implicit — we just bump the counter.
            base_row = next_row
            next_row += row_count

            # rhs vector — Expr/Var on rhs gets moved to lhs as -terms.
            rhs_vec = np.zeros(row_count, dtype=np.float64)
            if isinstance(rhs, (int, float)):
                rhs_vec[:] = float(rhs)
            elif isinstance(rhs, Param):
                missing = [d for d in rhs.dims if d not in axis_cols]
                if missing:
                    raise ValueError(
                        f"constraint {name!r}: rhs Param has dim {missing} not in over={axis_cols}"
                    )
                on = list(rhs.dims)
                if on:
                    # Bound the peak RAM of multi-Param rhs products.
                    # rhs.lazy may be a chain of Param * Param * ... whose
                    # eager materialisation (``rhs.frame``) explodes
                    # intermediate join buffers by orders of magnitude
                    # vs the constraint's row_count (FlexTool's
                    # ``p_profile_value * p_process_existing_count *
                    # p_process_availability`` allocates ~30 GB for a
                    # constraint whose final row count is ~76k).
                    #
                    # Pre-prune via a semi-join against row_index's
                    # actual join keys, then collect through polars's
                    # streaming engine.  The semi-join gives the
                    # optimiser an explicit "keep only these keys" hint;
                    # the streaming engine bounds intermediate buffers
                    # while the upstream Param product runs.
                    ri_a, rf_a = _align_enum_join_keys(
                        row_index.lazy(),
                        rhs.lazy,
                        on,
                    )
                    # Build the semi-join key set from the ALIGNED row
                    # index so it shares the join-key dtypes with rf_a.
                    # Building from the un-aligned ``row_index.lazy()``
                    # raises a polars SchemaError on Enum mismatch when
                    # rhs.lazy carries a subset-aligned Enum on the join
                    # column (e.g. flextool's Lagrangian-region path).
                    keys_lazy = ri_a.select(on).unique()
                    rf_pruned = rf_a.join(keys_lazy, on=on, how="semi")
                    _plan = ri_a.join(rf_pruned, on=on, how="left")
                    try:
                        j = _plan.collect(engine="streaming")
                    except TypeError:
                        # polars < 1.x used the streaming=True kwarg.
                        j = _plan.collect(streaming=True)
                    if j.height != row_count:
                        dup_rids = (
                            j.group_by("_rid")
                            .agg(pl.len().alias("__n"))
                            .filter(pl.col("__n") > 1)
                            .sort("_rid")
                            .head(5)["_rid"]
                            .to_list()
                        )
                        sample = (
                            j.filter(pl.col("_rid").is_in(dup_rids))
                            .select(*on, "_rid", "value")
                            .head(10)
                        )
                        raise ValueError(
                            f"constraint {name!r}: rhs Param has duplicate keys on "
                            f"{on!r} — left join from row_index (rows={row_count}) "
                            f"produced {j.height} rows. Sample duplicates:\n{sample}"
                        )
                    rhs_vec = (
                        j.sort("_rid")["value"]
                        .fill_null(0.0)
                        .to_numpy()
                        .astype(np.float64, copy=False)
                    )
                else:
                    rhs_vec[:] = float(rhs.frame["value"][0])
            else:
                raise TypeError(f"constraint {name!r}: unsupported rhs type {type(rhs).__name__}")

            # Layer 2 row-factor on RHS (off ⇒ no-op).  base_row is
            # the 0-based absolute constraint row id; row_factor is
            # 0-indexed over constraints (HiGHS row index space starts
            # at 0 here — no cost row in the constraint space).
            if self._layer2_row_factor is not None and row_count:
                rhs_vec = rhs_vec * self._layer2_row_factor[base_row : base_row + row_count]

            if sense == "<=":
                row_lb = np.full(row_count, -inf, dtype=np.float64)
                row_ub = np.where(rhs_vec == np.inf, inf, rhs_vec).astype(np.float64, copy=False)
            elif sense == ">=":
                row_lb = np.where(rhs_vec == -np.inf, -inf, rhs_vec).astype(np.float64, copy=False)
                row_ub = np.full(row_count, inf, dtype=np.float64)
            elif sense == "==":
                rhs_h = np.where(rhs_vec == -np.inf, -inf, rhs_vec)
                rhs_h = np.where(rhs_h == np.inf, inf, rhs_h).astype(np.float64, copy=False)
                row_lb = rhs_h
                row_ub = rhs_h
            else:
                raise ValueError(f"sense must be '<=', '>=' or '=='; got {sense!r}")

            # Small-coefficient cutoff on the RHS (row bounds).  Floors
            # finite |bound| < threshold to 0.0; ±inf sentinels survive.
            if _coef_zero_thr > 0.0:
                row_lb = _floor_small_coefs(row_lb, _coef_zero_thr)
                row_ub = _floor_small_coefs(row_ub, _coef_zero_thr)

            if _sp_on:
                _sp_emit(
                    "fam_rhs_built",
                    family=name,
                    family_idx=_fam_idx,
                    row_count=row_count,
                )

            # Row names — same polars-side formatting as the
            # non-streaming path; stash for later passRowName loop.
            if over is None:
                row_names.append(name)
            else:
                row_names.extend(
                    over.select(
                        pl.format(
                            "{}[{}]",
                            pl.lit(name),
                            pl.concat_str(
                                [pl.col(d).cast(pl.String) for d in axis_cols], separator=","
                            ),
                        ).alias("__rn")
                    )["__rn"].to_list()
                )

            # Build this family's COO contribution.  Each term yields
            # either a "dim" plan (joined to row_index) or a "scalar"
            # plan (tiled across row_count rows).  We collect_all per
            # family so polars can still parallelize within a family,
            # but the materialised frames go out of scope at the end of
            # this iteration.
            row_index_lf = row_index.lazy()
            term_plans: list[tuple] = []
            for _term_idx, term in enumerate(expr.terms):
                if term.dims:
                    missing = [d for d in term.dims if d not in axis_cols]
                    if missing:
                        raise ValueError(
                            f"constraint {name!r}: term has open dims {term.dims}, "
                            f"but constraint axes are {axis_cols}; aggregate "
                            f"{missing} via Sum() before adding."
                        )
                    on = [d for d in term.dims if d in axis_cols]
                    # Prefer LHS prune-down when the term carries a Var
                    # reference + Param-chain (>=2 atomics) — mirror of
                    # _build_canonical_matrix's LHS prune-down branch.
                    # Single-Param / Sum-collapsed terms (var_source is
                    # None) keep the original semi-join path verbatim.
                    _lhs_psrc = term.param_sources if isinstance(term.param_sources, list) else None
                    _use_lhs_prune = (
                        term.var_source is not None
                        and _lhs_psrc is not None
                        and len(_lhs_psrc) >= 2
                        and not _prune_down_disabled()
                    )
                    # Block-COO sibling arm — identical dispatch to the
                    # non-streaming site (_build_canonical_matrix): fires
                    # ONLY when the Problem declared dense_axes AND this
                    # non-Sum Var×Param chain matches the dense-suffix
                    # contract.  Default ON; POLAR_HIGH_DISABLE_BLOCK_COO=1
                    # is the off switch.  Bit-identical to the polars path;
                    # any shape it can't reproduce returns None and we fall
                    # through to prune-down / fallback unchanged.  The
                    # helper returns a collected (_rid, col_id, coef) frame
                    # whose ``_rid`` is family-local (it joined the local
                    # int-range ``row_index_lf``), exactly matching the
                    # ``rids_local = j["_rid"]`` consumed below; we wrap it
                    # ``.lazy()`` so it appends as ``("dim", plan)`` like
                    # every other streaming term and flows through
                    # ``_collect_one``.
                    _block_spec = None
                    if not _block_coo_disabled():
                        _block_spec = _block_coo_classify(term, axis_cols, on, self._dense_axes)
                    # See _build_canonical_matrix: a deferred map-effect
                    # Where cannot be carried by the block-COO seed —
                    # bake-before-block (route to prune / fallback) keeps
                    # the path byte-identical until Phase C-3.
                    if term.where_map_frames is not None:
                        _block_spec = None
                    _use_block_coo = _block_spec is not None
                    # Sum-block-COO sibling arm — identical dispatch to
                    # Site 1 (_build_canonical_matrix).  Fires on a
                    # ``Sum``-wrapped Var×Param chain (var_source cleared by
                    # Sum, SumBlockMeta recipe captured) when the Problem
                    # declared dense_axes AND the recipe matches the suffix
                    # contract.  Same off switch; mutually exclusive with the
                    # non-Sum arm (a Sum term has var_source=None ⇒
                    # _block_spec is already None).  The builder joins the
                    # family-local ``row_index_lf`` so its ``_rid`` is
                    # family-local — exactly matching ``rids_local =
                    # j["_rid"]`` consumed below.  Wrapped ``.lazy()`` so it
                    # appends as ``("dim", plan)`` like every other streaming
                    # term.  Anything the rebuild can't reduce
                    # bit-equivalently returns None / raises the fallback ⇒
                    # the term reads its reduced ``term.lazy`` verbatim.
                    _sum_block_spec = None
                    if not _use_block_coo and not _block_coo_disabled():
                        _sum_block_spec = _sum_block_coo_classify(
                            term, axis_cols, on, self._dense_axes
                        )
                    _use_sum_block_coo = _sum_block_spec is not None
                    if _use_block_coo:
                        _verify_dense_sorted(
                            term.var_source.frame,
                            _block_spec["non_dense_dims"],
                            _block_spec["dense_dims"],
                            getattr(term.var_source, "name", None),
                        )
                        _t_blk0 = time.monotonic()
                        plan = _build_block_coo_plan(
                            row_index_lf,
                            axis_cols,
                            term.var_source,
                            _lhs_psrc,
                            on,
                            term.coef_scalar,
                            term.where_frames,
                            _block_spec,
                        )
                        if os.environ.get("POLAR_HIGH_BLOCK_COO_PROFILE") == "1":
                            _blk_wall = time.monotonic() - _t_blk0
                            _n_rows = int(plan.height)
                            _dense = _block_spec["dense_dims"]
                            _nb = _block_spec["dense_card"]
                            _avg = (_n_rows / _nb) if _nb else 0.0
                            sys.stderr.write(
                                f"[block_coo profile]\tphase=block_coo_term"
                                f"\tfamily={name}\tfamily_idx={_fam_idx}"
                                f"\tterm_idx={_term_idx}"
                                f"\tdense_dims={','.join(_dense)}"
                                f"\tn_blocks={_nb}"
                                f"\tavg_block_size={_avg:.2f}"
                                f"\twall_s={_blk_wall:.4f}\n"
                            )
                            sys.stderr.flush()
                        plan = plan.lazy()
                    elif _use_sum_block_coo:
                        # Verify the dense_axes sort contract on the
                        # RECIPE's Var (the seed block-COO slices) first.
                        _sm = term.sum_block_meta
                        _verify_dense_sorted(
                            _sm.var_source.frame,
                            _sum_block_spec["non_dense_dims"],
                            _sum_block_spec["dense_dims"],
                            getattr(_sm.var_source, "name", None),
                        )
                        _t_blk0 = time.monotonic()
                        try:
                            plan = _build_sum_block_coo_plan(
                                row_index_lf,
                                axis_cols,
                                _sm,
                                on,
                                _sum_block_spec,
                            )
                            _sum_block_fired = True
                        except _SumBlockCooFallback:
                            _sum_block_fired = False
                            plan = None
                        if _sum_block_fired:
                            if os.environ.get("POLAR_HIGH_BLOCK_COO_PROFILE") == "1":
                                _blk_wall = time.monotonic() - _t_blk0
                                _n_rows = int(plan.height)
                                _dense = _sum_block_spec["dense_dims"]
                                _nb = _sum_block_spec["dense_card"]
                                _avg = (_n_rows / _nb) if _nb else 0.0
                                sys.stderr.write(
                                    f"[block_coo profile]\tphase=block_coo_term"
                                    f"\tkind=sum\tphase_site=streaming"
                                    f"\tfamily={name}\tfamily_idx={_fam_idx}"
                                    f"\tterm_idx={_term_idx}"
                                    f"\tdense_dims={','.join(_dense)}"
                                    f"\tn_blocks={_nb}"
                                    f"\tavg_block_size={_avg:.2f}"
                                    f"\twall_s={_blk_wall:.4f}\n"
                                )
                                sys.stderr.flush()
                            plan = plan.lazy()
                        else:
                            # Reduced-``term.lazy`` fallback (identical to
                            # the final else arm): bake deferred filters,
                            # then the row_index semi-join + inner-join.
                            term_lazy_filtered = _apply_where_frames(
                                term.lazy, term.dims, term.where_frames
                            )
                            term_lazy_filtered, _ = _apply_where_map_frames(
                                term_lazy_filtered,
                                term.dims,
                                term.where_map_frames,
                            )
                            rl_a, tl_a = _align_enum_join_keys(row_index_lf, term_lazy_filtered, on)
                            keys_lazy = rl_a.select(on).unique()
                            tl_pruned = tl_a.join(keys_lazy, on=on, how="semi")
                            plan = rl_a.join(tl_pruned, on=on, how="inner").select(
                                "_rid", "col_id", "coef"
                            )
                    elif _use_lhs_prune:
                        plan = _build_lhs_pruned_plan(
                            row_index_lf,
                            axis_cols,
                            term.var_source,
                            _lhs_psrc,
                            on,
                            coef_scalar=term.coef_scalar,
                            where_frames=term.where_frames,
                            where_map_frames=term.where_map_frames,
                        ).select("_rid", "col_id", "coef")
                        if _sp_on:
                            _sp_emit(
                                "family_term_pruned_down",
                                family=name,
                                family_idx=_fam_idx,
                                term_idx=_term_idx,
                                n_atomics=len(_lhs_psrc),
                            )
                    else:
                        # Bake any deferred Where filters before the
                        # semi-join — fallback path applies them since
                        # prune-down isn't firing here.  Pure-filter then
                        # map-effect (dim-extending).
                        term_lazy_filtered = _apply_where_frames(
                            term.lazy, term.dims, term.where_frames
                        )
                        term_lazy_filtered, _ = _apply_where_map_frames(
                            term_lazy_filtered, term.dims, term.where_map_frames
                        )
                        rl_a, tl_a = _align_enum_join_keys(row_index_lf, term_lazy_filtered, on)
                        # Semi-join + streaming pattern, mirroring write_mps
                        # and _build_lp_arrays: prune the term plan against
                        # the row-index key set so polars can prune Param-
                        # product join chains rather than materialise a wide
                        # intermediate.  Same bug class on the LHS as RHS.
                        keys_lazy = rl_a.select(on).unique()
                        tl_pruned = tl_a.join(keys_lazy, on=on, how="semi")
                        plan = rl_a.join(tl_pruned, on=on, how="inner").select(
                            "_rid", "col_id", "coef"
                        )
                    term_plans.append(("dim", plan))
                else:
                    term_plans.append(("scalar", term.lazy.select("col_id", "coef")))

            fam_rows: list[np.ndarray] = []
            fam_cols: list[np.ndarray] = []
            fam_vals: list[np.ndarray] = []
            if term_plans:
                # Collect one term at a time with the streaming engine
                # to bound peak memory; per-term semi-join above is the
                # prerequisite that lets streaming actually prune.
                # Triple fallback for older polars / unsupported plans.
                def _collect_one(p: pl.LazyFrame) -> pl.DataFrame:
                    try:
                        return p.collect(engine="streaming")
                    except TypeError:
                        try:
                            return p.collect(streaming=True)
                        except TypeError:
                            return p.collect()
                    except Exception:
                        return p.collect()

                # Layer 2 side-vector multiplication (off ⇒ no-op).
                # fam_rows/cols use _local_ (0..row_count) row ids
                # within the family; row_factor is indexed by the
                # absolute constraint row id ``base_row + local``.
                _rf = self._layer2_row_factor
                _cf = self._layer2_col_factor
                collected = [_collect_one(p) for _, p in term_plans]
                for (kind, _), j in zip(term_plans, collected):
                    if kind == "dim":
                        if j.height == 0:
                            continue
                        rids_local = j["_rid"].to_numpy().astype(np.int64, copy=False)
                        cids = j["col_id"].to_numpy().astype(np.int64, copy=False)
                        vals = j["coef"].to_numpy().astype(np.float64, copy=False)
                        if _rf is not None:
                            vals = vals * _rf[base_row + rids_local]
                        if _cf is not None:
                            vals = vals * _cf[cids]
                        fam_rows.append(rids_local)
                        fam_cols.append(cids)
                        fam_vals.append(vals)
                    else:  # scalar — tile across the row_count rows
                        cids = j["col_id"].to_numpy().astype(np.int64, copy=False)
                        vals = j["coef"].to_numpy().astype(np.float64, copy=False)
                        if cids.size == 0:
                            continue
                        tiled_rows = np.repeat(np.arange(row_count, dtype=np.int64), cids.size)
                        tiled_cols = np.tile(cids, row_count)
                        tiled_vals = np.tile(vals, row_count)
                        if _rf is not None:
                            tiled_vals = tiled_vals * _rf[base_row + tiled_rows]
                        if _cf is not None:
                            tiled_vals = tiled_vals * _cf[tiled_cols]
                        fam_rows.append(tiled_rows)
                        fam_cols.append(tiled_cols)
                        fam_vals.append(tiled_vals)
                del collected

            if _sp_on:
                _sp_emit(
                    "fam_lhs_collected",
                    family=name,
                    family_idx=_fam_idx,
                    term_count=len(term_plans),
                    n_chunks=len(fam_rows),
                )

            if fam_rows:
                fr = np.concatenate(fam_rows)
                fc = np.concatenate(fam_cols)
                fv = np.concatenate(fam_vals)
                # Sum coefs for duplicate (row, col) pairs within the
                # family — same dedup the passModel path does globally.
                dedup = (
                    pl.DataFrame({"r": fr, "c": fc, "v": fv})
                    .group_by(["r", "c"])
                    .agg(pl.col("v").sum())
                )
                fr = dedup["r"].to_numpy()
                fc = dedup["c"].to_numpy()
                fv = dedup["v"].to_numpy().astype(np.float64, copy=False)
                del dedup
            else:
                fr = np.zeros(0, dtype=np.int64)
                fc = np.zeros(0, dtype=np.int64)
                fv = np.zeros(0, dtype=np.float64)

            if _sp_on:
                _sp_emit(
                    "fam_dedup_done",
                    family=name,
                    family_idx=_fam_idx,
                    nnz=int(fr.size),
                )

            # Convert COO → row-CSR.  HiGHS expects int32 for both
            # ``start`` and ``index`` arrays in addRows; nnz per family
            # is bounded by row_count * cols-per-row, so int32 is
            # always sufficient (and matches the int32 fast path the
            # non-streaming branch already uses globally for nnz <
            # 2**31).
            nnz = int(fr.size)
            if nnz:
                order = np.argsort(fr, kind="stable")
                sorted_r = fr[order]
                idx32 = fc[order].astype(np.int32)
                val64 = fv[order]
                # Small-coefficient cutoff on the LHS matrix coefficients.
                # Replaces |coef| < threshold with 0.0; keeps the entry
                # (structure/determinism preserved).
                if _coef_zero_thr > 0.0:
                    val64 = _floor_small_coefs(val64, _coef_zero_thr)
                starts = np.zeros(row_count + 1, dtype=np.int32)
                # bincount of row indices → counts per row, then cumsum
                counts = np.bincount(sorted_r.astype(np.int64), minlength=row_count)
                starts[1:] = np.cumsum(counts).astype(np.int32)
            else:
                idx32 = np.zeros(0, dtype=np.int32)
                val64 = np.zeros(0, dtype=np.float64)
                starts = np.zeros(row_count + 1, dtype=np.int32)

            if _sp_on:
                _sp_emit(
                    "fam_csr_built",
                    family=name,
                    family_idx=_fam_idx,
                    row_count=row_count,
                    nnz=nnz,
                )

            # addRows expects ``start`` of length num_new_row (no
            # trailing entry) — slice off the last cumulative count.
            h.addRows(
                int(row_count),
                row_lb,
                row_ub,
                int(nnz),
                starts[:row_count],
                idx32,
                val64,
            )

            if _sp_on:
                _sp_emit(
                    "fam_addRows_done",
                    family=name,
                    family_idx=_fam_idx,
                    row_count=row_count,
                    nnz=nnz,
                )

            # Update stream-time LP ranges from this family's arrays.
            # Process row_lb / row_ub separately (no concat copy).
            _r_matrix_lo, _r_matrix_hi = _running_finite_nonzero_min_max(
                val64,
                _r_matrix_lo,
                _r_matrix_hi,
            )
            _r_row_lo, _r_row_hi = _running_finite_nonzero_min_max(
                row_lb,
                _r_row_lo,
                _r_row_hi,
            )
            _r_row_lo, _r_row_hi = _running_finite_nonzero_min_max(
                row_ub,
                _r_row_lo,
                _r_row_hi,
            )

            # Track row range so dual lookup keeps working.  We don't
            # populate Problem-level metadata (Problem doesn't carry
            # _cstr_meta — that's a WarmProblem field), so this is
            # purely a local consistency check; the lazy plans on
            # _Term survive untouched.
            del term_plans, fam_rows, fam_cols, fam_vals, fr, fc, fv, idx32, val64, starts

        n_rows = next_row

        if _sp_on:
            _sp_emit("all_fams_done", n_rows=n_rows, n_fams=len(self._cstrs))

        # Materialise the four LP-range tuples (matrix / cost / col_bound /
        # row_bound) for :attr:`Solution.streamed_lp_ranges` — no per-key
        # ``_smallest`` / ``_largest`` lists, since they'd require name
        # lookup off CSC arrays we don't keep here.  ``None`` for an empty
        # category — detected by the still-zero ``_hi`` sentinel.
        def _pack(lo: float, hi: float) -> tuple[float, float] | None:
            if hi == 0.0:
                return None
            return (lo, hi)

        streamed_lp_ranges: dict[str, tuple[float, float] | None] = {
            "matrix": _pack(_r_matrix_lo, _r_matrix_hi),
            "cost": _pack(_r_cost_lo, _r_cost_hi),
            "col_bound": _pack(_r_cb_lo, _r_cb_hi),
            "row_bound": _pack(_r_row_lo, _r_row_hi),
        }

        # Row names — pass after all rows are added.  HiGHS row indices
        # are monotonic across addRows calls, so the global ``i`` here
        # matches the row index inside HiGHS.
        #
        # HiGHS' pybind11 binding requires ``str`` for passRowName; the
        # caller may pass ``None`` for rows without an explicit name
        # (e.g. internal slack constraints emitted by FlexTool's
        # ``_derived_block`` pipeline).  Fall back to a synthetic
        # ``row_<idx>`` name so diagnostics still work and the API
        # contract is satisfied.  The right long-term fix is for the
        # caller to assign a name to every row before solving.
        for i, nm in enumerate(row_names):
            h.passRowName(i, nm if nm is not None else f"row_{i}")

        if _sp_on:
            _sp_emit("row_names_passed", n_rows=n_rows)

        # Same logic as col_names above — HiGHS now owns the row name
        # storage; the Python list is ~1.1 GB of redundant strings at
        # N=3000 dense.
        if save_memory:
            row_names = []

        # Drop polar-high's polars/numpy LP source-of-truth before
        # handing off to HiGHS.  HiGHS already holds its own copy of
        # columns + rows; the lazy plans on ``_Term``, the rhs Param
        # frames on ``_CstrProto``, and the constraint family lists are
        # all redundant from here on.  Used to make peak RSS during
        # ``h.run()`` comparable to solvers that serialise to disk and
        # free their Python-side copy (e.g. linopy's ``io_api="lp"``).
        # See :meth:`_release_python_lp_inputs` for the contract.
        if save_memory:
            self._release_python_lp_inputs()
            if _sp_on:
                _sp_emit("release_python_lp_inputs_done")

            # HiGHS allocator round-trip via disk.  Even after every
            # polar-side reference is dropped, the C++ Highs instance
            # holds ~5 GB of slack on a N=3000 dense LP: the streaming
            # ``addRows`` path grows internal vectors incrementally, and
            # the resulting allocator high-water mark sits ~5 GB above
            # what ``readModel`` (which sizes everything once up front
            # from the MPS header) ends up using.  Writing the LP out,
            # tearing the original Highs down, asking glibc to release
            # heap arenas back to the OS, and reading the LP back into a
            # fresh Highs collapses that slack — at the cost of MPS file
            # I/O (~+90 s at N=3000 dense).  The trade is the whole
            # point of save_memory=True: lowest peak RSS, one shot.
            #
            # ``_mps_out_path`` / ``_build_only`` (private, used by
            # :meth:`build_only`): when ``_mps_out_path`` is set, write
            # to the caller's path instead of a temp file, and after the
            # drop+malloc_trim return ``None`` so the caller can drive
            # an external solver (e.g. a subprocess HiGHS).  The Problem
            # ends up in ``_released`` state — same contract as save_memory.
            if _build_only and _mps_out_path is None:
                raise ValueError(
                    "_build_only=True requires _mps_out_path to be set",
                )
            _caller_owns_mps = _mps_out_path is not None
            mps_path = (
                _mps_out_path
                if _mps_out_path is not None
                else tempfile.NamedTemporaryFile(suffix=".mps", delete=False, dir=tmp_dir).name
            )
            try:
                # Silence HiGHS' own "Writing the model to ..." /
                # "Reading the model from ..." chatter so benchmark
                # stdout stays clean.  We restore output_flag on the
                # fresh Highs from the caller's effective options dict
                # (re-applied below), so the user's preference wins.
                try:
                    h.setOptionValue("output_flag", False)
                except Exception:
                    pass
                h.writeModel(mps_path)
                if _sp_on:
                    _sp_emit("savemem_writeModel_done", mps_path=mps_path)
                h.clearModel()
                del h
                if _sp_on:
                    _sp_emit("savemem_highs_cleared")
                # Best-effort allocator release: glibc's malloc holds
                # freed arenas on its free-list by default; trim hands
                # them back to the OS so RSS actually drops.  Linux-
                # only (libc.so.6); skip silently elsewhere.
                if sys.platform.startswith("linux"):
                    try:
                        ctypes.CDLL("libc.so.6").malloc_trim(0)
                    except Exception:
                        pass

                # External-solve handoff: the caller takes it from here
                # (typically by spawning a subprocess HiGHS on
                # ``_mps_out_path``).  Skip the in-process
                # readModel + run + Solution-build steps below.  The
                # caller is expected to construct its own Solution from
                # the external solver's output arrays + ``self._vars``.
                if _build_only:
                    return None

                h = highspy.Highs()
                # See the addCols-time block above — same rationale:
                # tear the global scheduler down before re-applying
                # ``threads`` / ``parallel`` on the fresh Highs.
                if opts and ("threads" in opts or "parallel" in opts):
                    try:
                        h.resetGlobalScheduler(False)
                    except Exception:
                        pass
                # Re-apply the same solver options that were set on the
                # original Highs — they live on the C++ instance, not
                # on the MPS file, so the fresh Highs would otherwise
                # run with defaults.  ``opts`` is the effective dict
                # (per-call ``options`` overrides ``self._solver_options``;
                # see addCols-time block above).
                if opts:
                    import warnings as _warnings

                    _ok = getattr(highspy.HighsStatus, "kOk", None)
                    for _k, _v in opts.items():
                        try:
                            _st = h.setOptionValue(_k, _v)
                        except Exception as _exc:
                            _warnings.warn(
                                f"HiGHS rejected option {_k}={_v!r} on "
                                f"save_memory round-trip: {_exc}",
                                stacklevel=2,
                            )
                            continue
                        if _ok is not None and _st != _ok:
                            _warnings.warn(
                                f"HiGHS rejected option {_k}={_v!r} on "
                                f"save_memory round-trip (status={_st!r})",
                                stacklevel=2,
                            )
                # Mute readModel chatter on the fresh Highs too — the
                # caller's output_flag preference (if any) is already
                # in ``opts`` and got re-applied above.
                if not (opts and "output_flag" in opts):
                    try:
                        h.setOptionValue("output_flag", False)
                    except Exception:
                        pass
                h.readModel(mps_path)
                if _sp_on:
                    _sp_emit("savemem_readModel_done", mps_path=mps_path)
                # Restore the user's requested output_flag for run().
                if not (opts and "output_flag" in opts):
                    try:
                        h.setOptionValue("output_flag", True)
                    except Exception:
                        pass
            finally:
                # The caller-supplied MPS path (build_only / external
                # solve) is the caller's responsibility — they need it
                # to drive the subprocess.  Only sweep the temp file
                # we allocated ourselves.
                if not _caller_owns_mps and os.path.exists(mps_path):
                    os.unlink(mps_path)

        if _sp_on:
            _sp_emit("before_highs_run", n_rows=n_rows, n_cols=n_cols)

        h.run()

        if _sp_on:
            _sp_emit("after_highs_run")

        sol = h.getSolution()
        status_ok = h.getModelStatus() == highspy.HighsModelStatus.kOptimal
        obj_val = h.getObjectiveValue()
        col_value = np.asarray(sol.col_value, dtype=np.float64)
        row_dual = np.asarray(sol.row_dual, dtype=np.float64) if sol.row_dual else np.zeros(n_rows)
        col_dual = np.asarray(sol.col_dual, dtype=np.float64) if sol.col_dual else np.zeros(n_cols)

        if _sp_on:
            _sp_emit(
                "solution_extracted",
                optimal=int(bool(status_ok)),
                n_rows=n_rows,
                n_cols=n_cols,
            )

        if keep_solver:
            sol_highs: highspy.Highs | None = h
        else:
            sol_highs = None
            del h

        return Solution(
            optimal=status_ok,
            obj=obj_val,
            col_value=col_value,
            row_dual=row_dual,
            col_dual=col_dual,
            col_names=col_names,
            row_names=row_names,
            vars=dict(self._vars),
            highs=sol_highs,
            streamed_lp_ranges=streamed_lp_ranges,
        )


# ---------------------------------------------------------------------------
# Solution


class Solution:
    """Read-only view of the solved LP.  Look up variable values by
    name; values come back as a polars frame ``(*dims, value)``."""

    def __init__(
        self,
        *,
        optimal: bool,
        obj: float,
        col_value: np.ndarray,
        row_dual: np.ndarray,
        col_names: list[str],
        row_names: list[str],
        vars: dict[str, Var],
        col_dual: np.ndarray | None = None,
        highs: highspy.Highs | None = None,
        streamed_lp_ranges: dict | None = None,
    ):
        self.optimal = optimal
        self.obj = obj
        self.col_value = col_value
        self.row_dual = row_dual
        # Reduced-cost duals (per-column).  ``None`` keeps backwards
        # compatibility for callers that build :class:`Solution` directly
        # without a live HiGHS instance; the live solve paths populate it.
        self.col_dual = np.zeros(len(col_value), dtype=np.float64) if col_dual is None else col_dual
        self.col_names = col_names
        self.row_names = row_names
        self._vars = vars
        # Live ``highspy.Highs`` instance the solution came from.  Output
        # adapters (e.g. flextool's ``process_outputs`` writers, which read
        # MPS variable / row names directly off the solver) consume this.
        # ``None`` for callers that synthesise a Solution outside a real
        # solve.
        self.highs = highs
        # Stream-time LP coefficient ranges captured during
        # :meth:`Problem._solve_streaming` — dict with keys ``matrix`` /
        # ``cost`` / ``col_bound`` / ``row_bound``, each
        # ``(abs_min, abs_max) | None``.  ``None`` when the solution was
        # synthesised outside a streaming solve.
        self.streamed_lp_ranges = streamed_lp_ranges

    @property
    def max_primal_infeasibility(self) -> float:
        """Largest primal-constraint violation in the returned solution.

        A solver returns a vertex that sits within its feasibility tolerance
        of the active constraints, so a hand-rolled feasibility re-check on the
        solution (``f <= cap``, balance rows, …) must use a tolerance at least
        this large — a hard-coded magic constant either masks real violations
        or trips on the normal solver slack.  HiGHS enforces feasibility on the
        INTERNALLY-SCALED problem, so the unscaled slack reported here can
        exceed :attr:`primal_feasibility_tolerance`.

        ``0.0`` when no live solver is attached (a synthesised Solution)."""
        if self.highs is None:
            return 0.0
        return float(self.highs.getInfo().max_primal_infeasibility)

    @property
    def primal_feasibility_tolerance(self) -> float:
        """The solver's primal feasibility tolerance for this solve (the
        nominal, scaled-problem tolerance; see
        :attr:`max_primal_infeasibility` for the achieved unscaled slack).

        ``0.0`` when no live solver is attached (a synthesised Solution)."""
        if self.highs is None:
            return 0.0
        _status, val = self.highs.getOptionValue("primal_feasibility_tolerance")
        return float(val)

    def value(self, var_name: str) -> pl.DataFrame:
        """Long-form per-variable solution: ``(*dims, value)``."""
        v = self._vars[var_name]
        ids = v.frame["col_id"].to_numpy()
        vals = self.col_value[ids]
        return v.frame.select(*v.dims).with_columns(value=pl.Series(vals))

    def value_wide(
        self, var_name: str, time_dims: tuple[str, ...] = ("d", "t"), solve_name: str | None = None
    ) -> pl.DataFrame:
        """Wide-form, flextool-compatible: time dims become rows, the
        remaining dims are encoded as a tuple-stringified column header.

        For a 2-d variable like ``vq_state_up(n, d, t)``:
          long  :  rows = (n, d, t, value)
          wide  :  rows = (d, t)  +  one column per ``n`` (header = "west").

        For a 5-d variable like ``v_flow(p, source, sink, d, t)``:
          wide  :  rows = (d, t)  +  one column per (p, source, sink),
                   header = ``"('coal_plant', 'coal_market', 'west')"``
                   to match flextool's MultiIndex parquet round-trip.

        If ``solve_name`` is given, prepend a constant ``solve`` column
        for fuller flextool-output compatibility.
        """
        v = self._vars[var_name]
        long = self.value(var_name)
        for td in time_dims:
            if td not in v.dims:
                # variable doesn't carry this time dim; nothing to pivot on
                # → return long form
                return long
        other_dims = [d for d in v.dims if d not in time_dims]
        if not other_dims:
            return long.select(*time_dims, "value")

        if len(other_dims) == 1:
            key_expr = pl.col(other_dims[0]).cast(pl.String)
        else:
            inner = pl.concat_str(
                [pl.format("'{}'", pl.col(d).cast(pl.String)) for d in other_dims],
                separator=", ",
            )
            key_expr = pl.format("({})", inner)
        long = long.with_columns(_key=key_expr)
        wide = long.pivot(values="value", on="_key", index=list(time_dims))
        if solve_name is not None:
            wide = wide.with_columns(solve=pl.lit(solve_name)).select(
                "solve",
                *time_dims,
                *[c for c in wide.columns if c not in time_dims],
            )
        return wide

    def constraint_dual(self, name: str) -> pl.DataFrame:
        """Per-row dual values for a named constraint.  Returns a frame
        ``(over_dims..., dual)`` if the constraint had ``over=`` rows,
        else a single-row scalar frame ``(dual,)``."""
        # Walk row_names: each entry is "name[<key>]" or just "name" for
        # scalar.  Match leading prefix.
        prefix = f"{name}["
        idx = [i for i, rn in enumerate(self.row_names) if rn == name or rn.startswith(prefix)]
        if not idx:
            raise KeyError(f"constraint {name!r} not found in solution")
        duals = self.row_dual[idx]
        if len(idx) == 1 and self.row_names[idx[0]] == name:
            return pl.DataFrame({"dual": duals})
        # parse "name[key1,key2,…]" tags back into dim columns
        # (we don't carry over the original dim names here — caller passes
        # them in if needed; otherwise emit a single ``key`` string column)
        keys = [self.row_names[i][len(prefix) : -1] for i in idx]
        return pl.DataFrame({"key": keys, "dual": duals})


# ---------------------------------------------------------------------------
# WarmProblem — Option 4 hot LP updates
#
# Wraps a :class:`Problem` after it has been fully populated (vars,
# constraints, objective).  The first ``solve()`` builds the LP and runs
# HiGHS; subsequent ``solve()`` calls reuse the live ``Highs`` instance,
# letting the caller mutate just the parts that change between rolls
# (RHS values via :meth:`update_rhs`, objective costs via
# :meth:`update_obj_coef`) and re-run.
#
# This implements Option 4 from the design audit: warm LP updates that
# leverage highspy's ``changeRowsBounds`` / ``changeColsCost`` /
# ``changeCoeff`` APIs, which the GMPL pipeline in flextool can't.
#
# Scope guarantees:
#   * ``WarmProblem`` is a SIBLING of :class:`Problem`, not a subclass.
#   * Single-solve callers using ``Problem.solve()`` see ZERO behavioural
#     change — :class:`Problem` is not modified by this code path.
#   * The semantic-key indexes (``(var, *dims) → col_id`` and
#     ``(cstr, *over_dims) → row_id``) are derived from the underlying
#     :class:`Problem` after build, so callers can refer to LP cells by
#     their dim tuples without tracking col_ids manually.
#
# Restrictions in this Phase-2 minimum implementation:
#   * Update primitives:
#       - :meth:`update_rhs(cstr_name, new_param)` — replace the RHS
#         Param of a constraint family.  RHS values come from a join of
#         the new Param onto the original ``over`` frame.
#       - :meth:`update_obj_coef(var_name, new_param)` — replace the
#         objective coefficients of every column belonging to ``var_name``,
#         using values from a Param that broadcasts against the var's
#         dims.  Only useful when the objective contribution is exactly
#         ``new_param * var`` (the common "cost vector" pattern).
#   * The LP MATRIX (rows × cols × A) is invariant across re-solves;
#     to change a coefficient inside A use :meth:`update_coef` (one cell
#     at a time).  Adding / removing rows or cols is out of scope here —
#     a future extension can add them on top of the same plumbing.


# Relative tolerance for the :meth:`WarmProblem.compact_cuts` verify-restore
# belt.  Dropping cuts that are strictly slack at the incumbent optimum is
# LB-neutral by complementary slackness, so the re-solved objective must match
# the pre-compaction objective to within this (generous) relative band; a
# larger drift means a supposedly-slack cut was actually supporting (a
# degenerate numerical edge) and the deletion is rolled back.
_COMPACT_VERIFY_TOL: float = 1e-6


class WarmProblem:
    """Warm-update wrapper around a :class:`Problem`.

    Build a :class:`Problem` as usual (``add_var``, ``add_cstr``,
    ``set_objective``).  Wrap with :class:`WarmProblem`, then alternate
    ``update_*`` calls and ``solve()`` calls — the LP is built ONCE and
    only the changed coefficients / RHS values are pushed to HiGHS
    between solves.

    Typical rolling-horizon usage::

        wp = WarmProblem(p)
        sol_0 = wp.solve()
        for r in range(1, n_rolls):
            wp.update_rhs("balance", demand_param_for_roll[r])
            wp.update_obj_coef("v_flow", cost_param_for_roll[r])
            sol_r = wp.solve()

    The ``update_*`` calls are O(rows_or_cols_in_family); the ``solve()``
    benefits from HiGHS's hot-start (basis is preserved across calls).
    """

    def __init__(self, problem: Problem):
        if not isinstance(problem, Problem):
            raise TypeError("WarmProblem requires a polar_high.Problem instance")
        self._p = problem
        # Lazy state — populated on first solve()
        self._h: highspy.Highs | None = None
        self._n_cols: int = 0
        self._n_rows: int = 0
        self._col_names: list[str] | None = None
        self._row_names: list[str] | None = None
        # base_row / row_count / over / axis_cols per cstr name
        self._cstr_meta: dict[str, dict] = {}
        # cstr_name → ndarray of original RHS values (so update_rhs can
        # rebroadcast onto the original axis correctly)
        # var_name → ndarray of col_ids in declaration order
        self._var_cols: dict[str, np.ndarray] = {}
        # caches populated lazily on first update_obj_coef
        self._var_cols_i32: dict[str, np.ndarray] = {}
        self._obj_coef_cache: dict = {}
        # Param-tracked auto-update bookkeeping.
        # ``_mutable_params`` is the set of Param names declared mutable
        # via :meth:`declare_mutable` BEFORE the first solve().  Tracking
        # is opt-in — empty set means no overhead.
        self._mutable_params: set[str] = set()
        # ``_param_cells`` maps a tracked Param's name to a structured
        # numpy record per (row, col, dim_tuple, factor) cell that the
        # Param contributed to.  See :meth:`update_param`.
        # name -> dict with keys:
        #   "rows":     int64 ndarray   (row indices in the LP)
        #   "cols":     int64 ndarray   (col indices in the LP)
        #   "dim_keys": pl.DataFrame    (one row per cell, columns = the
        #                                Param's dim names; gives the
        #                                lookup key into the new Param's
        #                                value column)
        #   "factor":   float64 ndarray (residual = old_coef / old_p_val
        #                                — multiplies the new param value
        #                                to recover the new cell coef)
        #   "direction": int8 ndarray   (+1 numerator, -1 denominator —
        #                                determines whether the update
        #                                multiplies or divides by the
        #                                ratio)
        self._param_cells: dict[str, dict] = {}
        # Output-flag preference set via :meth:`set_output_flag` BEFORE the
        # first solve().  ``None`` means "leave HiGHS at its default"; once
        # the handle is built the flag is applied immediately and this
        # remembers the last requested value.
        self._output_flag: bool | None = None
        # -- retained cut rows (cutting-plane compaction) ------------------
        # Every row appended via :meth:`add_cut_row` is retained here so
        # :meth:`compact_cuts` can re-classify it against a solution and
        # (if strictly slack) delete it, bounding the master LP.  Keyed by
        # the LIVE HiGHS row id (remapped by :meth:`_delete_cut_rows` after a
        # HiGHS row compaction).  Value = (col_ids, coefs, lower) — the exact
        # arguments the cut was created with, so a dropped cut can be
        # re-added verbatim by the verify-restore belt.
        self._cut_rows: dict[int, tuple[list[int], list[float], float]] = {}
        # LP row id of the FIRST appended cut row.  Build-time rows occupy
        # ``[0, _first_cut_row)`` and are NEVER tracked here or deleted — they
        # back ``constraint_dual`` / named-row reads.  ``None`` until the first
        # cut is appended.
        self._first_cut_row: int | None = None
        # Col ids of recourse variables appended via :meth:`add_recourse_col`
        # (the Benders ``η`` columns).  A retained cut ``η_r − Σ slope·f >= rhs``
        # names its recourse variable as the single member of
        # ``col_ids ∩ _recourse_cols``; :meth:`compact_cuts` groups cuts by that
        # member for the dominance policy.
        self._recourse_cols: set[int] = set()

    @property
    def problem(self) -> Problem:
        """The underlying :class:`Problem`.

        Useful for diagnostics that need to inspect the un-built LP.
        For coefficient ranges, prefer
        :attr:`Solution.streamed_lp_ranges` on a returned
        :class:`Solution` (populated during the streaming solve at zero
        extra cost).
        """
        return self._p

    # -- public update API -----------------------------------------------

    def update_rhs(self, cstr_name: str, new_param: Param | float | int) -> None:
        """Replace the RHS of constraint family ``cstr_name`` with values
        drawn from ``new_param``.

        ``new_param`` may be a :class:`Param` whose dims match a subset
        of the constraint's ``over=`` axis (broadcasting to the rest), a
        scalar (broadcast to all rows), or a numpy array (positional —
        length must equal the family's row count, in the order of the
        original ``over`` frame).
        """
        self._require_built()
        meta = self._cstr_meta.get(cstr_name)
        if meta is None:
            raise KeyError(
                f"WarmProblem: no constraint family named "
                f"{cstr_name!r}; known: "
                f"{sorted(self._cstr_meta)}"
            )
        base_row: int = meta["base_row"]
        row_count: int = meta["row_count"]
        sense: str = meta["sense"]

        # Cache the row_idx int32 array — same every call.
        row_idx = meta.get("_row_idx_i32")
        if row_idx is None:
            row_idx = np.arange(base_row, base_row + row_count, dtype=np.int32)
            meta["_row_idx_i32"] = row_idx

        # Resolve new RHS vector (length == row_count, aligned to over).
        new_rhs = self._resolve_rhs_vec(new_param, meta, cstr_name)

        # Build lower / upper bound vectors per sense.
        inf = highspy.kHighsInf
        if sense == "<=":
            lb = meta.get("_neg_inf_lb")
            if lb is None:
                lb = np.full(row_count, -inf, dtype=np.float64)
                meta["_neg_inf_lb"] = lb
            ub = new_rhs.astype(np.float64, copy=False)
        elif sense == ">=":
            lb = new_rhs.astype(np.float64, copy=False)
            ub = meta.get("_pos_inf_ub")
            if ub is None:
                ub = np.full(row_count, inf, dtype=np.float64)
                meta["_pos_inf_ub"] = ub
        else:
            lb = new_rhs.astype(np.float64, copy=False)
            ub = lb

        # Mirror the initial-build small-coefficient cutoff on the warm
        # RHS path: any updated row bound that newly lands in
        # ``(0, threshold)`` is floored to 0.0, exactly as ``row_lb`` /
        # ``row_ub`` are floored at build time.  ``±inf`` sentinels (the
        # cached one-sided bounds) are preserved verbatim.  No-op when
        # the threshold is 0.0 (default), so warm updates stay
        # byte-identical with the cutoff off.
        _coef_zero_thr = float(getattr(self._p, "coef_zero_threshold", 0.0) or 0.0)
        if _coef_zero_thr > 0.0:
            lb = _floor_small_coefs(lb, _coef_zero_thr)
            ub = _floor_small_coefs(ub, _coef_zero_thr)
        self._h.changeRowsBounds(int(row_count), row_idx, lb, ub)

    def update_obj_coef(self, var_name: str, new_param: Param | float | int) -> None:
        """Replace the objective coefficient on every column of ``var_name``.

        Assumes the objective contribution from ``var_name`` is exactly
        ``coef[*dims] * var[*dims]`` for some ``coef`` Param; this method
        OVERWRITES that coefficient via ``h.changeColsCost``.  If the
        objective also has contributions from this variable through more
        complex algebra (e.g. ``var * unitsize * slope``), the update is
        still valid as long as ``new_param`` carries the full product —
        the caller is responsible for collapsing multi-Param products
        ahead of the call.

        This DOES NOT touch the cost coefficients of other variables.
        """
        self._require_built()
        v = self._p._vars.get(var_name)
        if v is None:
            raise KeyError(
                f"WarmProblem: no variable named {var_name!r}; known: {sorted(self._p._vars)}"
            )

        col_frame = v.frame  # has cols *v.dims, col_id

        # Cache the int32 col_id array — same every call.
        cache = self._var_cols_i32
        cids = cache.get(var_name)
        if cids is None:
            cids = col_frame["col_id"].to_numpy().astype(np.int32)
            cache[var_name] = cids

        if isinstance(new_param, (int, float)):
            cost = np.full(cids.size, float(new_param), dtype=np.float64)
        elif isinstance(new_param, Param):
            missing = [d for d in new_param.dims if d not in v.dims]
            if missing:
                raise ValueError(
                    f"update_obj_coef({var_name!r}): new_param has dims "
                    f"{missing} not in var dims {v.dims}"
                )
            on = list(new_param.dims)
            if not on:
                cost = np.full(cids.size, float(new_param.frame["value"][0]), dtype=np.float64)
            else:
                # Hot path: detect when the Param frame's dim columns are
                # row-identical to the var frame's dim columns (in the
                # var's order), so we can grab values directly.
                pf = new_param.frame
                cache_key = ("_obj_inplace", var_name, new_param.dims)
                inplace = self._obj_coef_cache.get(cache_key)
                if inplace is None:
                    if (
                        pf.height == col_frame.height
                        and len(on) == len(v.dims)
                        and set(on) == set(v.dims)
                    ):
                        a = pf.select(*v.dims).to_numpy()
                        b = col_frame.select(*v.dims).to_numpy()
                        inplace = bool(a.shape == b.shape and (a == b).all())
                    else:
                        inplace = False
                    self._obj_coef_cache[cache_key] = inplace
                if inplace:
                    cost = pf["value"].to_numpy().astype(np.float64, copy=False)
                else:
                    cf_a, pf_a = _align_enum_join_keys(col_frame, pf, on)
                    j = cf_a.join(pf_a, on=on, how="left").with_columns(
                        value=pl.col("value").fill_null(0.0)
                    )
                    cost = j["value"].to_numpy().astype(np.float64, copy=False)
        else:
            raise TypeError(
                f"update_obj_coef: new_param must be Param or "
                f"scalar, got {type(new_param).__name__}"
            )
        self._h.changeColsCost(int(cids.size), cids, cost)

    def update_obj_coef_array(
        self, var_name: str, dim_tuples: list[tuple], values: np.ndarray
    ) -> None:
        """Array-form of :meth:`update_obj_coef`.

        ``dim_tuples`` is a list of dim-value tuples (one per cell) for
        variable ``var_name``; each tuple must have one entry per dim
        in the var's declared signature.  ``values`` is a same-length
        numpy array of new objective coefficients.

        The columns are resolved positionally: ``values[k]`` becomes the
        new objective coefficient on the column whose dim-tuple is
        ``dim_tuples[k]``.  Vectorised — a single ``changeColsCost``
        call regardless of cell count.
        """
        self._require_built()
        v = self._p._vars.get(var_name)
        if v is None:
            raise KeyError(
                f"WarmProblem: no variable named {var_name!r}; known: {sorted(self._p._vars)}"
            )
        cols = self._resolve_dim_tuples(var_name, dim_tuples)
        vals = np.asarray(values, dtype=np.float64)
        if vals.size != cols.size:
            raise ValueError(
                f"update_obj_coef_array({var_name!r}): values length "
                f"{vals.size} != dim_tuples length {cols.size}"
            )
        cols_i32 = cols.astype(np.int32, copy=False)
        self._h.changeColsCost(int(cols_i32.size), cols_i32, vals)

    def fix_cols(self, var_name: str, dim_tuples: list[tuple], values: np.ndarray) -> None:
        """Fix the listed columns of ``var_name`` to the given values.

        For each ``(dim_tuple, value)`` pair, sets both the column's
        lower and upper bound to ``value`` (so the LP has no choice
        but to set the column at that level).  Used by the Lagrangian
        primal-recovery step ("fix-and-resolve").  Vectorised — single
        ``changeColsBounds`` call.
        """
        self._require_built()
        v = self._p._vars.get(var_name)
        if v is None:
            raise KeyError(
                f"WarmProblem: no variable named {var_name!r}; known: {sorted(self._p._vars)}"
            )
        cols = self._resolve_dim_tuples(var_name, dim_tuples)
        vals = np.asarray(values, dtype=np.float64)
        if vals.size != cols.size:
            raise ValueError(
                f"fix_cols({var_name!r}): values length {vals.size} != "
                f"dim_tuples length {cols.size}"
            )
        cols_i32 = cols.astype(np.int32, copy=False)
        self._h.changeColsBounds(int(cols_i32.size), cols_i32, vals, vals)

    def _resolve_dim_tuples(self, var_name: str, dim_tuples: list[tuple]) -> np.ndarray:
        """Translate a list of dim-tuples into an int64 array of col_ids
        in the same order.  Shared by ``update_obj_coef_array`` and
        ``fix_cols``.
        """
        v = self._p._vars[var_name]
        if len(dim_tuples) == 0:
            return np.zeros(0, dtype=np.int64)
        n_dims = len(v.dims)
        # Validate tuple shape eagerly
        for k, dt in enumerate(dim_tuples):
            if not isinstance(dt, tuple):
                raise TypeError(
                    f"{var_name!r}: dim_tuples[{k}] must be a tuple, got {type(dt).__name__}"
                )
            if len(dt) != n_dims:
                raise ValueError(
                    f"{var_name!r}: dim_tuples[{k}] has {len(dt)} "
                    f"elements but variable has dims {v.dims}"
                )
        cols_data = {d: [] for d in v.dims}
        for dt in dim_tuples:
            for d, val in zip(v.dims, dt):
                cols_data[d].append(val)
        lookup = pl.DataFrame(cols_data).with_row_index("__rid")
        on_v = list(v.dims)
        lookup_a, vframe_a = _align_enum_join_keys(lookup, v.frame, on_v)
        joined = lookup_a.join(vframe_a, on=on_v, how="left").sort("__rid")
        if joined["col_id"].null_count() > 0:
            missing_idx = (
                joined.with_row_index("__r2").filter(pl.col("col_id").is_null()).head(1)["__r2"][0]
            )
            raise KeyError(
                f"{var_name!r}: dim_tuple {dim_tuples[missing_idx]!r} "
                f"does not resolve to a column (var dims={v.dims})"
            )
        return joined["col_id"].to_numpy().astype(np.int64)

    def update_coef(self, row: int, col: int, value: float) -> None:
        """Update a single (row, col) coefficient in the constraint
        matrix.  Use :meth:`row_id_of_cstr` / :meth:`col_id_of_var` to
        resolve indices semantically."""
        self._require_built()
        self._h.changeCoeff(int(row), int(col), float(value))

    # -- incremental row/col append (cutting-plane primitive) -----------

    def add_cut_row(self, col_ids: list[int], coefs: list[float], lower: float) -> int:
        """Append a ``>=`` constraint row to the live (already-built) LP and
        return the new row's id.

        Appends ``Σ coefs[k] · x[col_ids[k]] >= lower`` to the live HiGHS
        model via ``addRow``.  This is a POST-build mutation of the warm
        ``_h`` handle (the same class of live edit as :meth:`update_rhs` /
        :meth:`update_coef`); it deliberately bypasses the build-time
        ``Problem.add_cstr`` DSL lock, which only guards the fixed-size
        Layer-2 autoscale side vectors and is irrelevant once the model is
        built.  The caller is responsible for keeping the master autoscale
        OFF (or for pre-scaling ``coefs``) so the appended row lives on the
        built columns' scale — there is no auto-scaling for appended rows.

        ``col_ids`` may mix existing column ids (resolve via
        :meth:`col_id_of_var`) and ids of columns previously appended with
        :meth:`add_recourse_col`.  A subsequent :meth:`solve` warm
        re-optimises the grown model; the appended row's dual is then
        readable on the returned :class:`Solution` by ``row_dual[row_id]``
        (read BY ROW ID, not via :meth:`Solution.constraint_dual`, which
        only knows named build-time rows).

        Bumps the cached ``_n_rows`` (the zero-fill fallback size in
        :meth:`solve`) and appends a generated name to ``_row_names`` (kept
        index-aligned so name-indexed reads stay correct).  Returns the
        ``row_id`` HiGHS assigned (== the pre-append ``getNumRow()``).
        """
        self._require_built()
        if len(col_ids) != len(coefs):
            raise ValueError(
                f"add_cut_row: col_ids and coefs length mismatch ({len(col_ids)} != {len(coefs)})"
            )
        idx = np.asarray(col_ids, dtype=np.int32)
        val = np.asarray(coefs, dtype=np.float64)
        inf = highspy.kHighsInf
        row_id = int(self._h.getNumRow())
        # addRow(lower, upper, num_nz, indices_i32, values_f64)
        self._h.addRow(float(lower), inf, int(idx.size), idx, val)
        # Bump cached row metadata that an append does NOT auto-update:
        #  * _n_rows sizes the solve() zero-fill fallback (mis-sized after
        #    append would silently return a wrong-length dual array on the
        #    empty-dual LP edge case).
        #  * _row_names is passed verbatim into Solution and walked by
        #    constraint_dual; keep it index-aligned so a name-indexed read
        #    of a build-time row is not shifted.
        self._n_rows += 1
        name = f"benders_cut_{row_id}"
        if self._row_names is not None:
            self._row_names.append(name)
        # Mirror the cached name into the live HiGHS model so MPS/name
        # readers (Solution.highs consumers) see the appended row too.
        self._h.passRowName(row_id, name)
        # Retain the cut so :meth:`compact_cuts` can classify + drop it later.
        # The first appended cut fixes the build-time / cut row boundary.
        if self._first_cut_row is None:
            self._first_cut_row = row_id
        self._cut_rows[row_id] = (list(col_ids), list(coefs), float(lower))
        return row_id

    def add_recourse_col(
        self, name: str, cost: float, lower: float = -np.inf, upper: float = np.inf
    ) -> int:
        """Append a single free/bounded column (e.g. a Benders recourse
        ``η``) to the live LP and return its col id.

        Symmetric with :meth:`add_cut_row`: a POST-build ``addCol`` on the
        warm ``_h`` handle, bumping the cached ``_n_cols`` (zero-fill
        fallback size) and ``_col_names``.  ``±np.inf`` bounds are mapped to
        the HiGHS ``kHighsInf`` sentinel.  The returned id can be referenced
        from a later :meth:`add_cut_row`; its value is read on the
        :class:`Solution` by ``col_value[col_id]`` (appended columns are not
        in any ``Problem._vars`` frame, so :meth:`Solution.value` cannot see
        them — read by id).
        """
        self._require_built()
        inf = highspy.kHighsInf
        lo = -inf if lower == -np.inf else float(lower)
        hi = inf if upper == np.inf else float(upper)
        col_id = int(self._h.getNumCol())
        # addCol(cost, lower, upper, num_nz, indices_i32, values_f64)
        self._h.addCol(
            float(cost), lo, hi, 0, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
        )
        self._n_cols += 1
        if self._col_names is not None:
            self._col_names.append(name)
        self._h.passColName(col_id, name)
        # Register as a recourse column so :meth:`compact_cuts` (dominance
        # policy) can group retained cuts by the recourse variable they bound.
        self._recourse_cols.add(col_id)
        return col_id

    # -- cut compaction (bound the growing master LP) -------------------

    def compact_cuts(
        self,
        solution: Solution,
        *,
        policy: str = "slack",
        trial_col_values=None,
        tol_rel: float = 1e-7,
        verify: bool = True,
    ) -> dict:
        """Prune retained cut rows to bound the active master LP.

        A cutting-plane master (the Benders master) grows one ``>=`` cut row
        per iteration via :meth:`add_cut_row`; left unpruned its warm
        re-solve time grows super-linearly.  Two selection policies are
        available, both LB-safe (they only drop cuts that do not support the
        certified lower bound) and both belted by the same verify-restore
        rollback:

        ``policy="slack"`` (default):
            Drop the cut rows that are strictly slack (positive primal slack,
            hence zero dual by complementary slackness) at ``solution``,
            keeping the binding ones.  Classification is purely by PRIMAL
            slack of the stored cut against ``solution.col_value``: for a cut
            ``Σ coefs·x >= lower`` the slack is ``Σ coefs·x - lower >= 0``; a
            cut with ``slack <= tol`` is BINDING (kept), one with
            ``slack > tol`` is strictly slack (dropped).  ``tol`` is a per-cut
            relative tolerance scaled by the magnitude of ``lower`` and the
            terms.

        ``policy="dominance"`` (de Matos–Philpott–Finardi / Guigues
        Limited-Memory Level-1):
            At a DEGENERATE optimum almost every cut can be primal-binding
            (slack ≈ 0), so slack-deletion prunes nothing even though many
            cuts are redundant TIES at their recourse group's max value.
            Dominance selection instead groups the retained cuts by the
            recourse variable they bound (the single member of
            ``col_ids ∩ _recourse_cols``, registered by
            :meth:`add_recourse_col`) and, over a WINDOW of recent master
            trial points ``trial_col_values`` (a list of ``col_value`` arrays,
            most-recent last; defaults to ``[solution.col_value]``), keeps for
            each group at each trial point the OLDEST cut that achieves the
            group's max value there (within a relative tolerance).  Every cut
            not a keeper at any trial point is DROPPED — it is dominated (or a
            redundant tie) at every trial point, so the outer approximation of
            the recourse function at those points, and the LB it certifies,
            is unchanged.  A cut whose ``col_ids`` contains no recourse column
            forms its own singleton group and is always kept (generic
            non-Benders use degrades safely to "keep all").

        When ``verify`` (default), after deleting rows the master is re-solved
        and its objective compared to the incumbent ``solution.obj``.  A drift
        larger than :data:`_COMPACT_VERIFY_TOL` (relative) means a dropped cut
        was in fact supporting the optimum (a degenerate numerical edge) — the
        deletion is ROLLED BACK by re-appending every dropped cut verbatim and
        re-solving, and the returned dict reports ``restored=True`` with
        ``dropped=0``.  On a well-behaved master the belt never fires.

        Returns ``{"kept": int, "dropped": int, "restored": bool}`` — cut
        counts AFTER the operation (``kept`` == number of retained cut rows
        that remain, ``dropped`` == number removed; on a restore ``kept`` is
        the pre-compaction count and ``dropped`` is 0).
        """
        self._require_built()
        if policy == "slack":
            drop_ids = self._classify_slack_drops(solution, tol_rel)
        elif policy == "dominance":
            drop_ids = self._classify_dominance_drops(solution, trial_col_values, tol_rel)
        else:
            raise ValueError(
                f"compact_cuts: unknown policy {policy!r} (expected 'slack' or 'dominance')"
            )
        return self._apply_cut_drops(drop_ids, float(solution.obj), verify)

    def _classify_slack_drops(self, solution: Solution, tol_rel: float) -> list[int]:
        """Return the cut row ids strictly slack at ``solution`` (slack policy)."""
        x = solution.col_value
        drop_ids: list[int] = []
        for row_id, (col_ids, coefs, lower) in self._cut_rows.items():
            ax = 0.0
            abs_terms = 0.0
            for i, c in zip(col_ids, coefs):
                term = c * float(x[i])
                ax += term
                abs_terms += abs(term)
            slack = ax - lower
            tol = tol_rel * max(1.0, abs(lower), abs_terms)
            if slack > tol:
                drop_ids.append(row_id)  # strictly slack — drop candidate
        return drop_ids

    def _classify_dominance_drops(
        self, solution: Solution, trial_col_values, tol_rel: float
    ) -> list[int]:
        """Return the dominated cut row ids (dominance policy).

        Groups the retained cuts by their recourse column ``η_r`` and, over the
        window of trial points, keeps per group the oldest cut that imposes the
        TIGHTEST lower bound on ``η_r`` at each trial point (the active /
        dominant cut there); every other cut is dropped.

        A Benders optimality cut ``η_r − Σ slope·f >= rhs`` is stored as
        ``(col_ids, coefs, lower)`` with the recourse column carrying coef
        ``+1`` and the ``f`` columns carrying ``−slope``; it constrains
        ``η_r >= rhs + Σ slope·f``.  At a trial point ``x_t`` the lower bound
        the cut imposes on ``η_r`` is therefore ::

            d = rhs + Σ slope·f
              = lower − Σ_{i ∈ non-recourse cols} coef_i · x_t[i]

        (the recourse column's own ``+1·η_t`` term is EXCLUDED — it is common to
        every cut in the group, so it cancels in the comparison and, more
        importantly, must not be mixed in: including it would rank cuts by their
        SLACK at ``x_t`` and keep the most-slack cut instead of the binding
        one).  The dominant cut at ``x_t`` is the one with the greatest ``d``;
        keeping it preserves ``η_r``'s active lower bound there, so the master
        optimum — and the certified LB — is unchanged.  See :meth:`compact_cuts`.
        """
        # Assemble the trial-point window; always include the incumbent as the
        # latest point (most-recent last).
        if trial_col_values is None:
            trials = [solution.col_value]
        else:
            trials = list(trial_col_values)
            latest = solution.col_value
            if not trials or trials[-1] is not latest:
                trials.append(latest)
        if not trials:
            trials = [solution.col_value]

        # Group retained cuts by their recourse column.  A cut with no recourse
        # column is its own singleton group (keyed by a unique sentinel) so it
        # is always kept (generic non-Benders use degrades safely to keep-all).
        groups: dict[object, list[int]] = {}
        for row_id, (col_ids, _coefs, _lower) in self._cut_rows.items():
            recourse = [c for c in col_ids if c in self._recourse_cols]
            if len(recourse) == 1:
                key: object = recourse[0]
            else:
                # 0 (generic non-Benders) or >1 (ill-formed cut) → singleton.
                key = ("__singleton__", row_id)
            groups.setdefault(key, []).append(row_id)

        keep_ids: set[int] = set()
        for group_key, group_rows in groups.items():
            # Sort by row id so "oldest" (smallest id) is deterministic.
            group_rows.sort()
            recourse_col = group_key if not isinstance(group_key, tuple) else None
            for x in trials:
                best_d = -math.inf
                best_abs = 0.0
                dvals: dict[int, float] = {}
                for row_id in group_rows:
                    col_ids, coefs, lower = self._cut_rows[row_id]
                    # d = lower − Σ_{non-recourse cols} coef·x_t (the lower bound
                    # this cut imposes on η_r at x_t); the recourse column's own
                    # term is excluded so cuts are ranked by tightness, not slack.
                    d = float(lower)
                    a = abs(float(lower))
                    for i, c in zip(col_ids, coefs):
                        if i == recourse_col:
                            continue
                        term = c * float(x[i])
                        d -= term
                        a += abs(term)
                    dvals[row_id] = d
                    if d > best_d:
                        best_d = d
                    if a > best_abs:
                        best_abs = a
                tol = tol_rel * max(1.0, abs(best_d), best_abs)
                # Oldest cut (smallest row id, group_rows is sorted) imposing the
                # tightest bound within tol is the keeper at this trial point.
                for row_id in group_rows:
                    if dvals[row_id] >= best_d - tol:
                        keep_ids.add(row_id)
                        break

        return [r for r in self._cut_rows if r not in keep_ids]

    def _apply_cut_drops(self, drop_ids: list[int], obj0: float, verify: bool) -> dict:
        """Delete ``drop_ids`` with the shared verify-restore belt.

        Common tail for both selection policies: snapshot the dropped defs,
        delete the rows, re-solve, and roll back (re-append every dropped cut
        verbatim) if the objective drifts past :data:`_COMPACT_VERIFY_TOL`.
        """
        n_total = len(self._cut_rows)
        if not drop_ids:
            # Nothing to drop → no-op (leave the LP untouched).
            return {"kept": n_total, "dropped": 0, "restored": False}

        # Snapshot the dropped cut defs BEFORE mutating, so the verify-restore
        # belt can re-add them verbatim if the deletion moved the optimum.
        dropped_defs = [self._cut_rows[r] for r in drop_ids]
        dropped_count = len(drop_ids)
        kept_before = n_total - dropped_count

        self._delete_cut_rows(drop_ids)

        if verify:
            sol2 = self.solve()
            drift = abs(float(sol2.obj) - obj0) / max(1.0, abs(obj0))
            if drift > _COMPACT_VERIFY_TOL:
                # Degenerate edge: a dropped cut was actually supporting the
                # optimum.  Roll back — re-append every dropped cut verbatim
                # and re-solve to restore the certified LP.
                for col_ids, coefs, lower in dropped_defs:
                    self.add_cut_row(col_ids, coefs, lower)
                self.solve()
                return {
                    "kept": kept_before + dropped_count,
                    "dropped": 0,
                    "restored": True,
                }

        return {"kept": kept_before, "dropped": dropped_count, "restored": False}

    def _delete_cut_rows(self, row_ids) -> None:
        """Delete the given cut rows from the live LP and repair bookkeeping.

        Deletes ONLY appended cut rows (ids ``>= _first_cut_row``).  HiGHS
        compacts the remaining rows down after a ``deleteRows``, so every
        surviving row id above a deleted one shifts down by the count of
        deleted ids below it; this method remaps ``_cut_rows`` keys and
        rebuilds ``_row_names`` accordingly, and decrements ``_n_rows``.
        Build-time rows (``[0, _first_cut_row)``) are guaranteed untouched
        because every deleted id is ``>= _first_cut_row``, so their positions
        — and hence ``constraint_dual`` / named-row reads — are preserved.
        """
        ids = sorted({int(r) for r in row_ids})
        if not ids:
            return
        # Guard: never delete a build-time row (would corrupt named-row /
        # constraint_dual reads and the fixed-size autoscale side vectors).
        assert self._first_cut_row is not None and ids[0] >= self._first_cut_row, (
            f"_delete_cut_rows: refusing to delete build-time row(s); "
            f"ids={ids}, first_cut_row={self._first_cut_row}"
        )
        # Guard: row deletion invalidates the absolute row-index arrays that
        # back tracked mutable Params.  The Benders master has none; this only
        # fires on misuse of the primitive on a param-tracked WarmProblem.
        if self._mutable_params or self._param_cells:
            raise RuntimeError(
                "_delete_cut_rows: cannot delete rows on a WarmProblem with "
                "tracked mutable Params (_mutable_params / _param_cells "
                "non-empty) — the stored absolute row indices would be "
                "corrupted by the HiGHS row compaction."
            )
        self._require_built()
        # deleteRows(num, indices_i32) — verified against the highspy binding.
        self._h.deleteRows(len(ids), np.asarray(ids, dtype=np.int32))
        self._n_rows -= len(ids)

        drop_set = set(ids)
        # Rebuild _row_names index-aligned to the surviving rows (drop the
        # deleted positions, preserve order).
        if self._row_names is not None:
            self._row_names = [nm for i, nm in enumerate(self._row_names) if i not in drop_set]

        # Remap _cut_rows: HiGHS compacts, so a surviving row at old id ``r``
        # moves to ``r - (#deleted ids < r)``.  Build-time ids (< first_cut)
        # are unaffected since every deleted id is >= first_cut.
        def _shift(r: int) -> int:
            return r - sum(1 for d in ids if d < r)

        self._cut_rows = {
            _shift(r): defn for r, defn in self._cut_rows.items() if r not in drop_set
        }
        # The first-cut boundary is a build-time position and is <= every
        # surviving cut id after the shift; it does not move (deletions are
        # all >= it, and _shift on it would subtract 0).  Leave it as-is; if
        # no cuts remain it still correctly marks where cuts begin.

    # -- Param-tracked auto-update --------------------------------------

    def declare_mutable(self, *param_names: str) -> None:
        """Declare a set of :class:`Param` names whose values should be
        tracked into LP cells, so :meth:`update_param` can later push
        new values into the live HiGHS instance via ``changeCoeff``.

        MUST be called BEFORE the first :meth:`solve`.  Tracking is
        opt-in: Params not declared here pay no bookkeeping cost.

        Pass the same names that the Params carry on their ``.name``
        field — typically the FlexData attribute name (``"p_inflow"``,
        ``"p_penalty_up"`` etc.).
        """
        if self._h is not None:
            raise RuntimeError(
                "declare_mutable must be called before solve(); the LP "
                "has already been built and tracking is fixed."
            )
        for n in param_names:
            if not isinstance(n, str):
                raise TypeError(
                    f"declare_mutable: param names must be strings, got {type(n).__name__}"
                )
            self._mutable_params.add(n)

    def set_output_flag(self, enabled: bool) -> None:
        """Enable or disable HiGHS' native solve log for this problem.

        ``enabled=False`` mutes the per-solve HiGHS output (the
        ``output_flag`` HiGHS option) for THIS WarmProblem across all of
        its cold / warm / retry solves — callers that drive many parallel
        sub-solves (Benders regions, Lagrangian subproblems) use this to
        keep stdout clean and emit their own concise progress log instead.

        May be called either before the first :meth:`solve` (the flag is
        applied when the HiGHS handle is built) or after it (applied to the
        live handle immediately).  The preference persists on the handle,
        so a single call suffices for the whole lifetime.
        """
        self._output_flag = bool(enabled)
        if self._h is not None:
            self._h.setOptionValue("output_flag", bool(enabled))

    def update_param(self, param_name: str, new_param: Param | float | int) -> None:
        """Replace the values of a tracked Param.  Every LP cell whose
        coefficient was originally a function of ``param_name`` is
        re-computed from the new Param's values and pushed via
        ``h.changeCoeff``.

        ``new_param`` must be either a scalar (broadcast to all tracked
        cells) or a :class:`Param` whose dim signature matches the
        signature recorded for that Param at build time.

        Raises if ``param_name`` was not in :meth:`declare_mutable`'s
        list (silent corruption is worse than a hard error).
        """
        self._require_built()
        if param_name not in self._mutable_params:
            raise ValueError(
                f"update_param({param_name!r}): not declared mutable; "
                f"call declare_mutable({param_name!r}) before the first "
                f"solve(). Declared: {sorted(self._mutable_params)}"
            )
        cells = self._param_cells.get(param_name)
        if cells is None or cells["rows"].size == 0:
            # Param was tracked but never reached an LP cell (e.g. its
            # term was dropped by Sum aggregating away its dims).
            return

        rows = cells["rows"]
        cols = cells["cols"]
        n = rows.size

        # Resolve new value for every tracked cell.
        if isinstance(new_param, (int, float)):
            new_vals = np.full(n, float(new_param), dtype=np.float64)
        elif isinstance(new_param, Param):
            sig = cells["dim_signature"]
            if tuple(new_param.dims) != sig:
                raise ValueError(
                    f"update_param({param_name!r}): new_param dims "
                    f"{new_param.dims} differ from the originally tracked "
                    f"signature {sig}"
                )
            if not sig:
                new_vals = np.full(n, float(new_param.frame["value"][0]), dtype=np.float64)
            else:
                # Position-aligned lookup via a left-join on the cached
                # per-cell dim_keys frame.  We add a stable row index
                # so we can re-sort the join output back into cell
                # order.
                dim_keys: pl.DataFrame = cells["dim_keys"]
                _sig_on = list(sig)
                _dk_lhs = dim_keys.with_row_index("__rid")
                _dk_lhs, _np_rhs = _align_enum_join_keys(_dk_lhs, new_param.frame, _sig_on)
                lookup = _dk_lhs.join(_np_rhs, on=_sig_on, how="left").sort("__rid")
                new_vals = lookup["value"].fill_null(0.0).to_numpy().astype(np.float64, copy=False)
        else:
            raise TypeError(
                f"update_param({param_name!r}): new_param must be Param "
                f"or scalar, got {type(new_param).__name__}"
            )

        # Apply direction-aware update.  For numerator entries
        # (direction == +1) the new coefficient is factor × new_value.
        # For denominator entries (direction == -1) it is
        # factor / new_value.  ``factor`` was recorded at build time as
        # old_coef / old_value (numerator) or old_coef × old_value
        # (denominator), so this product / quotient recovers the
        # correct cell value with no further bookkeeping.
        directions = cells["direction"]  # int8 ndarray, +1 or -1
        factor = cells["factor"]
        # Numerator path (most common).
        new_coefs = factor * new_vals
        if (directions != 1).any():
            denom_mask = directions == -1
            # avoid division by zero — treat as 0 update
            safe_vals = np.where(denom_mask & (new_vals == 0.0), 1.0, new_vals)
            denom_coefs = factor / safe_vals
            denom_coefs = np.where(denom_mask & (new_vals == 0.0), 0.0, denom_coefs)
            new_coefs = np.where(denom_mask, denom_coefs, new_coefs)

        # Mirror the initial-build small-coefficient cutoff on the warm
        # in-place path: any updated matrix coefficient that newly lands
        # in ``(0, threshold)`` is floored to exactly 0.0 before it
        # reaches HiGHS, exactly as ``sorted_v`` / ``val64`` are floored
        # at build time.  No-op when the threshold is 0.0 (default), so
        # warm updates stay byte-identical with the cutoff off.
        _coef_zero_thr = float(getattr(self._p, "coef_zero_threshold", 0.0) or 0.0)
        if _coef_zero_thr > 0.0:
            new_coefs = _floor_small_coefs(new_coefs, _coef_zero_thr)

        h = self._h
        rows_list = rows.tolist()
        cols_list = cols.tolist()
        coefs_list = new_coefs.tolist()
        for r, c, v in zip(rows_list, cols_list, coefs_list):
            h.changeCoeff(r, c, v)

    # -- semantic-key lookups --------------------------------------------

    def col_id_of_var(self, var_name: str, dims: tuple | dict | None = None) -> int | np.ndarray:
        """Return the col_id(s) for a variable.

        ``dims=None`` returns every col_id in the variable's family
        (numpy array, ordered by the var's declaration order).
        ``dims`` as a tuple of dim values returns the single col_id for
        that one cell (a python int).  ``dims`` as a dict
        ``{dim_name: value}`` is a partial filter — returns a numpy
        array of the matching col_ids.
        """
        v = self._p._vars[var_name]
        f = v.frame
        if dims is None:
            return f["col_id"].to_numpy()
        if isinstance(dims, tuple):
            if len(dims) != len(v.dims):
                raise ValueError(
                    f"col_id_of_var({var_name!r}): expected "
                    f"{len(v.dims)} dim values for {v.dims}, "
                    f"got {len(dims)}"
                )
            mask = pl.lit(True)
            for d, val in zip(v.dims, dims):
                mask = mask & (pl.col(d) == val)
            sel = f.filter(mask)
            if sel.height != 1:
                raise KeyError(f"col_id_of_var({var_name!r}, {dims!r}): matched {sel.height} rows")
            return int(sel["col_id"][0])
        if isinstance(dims, dict):
            mask = pl.lit(True)
            for d, val in dims.items():
                if d not in v.dims:
                    raise ValueError(f"col_id_of_var({var_name!r}): {d!r} not in dims {v.dims}")
                mask = mask & (pl.col(d) == val)
            return f.filter(mask)["col_id"].to_numpy()
        raise TypeError(f"dims must be None, tuple or dict, got {type(dims).__name__}")

    def row_id_of_cstr(self, cstr_name: str, axis: tuple | dict | None = None) -> int | np.ndarray:
        """Return the row_id(s) for a constraint family.  Mirrors
        :meth:`col_id_of_var`."""
        self._require_built()
        meta = self._cstr_meta[cstr_name]
        base = meta["base_row"]
        over = meta["over"]
        if over is None:
            return int(base)
        if axis is None:
            return np.arange(base, base + over.height, dtype=np.int64)
        if isinstance(axis, tuple):
            cols = list(over.columns)
            if len(axis) != len(cols):
                raise ValueError(
                    f"row_id_of_cstr({cstr_name!r}): expected "
                    f"{len(cols)} axis values, got {len(axis)}"
                )
            mask = pl.lit(True)
            for d, val in zip(cols, axis):
                mask = mask & (pl.col(d) == val)
            with_rid = over.with_columns(_rid=pl.int_range(0, over.height, dtype=pl.Int64))
            sel = with_rid.filter(mask)
            if sel.height != 1:
                raise KeyError(
                    f"row_id_of_cstr({cstr_name!r}, {axis!r}): matched {sel.height} rows"
                )
            return int(base + sel["_rid"][0])
        if isinstance(axis, dict):
            mask = pl.lit(True)
            for d, val in axis.items():
                if d not in over.columns:
                    raise ValueError(
                        f"row_id_of_cstr({cstr_name!r}): {d!r} not in over {over.columns}"
                    )
                mask = mask & (pl.col(d) == val)
            with_rid = over.with_columns(_rid=pl.int_range(0, over.height, dtype=pl.Int64))
            return base + with_rid.filter(mask)["_rid"].to_numpy().astype(np.int64)
        raise TypeError(f"axis must be None, tuple or dict, got {type(axis).__name__}")

    # -- solve -----------------------------------------------------------

    def solve(self, *, options: dict | None = None, retry_on_unknown: bool = False) -> Solution:
        """Solve the LP.  First call builds the LP from scratch (same
        pipeline as :meth:`Problem.solve`); subsequent calls just run
        HiGHS again on the (possibly updated) live model.

        ``options`` is honoured on the FIRST solve only — subsequent
        solves use the same HiGHS instance.  To change options on a
        rebuilt LP, drop the WarmProblem and create a new one.

        ``retry_on_unknown`` (default ``False`` → byte-identical to the
        legacy path for every existing caller) enables a warm-restart
        retry: the solver runs WARM first (retaining the basis so dual
        simplex hot-starts after rows appended via :meth:`add_cut_row`);
        only if the warm run does NOT return a certified ``kOptimal`` —
        i.e. a stale-basis transient (kUnknown / kSolveError) or a
        spurious kUnbounded / kInfeasible / primal-infeasible miss — do we
        drop the basis with ``clearSolver()`` and re-run ONCE (the proven
        cold fallback) to re-certify the true status.  Used by the Benders
        master, which appends a cut each iteration; on a well-scaled
        master the warm path stays ``kOptimal`` and the fallback never
        fires.
        """
        if self._h is None:
            self._initial_build(options=options)
        # subsequent: just rerun
        h = self._h
        # Idempotent: installs once on the first solve, no-ops on reuse.
        route_highs_log_to_stdout(h)
        h.run()
        if retry_on_unknown and h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            # The warm re-solve did NOT return a certified optimum.  Appending
            # a >= cut row off a retained basis can leave HiGHS in a transient
            # non-optimal state — kUnknown / kSolveError (status not determined),
            # but also a spurious kUnbounded / kInfeasible / primal-infeasible
            # kOptimal-miss off the stale basis.  Drop the basis with
            # ``clearSolver()`` and re-presolve ONCE from cold (the proven
            # fallback): this re-certifies the true status.  Callers that expect
            # optimality (the Benders master) thus pay the cold cost only when
            # the warm path failed to certify, never on the hot common case.
            h.clearSolver()
            h.run()
        return self._build_solution(h)

    def _build_solution(self, h) -> Solution:
        """Read the live HiGHS handle into a :class:`Solution` (shared by
        the warm path and the cold-fallback retry in :meth:`solve`)."""
        sol = h.getSolution()
        status_ok = h.getModelStatus() == highspy.HighsModelStatus.kOptimal
        col_value = np.asarray(sol.col_value, dtype=np.float64)
        row_dual = (
            np.asarray(sol.row_dual, dtype=np.float64) if sol.row_dual else np.zeros(self._n_rows)
        )
        col_dual = (
            np.asarray(sol.col_dual, dtype=np.float64) if sol.col_dual else np.zeros(self._n_cols)
        )
        return Solution(
            optimal=status_ok,
            obj=h.getObjectiveValue(),
            col_value=col_value,
            row_dual=row_dual,
            col_dual=col_dual,
            col_names=self._col_names,
            row_names=self._row_names,
            vars=dict(self._p._vars),
            highs=h,
        )

    # -- internals -------------------------------------------------------

    def _require_built(self) -> None:
        if self._h is None:
            raise RuntimeError("WarmProblem: must call solve() once before update_*().")

    @staticmethod
    def _resolve_rhs_vec(new_param, meta: dict, cstr_name: str) -> np.ndarray:
        """Compute a length-row_count RHS vector aligned to the constraint's
        original ``over`` axis order.

        ``meta`` is the WarmProblem._cstr_meta entry; we read ``over``,
        ``row_count`` and (lazily) cache a per-Param-dim-set lookup index
        so subsequent calls with the same dim signature avoid the polars
        join.
        """
        over: pl.DataFrame | None = meta["over"]
        row_count: int = meta["row_count"]

        if isinstance(new_param, (int, float)):
            return np.full(row_count, float(new_param), dtype=np.float64)
        if isinstance(new_param, np.ndarray):
            if new_param.size != row_count:
                raise ValueError(
                    f"update_rhs({cstr_name!r}): array length "
                    f"{new_param.size} != row count {row_count}"
                )
            return new_param.astype(np.float64, copy=False)
        if isinstance(new_param, Param):
            if over is None:
                return np.full(row_count, float(new_param.frame["value"][0]), dtype=np.float64)
            missing = [d for d in new_param.dims if d not in over.columns]
            if missing:
                raise ValueError(
                    f"update_rhs({cstr_name!r}): new_param has "
                    f"dims {missing} not in over "
                    f"{over.columns}"
                )
            if not new_param.dims:
                return np.full(row_count, float(new_param.frame["value"][0]), dtype=np.float64)
            # Hot path: most rolling-horizon callers pass a Param whose
            # frame is already in ``over``-row order with the same dim
            # values.  Detect that and skip the join entirely.
            on = list(new_param.dims)
            param_frame = new_param.frame
            if (
                param_frame.height == row_count
                and len(on) == len(over.columns)
                and set(on) == set(over.columns)
            ):
                # Cheap structural check: are the dim columns row-equal?
                # We hash the comparison once per (cstr, param_dims) and
                # cache the result.
                cache_key = ("_rhs_inplace_id", new_param.dims, id(over))
                inplace = meta.get(cache_key)
                if inplace is None:
                    # Reorder param frame to over column order, then check
                    # row-by-row equality.  np.array_equal on numpy views
                    # is the fast path.
                    pf = param_frame.select(*over.columns).to_numpy()
                    of = over.select(*over.columns).to_numpy()
                    inplace = bool(pf.shape == of.shape and (pf == of).all())
                    meta[cache_key] = inplace
                if inplace:
                    return param_frame["value"].to_numpy().astype(np.float64, copy=False)
            # Fallback — join on dims; left-join preserves over-frame
            # order so the resulting "value" column is row-aligned.
            over_a, pf_a = _align_enum_join_keys(over, param_frame, on)
            j = over_a.join(pf_a, on=on, how="left")
            return j["value"].fill_null(0.0).to_numpy().astype(np.float64, copy=False)
        raise TypeError(
            f"update_rhs({cstr_name!r}): unsupported new_param type {type(new_param).__name__}"
        )

    def _initial_build(self, *, options: dict | None) -> None:
        """Run the same pipeline as :meth:`Problem.solve` up to
        ``passModel``.  Stores ``self._h`` and the row/col metadata maps
        for later updates.

        Stage B3: the bulk LP build (LHS CSC, RHS, obj, bounds) is now
        delegated to :meth:`Problem.canonicalise` — the same canonical
        matrix consumed by ``write_mps`` and ``_build_lp_arrays``.  The
        Layer-2 side vectors are already baked into ``m.val`` /
        ``m.col_obj`` / ``m.row_lb`` / ``m.row_ub`` per orchestrator
        decision D8, so this method does NOT re-multiply.

        A SEPARATE second pass over ``_cstrs`` rebuilds the tracked-
        source ``_param_cells`` map.  Only terms with a non-empty
        ``param_sources`` are touched; the per-term collect cost is
        bounded by the (typically tiny) declared-mutable set.  The
        factor cached per cell is the SCALED coef so that
        ``update_param`` 's ``factor * new_value^pdir`` formula
        reproduces the post-Layer-2 matrix entry — matching the
        semantics the pre-B3 code maintained inline.
        """
        p = self._p
        _sp_emit, _sp_on = _make_solve_profile_emitter()
        if _sp_on:
            _sp_emit(
                "initial_build_enter",
                n_vars=len(p._vars),
                n_cstrs=len(p._cstrs),
                n_obj_terms=len(p._obj_terms),
                n_mutable_params=len(self._mutable_params),
            )

        # ---- Step 1: bulk LP arrays from the canonical matrix. -----
        # NOTE: internal POLAR_HIGH_WRITE_MPS_PROFILE-gated checkpoints
        # inside ``_build_canonical_matrix`` fire when that env var is
        # also set; we mark the boundary here so the [solve profile]
        # stream brackets the canonicalise cliff cleanly.
        if _sp_on:
            _sp_emit("canonicalise_enter")
        m = p.canonicalise()
        n_cols = m.n_cols
        n_rows = m.n_rows
        if _sp_on:
            _sp_emit(
                "canonicalise_returned",
                n_rows=n_rows,
                n_cols=n_cols,
                nnz=int(m.val.size),
            )

        inf = highspy.kHighsInf
        col_lb_h = np.where(m.col_lb == -np.inf, -inf, m.col_lb).astype(np.float64, copy=False)
        col_ub_h = np.where(m.col_ub == np.inf, inf, m.col_ub).astype(np.float64, copy=False)
        row_lb_h = np.where(m.row_lb == -np.inf, -inf, m.row_lb).astype(np.float64, copy=False)
        row_ub_h = np.where(m.row_ub == np.inf, inf, m.row_ub).astype(np.float64, copy=False)
        if _sp_on:
            _sp_emit("bounds_translated", n_cols=n_cols, n_rows=n_rows)

        # Populate _var_cols (still needed by update_obj_coef / fix_cols /
        # _resolve_dim_tuples).  Cheap: one int64 array per declared var.
        for v in p._vars.values():
            ids = v.frame["col_id"].to_numpy()
            self._var_cols[v.name] = ids.astype(np.int64)
        if _sp_on:
            _sp_emit("var_cols_populated", n_vars=len(p._vars))

        # Use the canonical matrix's col_names verbatim.  Unused slots
        # are empty strings (vs the pre-B3 ``None``); ``passColName`` is
        # skipped for those — same observable effect as the old loop.
        col_names: list[str] = list(m.col_names)
        row_names: list[str] = list(m.row_names)
        if _sp_on:
            _sp_emit(
                "names_snapshotted",
                n_col_names=len(col_names),
                n_row_names=len(row_names),
            )

        # ---- Step 2: per-cstr metadata + tracked-source second pass.
        # The bulk path doesn't need any of this; both loops only fire
        # for warm-update bookkeeping.
        #
        # ``_cstr_meta`` carries the user-declared family shape needed
        # by ``update_rhs`` (and friends).  ``_param_cells`` maps the
        # declared-mutable Params to their LP cells.  Both reuse the
        # same family-walk arithmetic as ``_build_canonical_matrix`` —
        # we only redo the small per-tracked-term collect, NOT the full
        # LHS sweep.
        _rf = p._layer2_row_factor
        _cf = p._layer2_col_factor
        track_acc: dict[str, list[dict]] = {}
        next_row = 0
        if _sp_on:
            _sp_emit(
                "cstr_meta_loop_start",
                n_cstrs=len(p._cstrs),
                n_mutable_params=len(self._mutable_params),
                has_row_factor=int(_rf is not None),
                has_col_factor=int(_cf is not None),
            )
        # Per-family tracked-work counters (only emitted for families
        # that actually ran the second-pass collect — avoids stderr
        # flooding when tracked work is sparse).  Counts at the
        # checkpoint capture the family's cumulative tracked rows so
        # the ramp is attributable.
        _fam_tracked_total = 0
        _fam_tracked_families = 0
        for _fam_idx, (cname, proto, over) in enumerate(p._cstrs):
            expr, sense = proto.expr, proto.sense

            if over is None:
                row_count = 1
                row_index = pl.DataFrame({"_rid": [0]})
                axis_cols: list[str] = []
            else:
                row_count = int(over.height)
                axis_cols = list(over.columns)
                row_index = over.with_columns(_rid=pl.int_range(0, over.height, dtype=pl.Int64))

            base_row = next_row
            next_row += row_count

            # Stash family metadata (mirrors the pre-B3 cstr_meta dict).
            self._cstr_meta[cname] = {
                "base_row": int(base_row),
                "row_count": int(row_count),
                "sense": sense,
                "over": over,  # may be None for scalar cstr
                "axis_cols": tuple(axis_cols),
            }

            # If no tracked Params are declared OR none of this family's
            # terms reference a mutable one, skip the second-pass collect
            # entirely (zero overhead vs stock Problem.solve).
            if not self._mutable_params:
                continue

            # Per-family tracked-row counter — incremented inside the
            # term loop so the post-family checkpoint can attribute
            # rows added by this family to the track_acc accumulator.
            _fam_tracked_rows = 0
            row_index_lf = row_index.lazy()
            for term in expr.terms:
                if not term.param_sources:
                    continue
                tracked_sources: list[tuple[Param, int]] = [
                    (pobj, pdir)
                    for pobj, pdir in term.param_sources
                    if pobj.name in self._mutable_params
                ]
                if not tracked_sources:
                    continue

                if term.dims:
                    missing = [d for d in term.dims if d not in axis_cols]
                    if missing:
                        # The canonical matrix build already raised on
                        # this; if we got here something is wildly out of
                        # sync.  Reraise with the same error shape.
                        raise ValueError(
                            f"constraint {cname!r}: term has open dims "
                            f"{term.dims}, but constraint axes are "
                            f"{axis_cols}; aggregate {missing} via Sum() "
                            f"before adding."
                        )
                    on = [d for d in term.dims if d in axis_cols]
                    # Prefer LHS prune-down when this term carries both a
                    # Var reference and a multi-atomic Param chain — same
                    # criteria as _build_canonical_matrix / _solve_streaming.
                    # Sum-collapsed terms (var_source=None) fall back to
                    # the original merged-lazy semi-join path verbatim so
                    # the warm-path output matches the canonical build.
                    _lhs_psrc = term.param_sources if isinstance(term.param_sources, list) else None
                    _use_lhs_prune = (
                        term.var_source is not None
                        and _lhs_psrc is not None
                        and len(_lhs_psrc) >= 2
                        and not _prune_down_disabled()
                    )
                    # Block-COO sibling arm — identical dispatch to Sites 1
                    # (_build_canonical_matrix) and 2 (_solve_streaming),
                    # adapted to the warm path.  Fires ONLY when the source
                    # Problem declared dense_axes AND this non-Sum Var×Param
                    # chain matches the dense-suffix contract; default ON
                    # (POLAR_HIGH_DISABLE_BLOCK_COO=1 is the off switch);
                    # bit-identical to the polars path; any shape it can't
                    # reproduce returns None and we fall through to the
                    # UNCHANGED prune-down / semi-join warm arms.
                    #
                    # The warm path needs the per-term dims on the emitted
                    # frame (the prune/fallback arms select ``*term.dims``)
                    # because the downstream param-TRACKING re-joins each
                    # tracked Param on its dims for incremental update_param.
                    # We therefore call the helper with
                    # ``keep_dims=tuple(term.dims)`` so the returned frame
                    # carries ``_rid, col_id, coef, *term.dims`` — the exact
                    # shape ``j = _collect_streaming(plan)`` consumes.  The
                    # helper returns a collected frame whose ``_rid`` is
                    # family-local (it joined the local int-range
                    # ``row_index_lf``), matching the ``abs_rows = base_row +
                    # rids`` bake below; we ``.lazy()`` it so it flows through
                    # ``_collect_streaming`` like the other warm arms.
                    _block_spec = None
                    if not _block_coo_disabled():
                        _block_spec = _block_coo_classify(term, axis_cols, on, p._dense_axes)
                    # See _build_canonical_matrix: a deferred map-effect
                    # Where cannot be carried by the block-COO seed —
                    # bake-before-block (route to prune / fallback) keeps
                    # the warm path byte-identical until Phase C-3.
                    if term.where_map_frames is not None:
                        _block_spec = None
                    # Sum-block-COO sibling arm for the warm path (Site 3).
                    # The warm tracker re-joins each tracked Param on its
                    # dims against the emitted (_rid, col_id, coef,
                    # *term.dims) frame to recover the per-cell old value and
                    # cache factor = coef / old_value (numerator) so
                    # ``update_param`` can recompute factor * new_value.  That
                    # model is exact ONLY when each emitted cell's coef is a
                    # SINGLE product linear in the tracked Param's value AND
                    # every tracked Param's dims are recoverable from the
                    # emitted ``*term.dims`` (= keep) columns.  So we admit
                    # the Sum-block arm here ONLY for the RELABEL shape
                    # (reduce_dims ⊆ var.dims ⇒ every reduce group is
                    # single-element ⇒ coef is one product, bit-identical to
                    # the reduced path) AND only when every tracked Param's
                    # dims ⊆ keep AND keep carries no map-introduced extra
                    # (keep ⊆ var.dims), so the tracker's re-join keys and the
                    # cached factor reproduce the cell exactly.  The combining
                    # shape (coef is a SUM over a reduced dim) and any
                    # tracked Param keyed on a reduced/map dim would make
                    # factor * new_value wrong, so they fall through to the
                    # UNCHANGED reduced ``term.lazy`` warm path (its tracking
                    # is the guaranteed-correct fallback).  Same off switch as
                    # the non-Sum arm.
                    _sum_block_spec = None
                    if _block_spec is None and not _block_coo_disabled():
                        _sum_block_spec = _sum_block_coo_classify(
                            term, axis_cols, on, p._dense_axes
                        )
                        if _sum_block_spec is not None:
                            _keep_set = set(_sum_block_spec["keep"])
                            _var_dims_set = set(_sum_block_spec["var_dims"])
                            _relabel = set(_sum_block_spec["reduce_dims"]).issubset(_var_dims_set)
                            _keep_in_var = _keep_set.issubset(_var_dims_set)
                            _tracked_ok = all(
                                set(pobj.dims).issubset(_keep_set)
                                for pobj, _pdir in tracked_sources
                            )
                            if not (_relabel and _keep_in_var and _tracked_ok):
                                # Warm-tracker-unsafe ⇒ decline; the reduced
                                # ``term.lazy`` warm path tracks it correctly.
                                _sum_block_spec = None
                    if _block_spec is not None:
                        _verify_dense_sorted(
                            term.var_source.frame,
                            _block_spec["non_dense_dims"],
                            _block_spec["dense_dims"],
                            getattr(term.var_source, "name", None),
                        )
                        _t_blk0 = time.monotonic()
                        _blk_df = _build_block_coo_plan(
                            row_index_lf,
                            axis_cols,
                            term.var_source,
                            _lhs_psrc,
                            on,
                            term.coef_scalar,
                            term.where_frames,
                            _block_spec,
                            keep_dims=tuple(term.dims),
                        )
                        if os.environ.get("POLAR_HIGH_BLOCK_COO_PROFILE") == "1":
                            _blk_wall = time.monotonic() - _t_blk0
                            _n_rows = int(_blk_df.height)
                            _dense = _block_spec["dense_dims"]
                            _nb = _block_spec["dense_card"]
                            _avg = (_n_rows / _nb) if _nb else 0.0
                            sys.stderr.write(
                                f"[block_coo profile]\tphase=block_coo_term"
                                f"\tphase_site=warm"
                                f"\tfamily={cname}\tfamily_idx={_fam_idx}"
                                f"\tterm_idx=0"
                                f"\tdense_dims={','.join(_dense)}"
                                f"\tn_blocks={_nb}"
                                f"\tavg_block_size={_avg:.2f}"
                                f"\twall_s={_blk_wall:.4f}\n"
                            )
                            sys.stderr.flush()
                        plan = _blk_df.lazy()
                    elif _sum_block_spec is not None:
                        # Relabel-shape Sum-block term, tracker-safe (gated
                        # above).  Build with keep_dims=tuple(term.dims) so
                        # the emitted frame carries (_rid, col_id, coef,
                        # *term.dims) — the exact shape the warm tracker's
                        # re-join + ``abs_rows = base_row + rids`` path below
                        # consumes.  A fallback sentinel ⇒ reduced
                        # ``term.lazy`` warm path verbatim.
                        _sm = term.sum_block_meta
                        _verify_dense_sorted(
                            _sm.var_source.frame,
                            _sum_block_spec["non_dense_dims"],
                            _sum_block_spec["dense_dims"],
                            getattr(_sm.var_source, "name", None),
                        )
                        _t_blk0 = time.monotonic()
                        try:
                            _blk_df = _build_sum_block_coo_plan(
                                row_index_lf,
                                axis_cols,
                                _sm,
                                on,
                                _sum_block_spec,
                                keep_dims=tuple(term.dims),
                            )
                            _sum_block_fired = True
                        except _SumBlockCooFallback:
                            _sum_block_fired = False
                        if _sum_block_fired:
                            if os.environ.get("POLAR_HIGH_BLOCK_COO_PROFILE") == "1":
                                _blk_wall = time.monotonic() - _t_blk0
                                _n_rows = int(_blk_df.height)
                                _dense = _sum_block_spec["dense_dims"]
                                _nb = _sum_block_spec["dense_card"]
                                _avg = (_n_rows / _nb) if _nb else 0.0
                                sys.stderr.write(
                                    f"[block_coo profile]\tphase=block_coo_term"
                                    f"\tkind=sum\tphase_site=warm"
                                    f"\tfamily={cname}\tfamily_idx={_fam_idx}"
                                    f"\tterm_idx=0"
                                    f"\tdense_dims={','.join(_dense)}"
                                    f"\tn_blocks={_nb}"
                                    f"\tavg_block_size={_avg:.2f}"
                                    f"\twall_s={_blk_wall:.4f}\n"
                                )
                                sys.stderr.flush()
                            plan = _blk_df.lazy()
                        else:
                            # Reduced-``term.lazy`` fallback — identical to
                            # the final else arm (bake deferred filters, then
                            # the row_index semi-join + inner-join carrying
                            # ``*term.dims`` for the tracker re-join).
                            term_lazy_filtered = _apply_where_frames(
                                term.lazy, term.dims, term.where_frames
                            )
                            term_lazy_filtered, _ = _apply_where_map_frames(
                                term_lazy_filtered,
                                term.dims,
                                term.where_map_frames,
                            )
                            rl_a, tl_a = _align_enum_join_keys(row_index_lf, term_lazy_filtered, on)
                            keys_lazy = rl_a.select(on).unique()
                            tl_pruned = tl_a.join(keys_lazy, on=on, how="semi")
                            plan = rl_a.join(tl_pruned, on=on, how="inner").select(
                                "_rid", "col_id", "coef", *term.dims
                            )
                    elif _use_lhs_prune:
                        plan = _build_lhs_pruned_plan(
                            row_index_lf,
                            axis_cols,
                            term.var_source,
                            _lhs_psrc,
                            on,
                            coef_scalar=term.coef_scalar,
                            where_frames=term.where_frames,
                            where_map_frames=term.where_map_frames,
                        ).select("_rid", "col_id", "coef", *term.dims)
                        if _sp_on:
                            _sp_emit(
                                "family_term_pruned_down",
                                family=cname,
                                family_idx=_fam_idx,
                                term_idx=0,
                                n_atomics=len(_lhs_psrc),
                            )
                    else:
                        # Bake any deferred Where filters before the
                        # semi-join — fallback path applies them since
                        # prune-down isn't firing here.  Pure-filter then
                        # map-effect (dim-extending), so ``*term.dims``
                        # (which now includes any extras) is satisfied.
                        term_lazy_filtered = _apply_where_frames(
                            term.lazy, term.dims, term.where_frames
                        )
                        term_lazy_filtered, _ = _apply_where_map_frames(
                            term_lazy_filtered, term.dims, term.where_map_frames
                        )
                        rl_a, tl_a = _align_enum_join_keys(row_index_lf, term_lazy_filtered, on)
                        keys_lazy = rl_a.select(on).unique()
                        tl_pruned = tl_a.join(keys_lazy, on=on, how="semi")
                        plan = rl_a.join(tl_pruned, on=on, how="inner").select(
                            "_rid", "col_id", "coef", *term.dims
                        )
                    j = _collect_streaming(plan)
                    if j.height == 0:
                        continue
                    rids = j["_rid"].to_numpy().astype(np.int64, copy=False)
                    cids = j["col_id"].to_numpy().astype(np.int64, copy=False)
                    coefs = j["coef"].to_numpy().astype(np.float64, copy=False)
                    abs_rows = base_row + rids
                    _fam_tracked_rows += int(coefs.size)
                    # The Param-tracker cache stores the SCALED coef so
                    # subsequent mutate-and-resolve cycles produce values
                    # consistent with what ``_build_canonical_matrix``
                    # BAKED into ``m.val`` (Layer 2 side vectors).  Apply
                    # the same row/column factors here at second-pass
                    # collect time so the per-term factor matches the
                    # canonical assembly.
                    if _rf is not None:
                        coefs = coefs * _rf[abs_rows]
                    if _cf is not None:
                        coefs = coefs * _cf[cids]
                    term_dims = tuple(term.dims)
                    for pobj, pdir in tracked_sources:
                        pname = pobj.name
                        pdims = pobj.dims
                        missing_d = [d for d in pdims if d not in term_dims]
                        if missing_d:
                            continue
                        if pdims:
                            keys_df = j.select(*pdims)
                            _kd_lhs = keys_df.with_row_index("__ridx")
                            _pdims_on = list(pdims)
                            _kd_lhs, _pf_rhs = _align_enum_join_keys(_kd_lhs, pobj.frame, _pdims_on)
                            joined = _kd_lhs.join(_pf_rhs, on=_pdims_on, how="left").sort("__ridx")
                            old_vals = (
                                joined["value"]
                                .fill_null(0.0)
                                .to_numpy()
                                .astype(np.float64, copy=False)
                            )
                            if pdir == 1:
                                safe = np.where(old_vals == 0.0, 1.0, old_vals)
                                factor = np.where(old_vals == 0.0, 0.0, coefs / safe)
                            else:
                                factor = coefs * old_vals
                            track_acc.setdefault(pname, []).append(
                                dict(
                                    rows=abs_rows.copy(),
                                    cols=cids.copy(),
                                    dim_keys=keys_df,
                                    factor=factor,
                                    direction=np.full(coefs.size, pdir, dtype=np.int8),
                                    dim_signature=tuple(pdims),
                                )
                            )
                        else:
                            old_v = float(pobj.frame["value"][0])
                            if pdir == 1:
                                if old_v == 0.0:
                                    factor = np.zeros_like(coefs)
                                else:
                                    factor = coefs / old_v
                            else:
                                factor = coefs * old_v
                            track_acc.setdefault(pname, []).append(
                                dict(
                                    rows=abs_rows.copy(),
                                    cols=cids.copy(),
                                    dim_keys=None,
                                    factor=factor,
                                    direction=np.full(coefs.size, pdir, dtype=np.int8),
                                    dim_signature=(),
                                )
                            )
                else:
                    # Scalar (collapsed) tracked terms are emitted into
                    # the matrix but ``update_param`` can't key off them
                    # (no dim columns); mirror the pre-B3 behaviour and
                    # skip _param_cells registration.
                    continue

            # Per-family checkpoint — only fires for families that
            # actually accumulated tracked rows.  Keeps stderr volume
            # bounded on large LPs where most families have no tracked
            # terms, while still surfacing the per-family cost when the
            # ramp is concentrated.
            if _sp_on and _fam_tracked_rows:
                _fam_tracked_total += _fam_tracked_rows
                _fam_tracked_families += 1
                _sp_emit(
                    "cstr_family_tracked",
                    fam_idx=_fam_idx,
                    family=cname,
                    row_count=row_count,
                    tracked_rows=_fam_tracked_rows,
                    cum_tracked_rows=_fam_tracked_total,
                )

        if _sp_on:
            _sp_emit(
                "cstr_meta_loop_done",
                n_cstrs=len(p._cstrs),
                tracked_families=_fam_tracked_families,
                tracked_rows_total=_fam_tracked_total,
                track_acc_params=len(track_acc),
            )

        # Consolidate per-Param tracking accumulators.
        for pname, chunks in track_acc.items():
            if not chunks:
                continue
            sig = chunks[0]["dim_signature"]
            same_sig = all(c["dim_signature"] == sig for c in chunks)
            if not same_sig:
                continue
            rows = np.concatenate([c["rows"] for c in chunks])
            cols = np.concatenate([c["cols"] for c in chunks])
            factor = np.concatenate([c["factor"] for c in chunks])
            direction = np.concatenate([c["direction"] for c in chunks])
            if sig:
                dim_keys = pl.concat([c["dim_keys"] for c in chunks])
            else:
                dim_keys = None
            self._param_cells[pname] = dict(
                rows=rows,
                cols=cols,
                factor=factor,
                direction=direction,
                dim_signature=sig,
                dim_keys=dim_keys,
            )

        if _sp_on:
            _sp_emit(
                "track_acc_consolidated",
                n_param_cells=len(self._param_cells),
            )

        # ---- Step 3: assemble HighsLp + passModel. ------------------
        # CSC arrays come straight from the canonical matrix; their
        # dtypes (int32 or int64 based on nnz) were chosen there.
        lp = highspy.HighsLp()
        lp.num_col_ = int(n_cols)
        lp.num_row_ = int(n_rows)
        lp.col_cost_ = m.col_obj.astype(np.float64, copy=False)
        lp.col_lower_ = col_lb_h
        lp.col_upper_ = col_ub_h
        lp.row_lower_ = row_lb_h
        lp.row_upper_ = row_ub_h
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.num_col_ = int(n_cols)
        lp.a_matrix_.num_row_ = int(n_rows)
        lp.a_matrix_.start_ = m.col_ptr
        lp.a_matrix_.index_ = m.row_idx
        lp.a_matrix_.value_ = m.val
        lp.sense_ = (
            highspy.ObjSense.kMaximize if p._obj_sense == "max" else highspy.ObjSense.kMinimize
        )
        if p._obj_offset:
            lp.offset_ = float(p._obj_offset)
        if m.col_int.any():
            kCont = highspy.HighsVarType.kContinuous
            kInt = highspy.HighsVarType.kInteger
            integ_arr = np.where(m.col_int, kInt, kCont)
            lp.integrality_ = integ_arr.tolist()
        if _sp_on:
            _sp_emit(
                "lp_assembled",
                n_cols=int(n_cols),
                n_rows=int(n_rows),
                nnz=int(m.val.size),
                has_integers=int(bool(m.col_int.any())),
            )

        h = highspy.Highs()
        if _sp_on:
            _sp_emit("highs_constructed")
        opts = options if options is not None else p._solver_options
        # See the streaming-solve block — tear the global scheduler
        # down before re-applying ``threads`` / ``parallel`` on the
        # warm-built Highs.
        if opts and ("threads" in opts or "parallel" in opts):
            try:
                h.resetGlobalScheduler(False)
            except Exception:
                pass
        if opts:
            import warnings

            ok_status = getattr(highspy.HighsStatus, "kOk", None)
            for key, val in opts.items():
                try:
                    status = h.setOptionValue(key, val)
                except Exception as exc:
                    warnings.warn(f"HiGHS rejected option {key}={val!r}: {exc}", stacklevel=2)
                    continue
                if ok_status is not None and status != ok_status:
                    warnings.warn(
                        f"HiGHS rejected option {key}={val!r} (status={status!r})", stacklevel=2
                    )
        if _sp_on:
            _sp_emit(
                "highs_options_applied",
                n_opts=(len(opts) if opts else 0),
            )
        # Apply a set_output_flag() preference BEFORE routing the log / the
        # first log call: HiGHS emits the version banner on its first log call
        # (which ``route_highs_log_to_stdout`` can trigger pre-run), so a False
        # preference set here suppresses the banner too — not just the solve
        # log.  The build's own options dict was already applied above; this
        # persistent caller preference overrides it.
        if self._output_flag is not None:
            h.setOptionValue("output_flag", self._output_flag)
        # Route HiGHS' log through Python ``sys.stdout`` before passModel —
        # the version banner can be emitted on the first log call (pre-run),
        # so register early to capture it under Jupyter / Spine-Toolbox on
        # Windows.  Idempotent: the per-solve ``run()`` call site re-checks.
        route_highs_log_to_stdout(h)
        if _sp_on:
            _sp_emit("highs_passmodel_pre", n_cols=int(n_cols), n_rows=int(n_rows))
        h.passModel(lp)
        if _sp_on:
            _sp_emit("highs_passmodel_post")
        for i, n in enumerate(col_names):
            # Canonical matrix uses "" for unused col slots; skip those.
            if n:
                h.passColName(i, n)
        if _sp_on:
            _sp_emit("highs_col_names_passed", n_col_names=len(col_names))
        for i, n in enumerate(row_names):
            # HiGHS' pybind11 binding requires ``str``; the canonical
            # matrix may carry ``None`` (or ``""``) for unnamed rows.
            # Fall back to a synthetic ``row_<idx>`` so the API contract
            # is satisfied and diagnostics remain useful.
            h.passRowName(i, n if n else f"row_{i}")
        if _sp_on:
            _sp_emit("highs_row_names_passed", n_row_names=len(row_names))

        self._h = h
        self._n_cols = int(n_cols)
        self._n_rows = int(n_rows)
        self._col_names = col_names
        self._row_names = row_names

        # (output_flag preference is applied earlier — before the log routing —
        # so it suppresses the version banner, not just the solve log.)

        # Release the per-term ``SumBlockMeta`` recipe now the build has
        # consumed it.  A ``Sum``-reduced term survivor-filters its own
        # ``param_sources`` (dropping the summed-out factors), but the
        # captured recipe snapshots the FULL pre-Sum chain — pinning the
        # summed-out dense ``(d,t)`` Params (and their eager source frames)
        # for this WarmProblem's whole lifetime.  This build is the
        # recipe's LAST reader (the matrix-assembly block-COO Sum path
        # above + every autoscale readout that ran before it); the warm
        # update machinery (``update_param`` / the tracked-source second
        # pass) keys off ``term.param_sources`` and ``self._param_cells``,
        # NOT the recipe, so dropping it cannot perturb a warm update.
        # WarmProblem never calls ``_release_python_lp_inputs`` (the
        # save_memory cold path), so without this the recipe ratchets dense
        # Params up across rolls until OOM.
        for t in p._obj_terms:
            t.sum_block_meta = None
        for _name, _proto, _over in p._cstrs:
            for t in _proto.expr.terms:
                t.sum_block_meta = None

        if _sp_on:
            _sp_emit("initial_build_exit", n_cols=int(n_cols), n_rows=int(n_rows))
