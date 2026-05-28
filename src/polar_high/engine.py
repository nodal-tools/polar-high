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

import highspy
import numpy as np
import polars as pl

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

    __slots__ = ("dims", "lazy", "_frame_cache", "name", "_sources")

    def __init__(
        self,
        dims: tuple[str, ...],
        frame: pl.DataFrame | pl.LazyFrame,
        name: str | None = None,
        _sources: list[tuple[Param, int]] | None = None,
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
            )
        return NotImplemented

    def _sources_for_propagation(self) -> list[tuple[Param, int]] | None:
        """Like :meth:`_own_sources` but returns ``None`` for an anonymous
        Param with no sub-sources — saves an allocation in the common
        unnamed-Param case."""
        if self._sources is not None:
            return self._sources
        if self.name is not None:
            return [(self, 1)]
        return None

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
        return Expr([_Term(f, self.dims)])

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            f = (
                self.frame.lazy()
                .with_columns(coef=pl.lit(float(other)))
                .select(*self.dims, "col_id", "coef")
            )
            return Expr([_Term(f, self.dims)])
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
            return Expr([_Term(j, new_dims, param_sources=psrc)])
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
    """

    __slots__ = ("lazy", "dims", "param_sources")

    def __init__(
        self,
        lazy: pl.LazyFrame | pl.DataFrame,
        dims: tuple[str, ...],
        param_sources: list[tuple[Param, int]] | None = None,
    ):
        if isinstance(lazy, pl.DataFrame):
            lazy = lazy.lazy()
        self.lazy = lazy
        self.dims = tuple(dims)
        self.param_sources = param_sources

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
        neg = [
            _Term(t.lazy.with_columns(coef=-pl.col("coef")), t.dims, param_sources=t.param_sources)
            for t in _to_expr(other).terms
        ]
        return Expr(self.terms + neg)

    def __radd__(self, other):
        return _to_expr(other) + self

    def __rsub__(self, other):
        return _to_expr(other) - self

    def __neg__(self):
        return self * -1.0

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Expr(
                [
                    _Term(
                        t.lazy.with_columns(coef=pl.col("coef") * float(scalar)),
                        t.dims,
                        param_sources=t.param_sources,
                    )
                    for t in self.terms
                ]
            )
        if isinstance(scalar, Param):
            psrc_other = scalar._sources_for_propagation()
            new = []
            for t in self.terms:
                shared = [d for d in t.dims if d in scalar.dims]
                new_dims = tuple(dict.fromkeys(t.dims + scalar.dims))
                if shared:
                    left_lf, right_lf = _align_enum_join_keys(t.lazy, scalar.lazy, shared)
                    j = left_lf.join(right_lf, on=shared, how="inner")
                else:
                    j = t.lazy.join(scalar.lazy, how="cross")
                j = j.with_columns(coef=pl.col("coef") * pl.col("value")).select(
                    *new_dims, "col_id", "coef"
                )
                merged = _merge_param_sources(t.param_sources, psrc_other, flip_other=False)
                new.append(_Term(j, new_dims, param_sources=merged))
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
                shared = [d for d in t.dims if d in other.dims]
                new_dims = tuple(dict.fromkeys(t.dims + other.dims))
                if shared:
                    left_lf, right_lf = _align_enum_join_keys(t.lazy, other.lazy, shared)
                    j = left_lf.join(right_lf, on=shared, how="inner")
                else:
                    j = t.lazy.join(other.lazy, how="cross")
                j = j.with_columns(coef=pl.col("coef") / pl.col("value")).select(
                    *new_dims, "col_id", "coef"
                )
                merged = _merge_param_sources(t.param_sources, psrc_other, flip_other=True)
                new.append(_Term(j, new_dims, param_sources=merged))
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
        carry = [c for c in lag_cols if c in t.dims and c != time_dim and c != lag_col]
        lagged = t.lazy.rename({time_dim: "_lag_src"})
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
                j.select(*[d for d in t.dims if d != time_dim], time_dim, "col_id", "coef"),
                t.dims,
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
    """
    if isinstance(expr, Var):
        expr = expr.to_expr()
    if isinstance(frame, pl.LazyFrame):
        frame_lf = frame
        frame_cols = frame_lf.collect_schema().names()
    else:
        frame_lf = frame.lazy()
        frame_cols = frame.columns
    new: list[_Term] = []
    for t in expr.terms:
        # Term schema = dims + (col_id, coef).  Only dims overlap with
        # ``frame``'s join keys (col_id is internal).
        term_cols = set(t.dims) | {"col_id", "coef"}
        shared = [c for c in frame_cols if c in term_cols]
        extra = tuple(c for c in frame_cols if c not in term_cols)
        f = t.lazy
        if shared:
            f, frame_lf_a = _align_enum_join_keys(f, frame_lf, shared)
            f = f.join(frame_lf_a, on=shared, how="inner")
        new.append(_Term(f, t.dims + extra, param_sources=t.param_sources))
    return Expr(new)


def Sum(expr, over: tuple[str, ...] | str | None = None, where: pl.DataFrame | None = None) -> Expr:
    """Aggregate an Expr.  ``over`` lists the dims to sum out; the
    remaining dims become the term's open dims.  ``where`` is an index
    frame that pre-filters the term frames (inner join on shared
    columns) before the group-by-sum.

    ``Sum(expr)`` with ``over=None`` collapses every open dim — useful
    for a scalar (objective term, single-row constraint).
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
        f = t.lazy
        if where_lf is not None:
            shared = [c for c in where_cols if c in t.dims]
            if shared:
                where_sub = where_lf.select(shared).unique()
                f, where_sub = _align_enum_join_keys(f, where_sub, shared)
                f = f.join(where_sub, on=shared, how="inner")
        keep = tuple(d for d in t.dims if d not in over)
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
        new_terms.append(_Term(f, keep, param_sources=psrc))
    return Expr(new_terms)


# ---------------------------------------------------------------------------
# Problem container


class Problem:
    """LP container.  Generic — no flextool-specific knowledge."""

    def __init__(self) -> None:
        """Construct an empty LP container.

        Pure polar-high is a generic LP kernel; scaling decisions are
        left to the caller.  See :mod:`polar_high.autoscale` for the
        opt-in autoscaler (Layer 1 detect + Layer 3 recommendation)
        that callers (e.g. FlexTool) use to drive
        ``user_bound_scale`` / ``user_objective_scale`` automatically.
        """
        self._vars: dict[str, Var] = {}
        self._cstrs: list[tuple[str, _CstrProto, pl.DataFrame | None]] = []
        self._next_col = 0
        self._obj_terms: list[_Term] = []
        self._obj_sense = "min"
        self._obj_offset: float = 0.0
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
        frame), and the constraint-family list itself.  Sets
        :attr:`_released` so :meth:`solve` refuses to run again — the
        Problem is no longer re-emittable.
        """
        # Objective terms: drop lazy plans first so any Param objects
        # referenced via ``param_sources`` aren't extended past the
        # constraint walk below.
        for t in self._obj_terms:
            t.lazy = None  # type: ignore[assignment]
            t.param_sources = None
        self._obj_terms = []

        # Constraint families: clear each Expr's term list and drop the
        # rhs reference (which may be a Param holding a sizeable eager
        # frame).  We don't touch ``over`` — it's typically the row-
        # index DataFrame, already small compared to the LHS plans we
        # just dropped, and stripping it would complicate any future
        # diagnostic that wants to report which family came last.
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
                row_index = over.with_columns(
                    _rid=pl.int_range(0, over.height, dtype=pl.Int64)
                )

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
                        f"constraint {cname!r}: rhs Param has dim {missing} "
                        f"not in over={axis_cols}"
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
                    on
                    and sources is not None
                    and len(sources) >= 2
                )
                if use_prune_down:
                    # Start the accumulator from row_index with value=1.0.
                    # Each atomic contributes either as a left-joined value
                    # column (multiplied / divided in) or, for scalar
                    # atomics, as a literal scalar factor.
                    acc = row_index.lazy().with_columns(value=pl.lit(1.0, dtype=pl.Float64))
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
                            atomic_pruned = atomic_a.join(
                                keys_a, on=atomic_on, how="semi"
                            )
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
                                    value=pl.col("value")
                                    * pl.col("value__rhs_chain")
                                ).drop("value__rhs_chain")
                            else:
                                acc = joined.with_columns(
                                    value=pl.col("value")
                                    / pl.col("value__rhs_chain")
                                ).drop("value__rhs_chain")
                        else:
                            # Scalar atomic — fold the constant into the
                            # running value column directly.
                            scalar_val = float(atomic.frame["value"][0])
                            if direction >= 0:
                                acc = acc.with_columns(
                                    value=pl.col("value") * scalar_val
                                )
                            else:
                                acc = acc.with_columns(
                                    value=pl.col("value") / scalar_val
                                )
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
                        .astype(np.float64)
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
                        .astype(np.float64)
                    )
                else:
                    rhs_vec[:] = float(rhs.frame["value"][0])
            elif isinstance(rhs, (Var, Expr)):
                rhs_expr = rhs.to_expr() if isinstance(rhs, Var) else rhs
                neg = [
                    _Term(t.lazy.with_columns(coef=-pl.col("coef")), t.dims)
                    for t in rhs_expr.terms
                ]
                expr = Expr(expr.terms + neg)
            else:
                raise TypeError(
                    f"constraint {cname!r}: unsupported rhs type "
                    f"{type(rhs).__name__}"
                )

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
                    f"constraint {cname!r}: sense must be '<=', '>=' or "
                    f"'=='; got {sense!r}"
                )
            rows_lb_chunks.append(lb_vec)
            rows_ub_chunks.append(ub_vec)
            sense_chunks.append(
                np.full(row_count, ord(sc), dtype=np.uint8)
            )

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
            for term in expr.terms:
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
                    rl_a, tl_a = _align_enum_join_keys(
                        row_index_lf, term.lazy, on
                    )
                    keys_lazy = rl_a.select(on).unique()
                    tl_pruned = tl_a.join(keys_lazy, on=on, how="semi")
                    plan = (
                        rl_a.join(tl_pruned, on=on, how="inner")
                        .select("_rid", "col_id", "coef")
                    )
                    term_plans.append(("dim", plan, list(on)))
                else:
                    term_plans.append(
                        ("scalar", term.lazy.select("col_id", "coef"), [])
                    )

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
                        c if c.isalnum() or c in "._-" else "_"
                        for c in str(cname)
                    )
                    for _i, (_kind, _p, _on) in enumerate(term_plans):
                        _fname = (
                            f"{_plans_dir}/{_fam_idx:04d}_"
                            f"{_safe_cname}_term{_i}.txt"
                        )
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
                    rids_local = j["_rid"].to_numpy().astype(np.int64)
                    abs_rows = (base_row + rids_local).astype(np.int64)
                    cids = j["col_id"].to_numpy().astype(np.int64)
                    vals = j["coef"].to_numpy().astype(np.float64)
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
                    cids = j["col_id"].to_numpy().astype(np.int64)
                    vals = j["coef"].to_numpy().astype(np.float64)
                    if cids.size == 0:
                        continue
                    rs = np.repeat(
                        np.arange(
                            base_row, base_row + row_count, dtype=np.int64
                        ),
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
            tr = dedup["r"].to_numpy().astype(np.int64)
            tc = dedup["c"].to_numpy().astype(np.int64)
            tv = dedup["v"].to_numpy().astype(np.float64)
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
            f = t.lazy.collect()
            if f.height == 0:
                del f
                continue
            cids = f["col_id"].to_numpy().astype(np.int64)
            vals = f["coef"].to_numpy().astype(np.float64)
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
        for v in self._vars.values():
            ids = v.frame["col_id"].to_numpy()
            col_lb[ids] = float(v.lower)
            col_ub[ids] = float(v.upper)
            if v.integer:
                col_int[ids] = 1

        # ---- Pass 6: col names.  Used by write_mps + (future)
        # write_lp / diagnostic emitters.  Not gated on emit_names —
        # the canonical store always carries them; write_mps can
        # override with generic R/C names at emit time.
        col_names: list[str] = [""] * n_cols
        for v in self._vars.values():
            ids = v.frame["col_id"].to_numpy()
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
            row_names = ["cost"] + [
                f"R{i + 2:07d}" for i in range(n_constraint_rows)
            ]
            col_names = [f"C{j + 1:07d}" for j in range(n_cols)]
            # Note: cost row externally appears as "cost"; constraint
            # rows are R0000002, R0000003, ... so the generic numbering
            # gives 1-indexed sequential row ids that match how a user
            # would count rows in the file.

        # ---- Integer-col set from m.col_int (1 bit per column).
        integer_cols: set[int] = set(
            int(c) for c in np.nonzero(m.col_int)[0].tolist()
        )

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
                    f.write(
                        "    MARKER                 'MARKER'"
                        "                 'INTORG'\n"
                    )
                    in_integer = True
                elif (not is_int) and in_integer:
                    f.write(
                        "    MARKER                 'MARKER'"
                        "                 'INTEND'\n"
                    )
                    in_integer = False
                cname = col_names[j]
                if obj_nz:
                    f.write(f"    {cname}  cost  {_fmt(obj_v)}\n")
                # Matrix entries — row_idx is 0-based over constraint
                # rows; MPS row name is row_names[rid + 1].
                for k in range(start, end):
                    r = int(row_idx[k])
                    f.write(
                        f"    {cname}  {row_names[r + 1]}  {_fmt(val[k])}\n"
                    )
            if in_integer:
                f.write(
                    "    MARKER                 'MARKER'"
                    "                 'INTEND'\n"
                )
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
                    f.write(
                        f"    rhs  {row_names[rid + 1]}  {_fmt(rhs_arr[rid])}\n"
                    )

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
                            f.write(
                                f" LO bnd  {col_names[int(cid)]}  {_fmt(lo)}\n"
                            )
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
        col_lb_h = np.where(m.col_lb == -np.inf, -inf, m.col_lb).astype(np.float64)
        col_ub_h = np.where(m.col_ub == np.inf, inf, m.col_ub).astype(np.float64)
        row_lb_h = np.where(m.row_lb == -np.inf, -inf, m.row_lb).astype(np.float64)
        row_ub_h = np.where(m.row_ub == np.inf, inf, m.row_ub).astype(np.float64)

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
    ) -> Solution | None:
        _sp_emit, _sp_on = _make_solve_profile_emitter()
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
            f = t.lazy.collect()
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
        col_lb_h = np.where(col_lb == -np.inf, -inf, col_lb).astype(np.float64)
        col_ub_h = np.where(col_ub == np.inf, inf, col_ub).astype(np.float64)
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
                    rhs_vec = j.sort("_rid")["value"].fill_null(0.0).to_numpy().astype(np.float64)
                else:
                    rhs_vec[:] = float(rhs.frame["value"][0])
            elif isinstance(rhs, (Var, Expr)):
                rhs_expr = rhs.to_expr() if isinstance(rhs, Var) else rhs
                neg = [
                    _Term(t.lazy.with_columns(coef=-pl.col("coef")), t.dims) for t in rhs_expr.terms
                ]
                expr = Expr(expr.terms + neg)
            else:
                raise TypeError(f"constraint {name!r}: unsupported rhs type {type(rhs).__name__}")

            # Layer 2 row-factor on RHS (off ⇒ no-op).  base_row is
            # the 0-based absolute constraint row id; row_factor is
            # 0-indexed over constraints (HiGHS row index space starts
            # at 0 here — no cost row in the constraint space).
            if self._layer2_row_factor is not None and row_count:
                rhs_vec = rhs_vec * self._layer2_row_factor[
                    base_row : base_row + row_count
                ]

            if sense == "<=":
                row_lb = np.full(row_count, -inf, dtype=np.float64)
                row_ub = np.where(rhs_vec == np.inf, inf, rhs_vec).astype(np.float64)
            elif sense == ">=":
                row_lb = np.where(rhs_vec == -np.inf, -inf, rhs_vec).astype(np.float64)
                row_ub = np.full(row_count, inf, dtype=np.float64)
            elif sense == "==":
                rhs_h = np.where(rhs_vec == -np.inf, -inf, rhs_vec)
                rhs_h = np.where(rhs_h == np.inf, inf, rhs_h).astype(np.float64)
                row_lb = rhs_h
                row_ub = rhs_h
            else:
                raise ValueError(f"sense must be '<=', '>=' or '=='; got {sense!r}")

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
            for term in expr.terms:
                if term.dims:
                    missing = [d for d in term.dims if d not in axis_cols]
                    if missing:
                        raise ValueError(
                            f"constraint {name!r}: term has open dims {term.dims}, "
                            f"but constraint axes are {axis_cols}; aggregate "
                            f"{missing} via Sum() before adding."
                        )
                    on = [d for d in term.dims if d in axis_cols]
                    rl_a, tl_a = _align_enum_join_keys(row_index_lf, term.lazy, on)
                    # Semi-join + streaming pattern, mirroring write_mps
                    # and _build_lp_arrays: prune the term plan against
                    # the row-index key set so polars can prune Param-
                    # product join chains rather than materialise a wide
                    # intermediate.  Same bug class on the LHS as RHS.
                    keys_lazy = rl_a.select(on).unique()
                    tl_pruned = tl_a.join(keys_lazy, on=on, how="semi")
                    plan = (
                        rl_a.join(tl_pruned, on=on, how="inner")
                        .select("_rid", "col_id", "coef")
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
                        rids_local = j["_rid"].to_numpy().astype(np.int64)
                        cids = j["col_id"].to_numpy().astype(np.int64)
                        vals = j["coef"].to_numpy().astype(np.float64)
                        if _rf is not None:
                            vals = vals * _rf[base_row + rids_local]
                        if _cf is not None:
                            vals = vals * _cf[cids]
                        fam_rows.append(rids_local)
                        fam_cols.append(cids)
                        fam_vals.append(vals)
                    else:  # scalar — tile across the row_count rows
                        cids = j["col_id"].to_numpy().astype(np.int64)
                        vals = j["coef"].to_numpy().astype(np.float64)
                        if cids.size == 0:
                            continue
                        tiled_rows = np.repeat(
                            np.arange(row_count, dtype=np.int64), cids.size
                        )
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
                fv = dedup["v"].to_numpy().astype(np.float64)
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
                else tempfile.NamedTemporaryFile(
                    suffix=".mps", delete=False
                ).name
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

    def solve(self, *, options: dict | None = None) -> Solution:
        """Solve the LP.  First call builds the LP from scratch (same
        pipeline as :meth:`Problem.solve`); subsequent calls just run
        HiGHS again on the (possibly updated) live model.

        ``options`` is honoured on the FIRST solve only — subsequent
        solves use the same HiGHS instance.  To change options on a
        rebuilt LP, drop the WarmProblem and create a new one.
        """
        if self._h is None:
            self._initial_build(options=options)
        # subsequent: just rerun
        h = self._h
        h.run()
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
        col_lb_h = np.where(m.col_lb == -np.inf, -inf, m.col_lb).astype(np.float64)
        col_ub_h = np.where(m.col_ub == np.inf, inf, m.col_ub).astype(np.float64)
        row_lb_h = np.where(m.row_lb == -np.inf, -inf, m.row_lb).astype(np.float64)
        row_ub_h = np.where(m.row_ub == np.inf, inf, m.row_ub).astype(np.float64)
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
            expr, sense, rhs = proto.expr, proto.sense, proto.rhs

            if over is None:
                row_count = 1
                row_index = pl.DataFrame({"_rid": [0]})
                axis_cols: list[str] = []
            else:
                row_count = int(over.height)
                axis_cols = list(over.columns)
                row_index = over.with_columns(
                    _rid=pl.int_range(0, over.height, dtype=pl.Int64)
                )

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

            # Var/Expr-on-RHS fold so the second pass walks the same
            # expanded ``expr.terms`` that the canonical matrix walked.
            if isinstance(rhs, (Var, Expr)):
                rhs_expr = rhs.to_expr() if isinstance(rhs, Var) else rhs
                neg = [
                    _Term(t.lazy.with_columns(coef=-pl.col("coef")), t.dims)
                    for t in rhs_expr.terms
                ]
                expr = Expr(expr.terms + neg)

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
                    rl_a, tl_a = _align_enum_join_keys(
                        row_index_lf, term.lazy, on
                    )
                    keys_lazy = rl_a.select(on).unique()
                    tl_pruned = tl_a.join(keys_lazy, on=on, how="semi")
                    plan = rl_a.join(tl_pruned, on=on, how="inner").select(
                        "_rid", "col_id", "coef", *term.dims
                    )
                    j = _collect_streaming(plan)
                    if j.height == 0:
                        continue
                    rids = j["_rid"].to_numpy().astype(np.int64)
                    cids = j["col_id"].to_numpy().astype(np.int64)
                    coefs = j["coef"].to_numpy().astype(np.float64)
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
                            _kd_lhs, _pf_rhs = _align_enum_join_keys(
                                _kd_lhs, pobj.frame, _pdims_on
                            )
                            joined = _kd_lhs.join(
                                _pf_rhs, on=_pdims_on, how="left"
                            ).sort("__ridx")
                            old_vals = (
                                joined["value"]
                                .fill_null(0.0)
                                .to_numpy()
                                .astype(np.float64, copy=False)
                            )
                            if pdir == 1:
                                safe = np.where(old_vals == 0.0, 1.0, old_vals)
                                factor = np.where(
                                    old_vals == 0.0, 0.0, coefs / safe
                                )
                            else:
                                factor = coefs * old_vals
                            track_acc.setdefault(pname, []).append(
                                dict(
                                    rows=abs_rows.copy(),
                                    cols=cids.copy(),
                                    dim_keys=keys_df,
                                    factor=factor,
                                    direction=np.full(
                                        coefs.size, pdir, dtype=np.int8
                                    ),
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
                                    direction=np.full(
                                        coefs.size, pdir, dtype=np.int8
                                    ),
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
        if _sp_on:
            _sp_emit("initial_build_exit", n_cols=int(n_cols), n_rows=int(n_rows))
