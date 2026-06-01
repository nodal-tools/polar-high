"""Byte-identity parity for the side-vectors-OFF (ranges-PRE) range readout.

Background
----------
:func:`polar_high.autoscale._ranges._ranges_via_streaming` runs in TWO
passes:

* the production pass — the Layer-2 side vectors (``_layer2_row_factor`` /
  ``_layer2_col_factor``) are installed, so the readout reports
  POST-Layer-2 coefficient magnitudes; and
* the ranges-PRE pass (``detect_ranges`` in ``_orchestration``, BEFORE
  ``apply_layer2`` installs the side vectors) — ``_l2_rf is None`` and
  ``_l2_cf is None``, so the readout reports RAW ``|coef|`` magnitudes.

The prototype gated EVERY bounded-walk dispatch arm on ``_l2_rf is not
None``, so in ranges-pre the entire bounded infrastructure was inert: wide
families MATERIALISED the full ``Var × P1 × P2 …`` product (the dominant DES
autoscale spike) or were size-blind SKIPPED (silently dropping a family's
range).  The fix un-gates every arm — passing ``scale=(None, base_row,
None)`` in ranges-pre, which :class:`MinMaxAbsReducer` handles natively (the
row/col factor multiplies are skipped) — and retires the size cap.

These tests pin the ranges-pre RAW-``|coef|`` readout BYTE-IDENTICAL to an
independent whole-collect reference, for the three target shapes:

* a ``Var × Param × Param`` dense-axes LHS term;
* a ``Param × Param`` composite RHS chain (``from_rhs_chain``);
* a frame-constructed Param RHS (``from_rhs_param``);

plus a bare-Var LHS, a sparse RHS, and a mixed problem.  ``min/max(|coef|)``
is order-free, so the bounded per-batch fold is byte-identical to the
whole-collect regardless of how families partition into batches.  No
tolerance.
"""

from __future__ import annotations

import inspect
import itertools
import math
import os

import numpy as np
import polars as pl

from polar_high.autoscale import ScalingConfig
from polar_high.autoscale import _ranges as _ranges_mod
from polar_high.autoscale._ranges import _build_report, _ranges_via_streaming
from polar_high.engine import Param, Problem, Sum


def _config() -> ScalingConfig:
    return ScalingConfig(
        threshold_decades=9.0,
        user_bound_scale=None,
        report_yaml_path=None,
    )


def _clear_guard() -> None:
    os.environ.pop("POLAR_HIGH_DISABLE_BLOCK_COO", None)
    os.environ.pop("POLAR_HIGH_DISABLE_PRUNE_DOWN", None)
    os.environ.pop("POLAR_HIGH_BLOCK_COO_MIN_DENSE", None)
    os.environ.pop("POLAR_HIGH_BLOCK_COO_PROFILE", None)
    os.environ.pop("POLAR_HIGH_RANGES_PROFILE", None)


# ---------------------------------------------------------------------------
# Independent whole-collect reference RangeReport (RAW |coef|, no scaling).


def _reduce(arrs) -> tuple[float, float]:
    """``(min |a|, max |a|)`` over finite non-zero entries of the concatenated
    arrays, or ``(nan, nan)`` when empty — mirrors ``_abs_finite_nonzero_min_max``."""
    lo, hi = math.inf, 0.0
    for a in arrs:
        if a is None:
            continue
        a = np.asarray(a, dtype=np.float64)
        if a.size == 0:
            continue
        mask = np.isfinite(a) & (a != 0)
        if not mask.any():
            continue
        m = np.abs(a[mask])
        lo = min(lo, float(m.min()))
        hi = max(hi, float(m.max()))
    if hi == 0.0:
        return (math.nan, math.nan)
    return (lo, hi)


def _ref_report(prob: Problem, cfg: ScalingConfig):
    """Whole-collect reference: reduce each LP component to RAW ``|coef|``
    ranges WITHOUT any Layer-2 scaling and WITHOUT batching — the ground
    truth the ranges-pre bounded walk must match byte-for-byte."""
    # Bounds — finite non-zero |lower| / |upper| over all vars.
    bound_vals = []
    for v in prob._vars.values():
        for b in (v.lower, v.upper):
            bound_vals.append(b)
    bound = _reduce([np.asarray(bound_vals, dtype=np.float64)])

    # Cost — whole-collect (col_id, coef) over the objective terms.
    cost_arrs = []
    for t in prob._obj_terms:
        if t.lazy is None:
            continue
        df = t.lazy.select("coef").collect()
        cost_arrs.append(df["coef"].to_numpy().astype(np.float64))
    cost = _reduce(cost_arrs)

    # Matrix + RHS — per constraint family.
    matrix_arrs = []
    rhs_arrs = []
    for _cname, proto, over in prob._cstrs:
        rhs = proto.rhs
        if isinstance(rhs, (int, float)):
            rhs_arrs.append(np.full(1, float(rhs)))
        elif isinstance(rhs, Param):
            if over is not None and rhs.dims:
                on = list(rhs.dims)
                j = (
                    over.lazy()
                    .join(rhs.lazy, on=on, how="left")
                    .select("value")
                    .collect()
                )
                rhs_arrs.append(
                    j["value"].fill_null(0.0).to_numpy().astype(np.float64)
                )
            else:
                f = rhs.frame
                if "value" in f.columns and f.height > 0:
                    rhs_arrs.append(f["value"].to_numpy().astype(np.float64))

        # LHS terms — whole-collect coef, inner-joined to the over grid
        # (matches the surviving (row, col) cell set the readout reduces).
        axis_cols = list(over.columns) if over is not None else []
        for term in proto.expr.terms:
            term_lazy = term.lazy
            term_dims = term.dims
            if term_dims and over is not None:
                joon = [d for d in term_dims if d in axis_cols]
                df = (
                    over.lazy()
                    .join(term_lazy, on=joon, how="inner")
                    .select("coef")
                    .collect()
                )
            else:
                df = term_lazy.select("coef").collect()
            matrix_arrs.append(df["coef"].to_numpy().astype(np.float64))
    matrix = _reduce(matrix_arrs)
    rhs_r = _reduce(rhs_arrs)
    return _build_report(matrix, cost, bound, rhs_r, cfg)


def _report_tuple(rep) -> tuple:
    def _pair(x):
        return (repr(x[0]), repr(x[1]))

    return (
        _pair(rep.matrix),
        _pair(rep.cost),
        _pair(rep.bound),
        _pair(rep.rhs),
        repr(rep.cross_group_max_ratio),
        rep.trigger,
    )


# ---------------------------------------------------------------------------
# Problem builders.


def _vpp_lhs_problem() -> Problem:
    """A ``Var × Param × Param`` dense-axes LHS term (dense suffix (d, t))."""
    prob = Problem(dense_axes=("d", "t"))
    ps, ds, ts = [0, 1, 2], [10, 11, 12, 13], [100, 101, 102, 103, 104]
    cells = list(itertools.product(ps, ds, ts))
    over = pl.DataFrame(
        {"p": [c[0] for c in cells], "d": [c[1] for c in cells],
         "t": [c[2] for c in cells]}
    )
    v = prob.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)
    dt = list(itertools.product(ds, ts))
    Pa = Param(("d", "t"), pl.DataFrame(
        {"d": [c[0] for c in dt], "t": [c[1] for c in dt],
         "value": np.linspace(1e-3, 1e2, len(dt))}), name="Pa")
    Pb = Param(("d", "t"), pl.DataFrame(
        {"d": [c[0] for c in dt], "t": [c[1] for c in dt],
         "value": np.linspace(2.0, 5e3, len(dt))}), name="Pb")
    prob.add_cstr("vpp", over=over, sense="<=",
                  lhs_terms={"l": v * Pa * Pb}, rhs_terms={"r": 0.0})
    prob.set_objective(Sum(v), sense="min")
    return prob


def _rhs_chain_problem() -> Problem:
    """A ``Param × Param × Param`` composite RHS chain over a dense-complete
    (p, s, d, t) grid (dense suffix (d, t)) — the from_rhs_chain shape."""
    prob = Problem(dense_axes=("d", "t"))
    ps, ss, ds, ts = [0, 1, 2], ["s0", "s1"], [10, 11], [100, 101, 102, 103]
    rows = list(itertools.product(ps, ss, ds, ts))
    over = pl.DataFrame(
        {"p": [r[0] for r in rows], "s": [r[1] for r in rows],
         "d": [r[2] for r in rows], "t": [r[3] for r in rows]}
    )
    v = prob.add_var("v", ("p", "s", "d", "t"), over, lower=0.0, upper=1e6)
    pdt = list(itertools.product(ps, ds, ts))
    Pprofile = Param(("p", "d", "t"), pl.DataFrame(
        {"p": [c[0] for c in pdt], "d": [c[1] for c in pdt],
         "t": [c[2] for c in pdt], "value": np.linspace(1e-3, 5e2, len(pdt))}),
        name="Pprofile")
    psl = list(itertools.product(ps, ss))
    Pcount = Param(("p", "s"), pl.DataFrame(
        {"p": [c[0] for c in psl], "s": [c[1] for c in psl],
         "value": np.linspace(2.0, 4e3, len(psl))}), name="Pcount")
    dt = list(itertools.product(ds, ts))
    Pavail = Param(("d", "t"), pl.DataFrame(
        {"d": [c[0] for c in dt], "t": [c[1] for c in dt],
         "value": np.linspace(0.4, 0.95, len(dt))}), name="Pavail")
    prob.add_cstr("pful", over=over, sense="<=",
                  lhs_terms={"l": v}, rhs_terms={"r": Pprofile * Pcount * Pavail})
    prob.set_objective(Sum(v), sense="min")
    return prob


def _rhs_frame_param_problem() -> Problem:
    """A SINGLE frame-built Param RHS (``_sources is None``) over (p, d, t) —
    the from_rhs_param (DES ``maxToSink``) shape — plus a bare-Var LHS."""
    prob = Problem(dense_axes=("d", "t"))
    ps, ds, ts = [0, 1, 2], [10, 11], [100, 101, 102, 103]
    rows = list(itertools.product(ps, ds, ts))
    over = pl.DataFrame(
        {"p": [r[0] for r in rows], "d": [r[1] for r in rows],
         "t": [r[2] for r in rows]}
    )
    v = prob.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)
    rhs = Param(("p", "d", "t"), pl.DataFrame(
        {"p": [r[0] for r in rows], "d": [r[1] for r in rows],
         "t": [r[2] for r in rows],
         "value": np.linspace(1e-3, 5e2, len(rows))}), name="maxToSink")
    assert rhs._sources is None
    prob.add_cstr("mts", over=over, sense="<=",
                  lhs_terms={"l": v * 7.5}, rhs_terms={"r": rhs})
    prob.set_objective(Sum(v), sense="min")
    return prob


def _rhs_frame_param_sparse_problem() -> Problem:
    """A frame Param RHS that is SPARSE on (d, t) (drops cells) ⇒ the
    positional fast path declines and the prune-down backstop must reproduce
    the left-join ``fill_null(0.0)`` byte-identically."""
    prob = Problem(dense_axes=("d", "t"))
    ps, ds, ts = [0, 1, 2], [10, 11, 12], [100, 101, 102]
    rows = list(itertools.product(ps, ds, ts))
    over = pl.DataFrame(
        {"p": [r[0] for r in rows], "d": [r[1] for r in rows],
         "t": [r[2] for r in rows]}
    )
    v = prob.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)
    keep = [c for i, c in enumerate(rows) if i % 7 != 0]
    rhs = Param(("p", "d", "t"), pl.DataFrame(
        {"p": [r[0] for r in keep], "d": [r[1] for r in keep],
         "t": [r[2] for r in keep],
         "value": np.linspace(1e-2, 9e2, len(keep))}), name="maxToSink_sparse")
    prob.add_cstr("mts", over=over, sense="<=",
                  lhs_terms={"l": v}, rhs_terms={"r": rhs})
    prob.set_objective(Sum(v), sense="min")
    return prob


# ---------------------------------------------------------------------------
# Tests: ranges-PRE (no side vectors installed) == whole-collect reference.


def _assert_ranges_pre_matches_ref(prob: Problem) -> None:
    cfg = _config()
    # Side vectors NOT installed (default ``None``) ⇒ the ranges-PRE pass
    # (``_l2_rf is None``, ``_l2_cf is None``).
    assert getattr(prob, "_layer2_row_factor", None) is None
    assert getattr(prob, "_layer2_col_factor", None) is None
    rep = _ranges_via_streaming(prob, cfg)
    ref = _ref_report(prob, cfg)
    assert _report_tuple(rep) == _report_tuple(ref), (
        "ranges-pre bounded readout diverged from the whole-collect RAW "
        "|coef| reference:\n"
        f"  GOT = {_report_tuple(rep)}\n"
        f"  REF = {_report_tuple(ref)}"
    )


def test_ranges_pre_vpp_lhs_byte_identical() -> None:
    _clear_guard()
    try:
        _assert_ranges_pre_matches_ref(_vpp_lhs_problem())
    finally:
        _clear_guard()


def test_ranges_pre_rhs_chain_byte_identical() -> None:
    _clear_guard()
    try:
        _assert_ranges_pre_matches_ref(_rhs_chain_problem())
    finally:
        _clear_guard()


def test_ranges_pre_rhs_frame_param_byte_identical() -> None:
    _clear_guard()
    try:
        _assert_ranges_pre_matches_ref(_rhs_frame_param_problem())
    finally:
        _clear_guard()


def test_ranges_pre_rhs_frame_param_sparse_byte_identical() -> None:
    _clear_guard()
    try:
        _assert_ranges_pre_matches_ref(_rhs_frame_param_sparse_problem())
    finally:
        _clear_guard()


def test_no_family_row_cap_skip_path_remains() -> None:
    """The size-blind family-row cap (``_skip_unbounded_over_cap`` and the
    ``POLAR_HIGH_RANGES_MAX_FAMILY_ROWS`` env override) is RETIRED: no skip
    function, no cap env read, and no ``ranges-stream SKIP`` log line remains
    in ``_ranges``.  A remaining cap would re-introduce the coverage gap (a
    wide family's range silently dropped), so this pins its absence."""
    src = inspect.getsource(_ranges_mod)
    assert "_skip_unbounded_over_cap" not in src, (
        "the size-blind family-row cap skip function must be fully removed"
    )
    assert "POLAR_HIGH_RANGES_MAX_FAMILY_ROWS" not in src, (
        "the family-row cap env override must be fully removed"
    )
    assert "ranges-stream SKIP" not in src, (
        "no family-row-cap SKIP log line may remain"
    )
    assert "_max_family_rows" not in src


def test_large_row_family_range_is_folded_not_dropped() -> None:
    """A family whose row count is well over the OLD default cap
    (1_000_000) folds its coefficient range into the readout rather than
    being size-blind skipped.  Even with the old cap env var set to a TINY
    value (now a no-op), the family's range still reaches the report —
    byte-identical to the no-env run — proving no cap gate survives."""
    cfg = _config()
    _clear_guard()
    try:
        # Build a family with > 1000 rows (small but enough to exceed a tiny
        # cap), then run with the OLD cap env var set to 1 (a no-op now).
        prob = Problem(dense_axes=("d", "t"))
        ps = list(range(20))
        ds, ts = list(range(6)), list(range(12))  # 20*6*12 = 1440 rows
        cells = list(itertools.product(ps, ds, ts))
        over = pl.DataFrame(
            {"p": [c[0] for c in cells], "d": [c[1] for c in cells],
             "t": [c[2] for c in cells]}
        )
        v = prob.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)
        dt = list(itertools.product(ds, ts))
        Pa = Param(("d", "t"), pl.DataFrame(
            {"d": [c[0] for c in dt], "t": [c[1] for c in dt],
             "value": np.linspace(1e-3, 1e2, len(dt))}), name="Pa")
        prob.add_cstr("big", over=over, sense="<=",
                      lhs_terms={"l": v * Pa}, rhs_terms={"r": 0.0})
        prob.set_objective(Sum(v), sense="min")

        rep_default = _ranges_via_streaming(prob, cfg)
        # Set the OLD cap env var to 1 — if any cap gate survived it would
        # skip this 1440-row family and drop its matrix range to (nan, nan).
        os.environ["POLAR_HIGH_RANGES_MAX_FAMILY_ROWS"] = "1"
        rep_tiny_cap = _ranges_via_streaming(prob, cfg)
    finally:
        _clear_guard()

    # The matrix range MUST be present (non-NaN) — the family's coefficients
    # were folded in, not dropped.
    assert not math.isnan(rep_default.matrix[0])
    assert not math.isnan(rep_default.matrix[1])
    # And the tiny-cap run is byte-identical (the env var is a no-op now).
    assert _report_tuple(rep_tiny_cap) == _report_tuple(rep_default), (
        "the retired cap env var must be a no-op: the large family's range "
        "must be identical with and without it set"
    )


def test_ranges_pre_matches_with_block_coo_disabled() -> None:
    """Ranges-pre with block-COO ENABLED must equal ranges-pre with block-COO
    DISABLED (the walk's internal positional fast path declines ⇒ prune-down
    backstop) — proving the bounded builders agree value-for-value in the
    side-vectors-off pass too.  Same Problem instance, only the env lever
    differs."""
    cfg = _config()
    for builder in (
        _vpp_lhs_problem, _rhs_chain_problem,
        _rhs_frame_param_problem, _rhs_frame_param_sparse_problem,
    ):
        _clear_guard()
        try:
            prob = builder()
            rep_on = _ranges_via_streaming(prob, cfg)
            os.environ["POLAR_HIGH_DISABLE_BLOCK_COO"] = "1"
            rep_off = _ranges_via_streaming(prob, cfg)
        finally:
            _clear_guard()
        assert _report_tuple(rep_on) == _report_tuple(rep_off), (
            f"{builder.__name__}: ranges-pre block-COO on/off diverged:\n"
            f"  ON  = {_report_tuple(rep_on)}\n"
            f"  OFF = {_report_tuple(rep_off)}"
        )
