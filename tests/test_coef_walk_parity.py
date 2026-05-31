"""Parity + memory-bound harness for the Phase D-5 bounded
coefficient-walk primitive (:mod:`polar_high.autoscale._coef_walk`).

For each block-COO term shape (bare-Var, ``Var×P×P`` dense, Sum-combining,
map-Where, sparse/non-dense grid, Var-on-RHS) we assert that
:func:`bounded_coefficient_walk` driven by :class:`MinMaxAbsReducer`
returns ``(lo, hi)`` BYTE-IDENTICAL to a reference whole-collect min/max
(``_reduce_abs`` over the fully-materialised chain), for several
``batch_rows`` values including ``1`` and ``> n`` — proving batching
invariance.  :class:`Log2HistogramReducer` is checked against a
whole-collect histogram (exact for a single batch, within FP reassociation
across batches).  A tracemalloc check proves the per-batch peak is bounded
by ~batch on a wide shape, not the full product.
"""

from __future__ import annotations

import itertools
import math
import tracemalloc

import numpy as np
import polars as pl
import pytest

from polar_high.autoscale._coef_walk import (
    CoefWalkRecipe,
    Log2HistogramReducer,
    MinMaxAbsReducer,
    bounded_coefficient_walk,
)
from polar_high.engine import Param, Problem, Sum, Where

# ---------------------------------------------------------------------------
# Reference whole-collect reductions (mirror _ranges._reduce_abs exactly).


def _reduce_abs(vals: np.ndarray) -> tuple[float | None, float | None]:
    if vals.size == 0:
        return None, None
    mask = np.isfinite(vals) & (vals != 0)
    if not mask.any():
        return None, None
    a = np.abs(vals[mask])
    return float(a.min()), float(a.max())


def _scale_vals(
    rids: np.ndarray,
    cids: np.ndarray,
    coef: np.ndarray,
    scale: tuple[np.ndarray | None, int, np.ndarray | None],
) -> np.ndarray:
    l2_rf, base_row, l2_cf = scale
    vals = coef.astype(np.float64, copy=True)
    if l2_rf is not None and rids.size and rids.min() >= 0:
        vals = vals * np.abs(l2_rf[base_row + rids])
    if l2_cf is not None:
        vals = vals * np.abs(l2_cf[cids])
    return vals


def _ref_minmax_constraint(
    term, over: pl.DataFrame, scale
) -> tuple[float | None, float | None]:
    """Reference: collect the term's merged lazy plan whole, attach _rid by
    inner-joining over, apply scale, reduce.  This is the whole-collect
    streaming readout the bounded path must match byte-for-byte."""
    over_rid = over.with_columns(
        _rid=pl.int_range(0, over.height, dtype=pl.Int64)
    )
    term_lazy = term.frame.lazy()
    on = [d for d in term.dims if d in over.columns]
    df = (
        over_rid.lazy()
        .join(term_lazy, on=on, how="inner")
        .select("_rid", "col_id", "coef")
        .collect()
    )
    if df.height == 0:
        return None, None
    rids = df["_rid"].to_numpy().astype(np.int64)
    cids = df["col_id"].to_numpy().astype(np.int64)
    coef = df["coef"].to_numpy().astype(np.float64)
    return _reduce_abs(_scale_vals(rids, cids, coef, scale))


def _ref_minmax_rhs_chain(
    rhs, over: pl.DataFrame, scale
) -> tuple[float | None, float | None]:
    """Reference for a Var-LESS RHS Param chain: collect the merged
    ``over ⋈ rhs.lazy`` product whole (the materialising path the bounded
    walk replaces), attach ``_rid``, apply the row-factor scale (NO col
    factor on the RHS), reduce.  Byte-identity target for the param-only
    walk mode."""
    over_rid = over.with_columns(
        _rid=pl.int_range(0, over.height, dtype=pl.Int64)
    )
    on = list(rhs.dims)
    j = (
        over_rid.lazy()
        .join(rhs.lazy, on=on, how="left")
        .select("_rid", "value")
        .collect()
    )
    if j.height == 0:
        return None, None
    rids = j["_rid"].to_numpy().astype(np.int64)
    vals = j["value"].fill_null(0.0).to_numpy().astype(np.float64)
    cids = np.full(rids.size, -1, dtype=np.int64)
    return _reduce_abs(_scale_vals(rids, cids, vals, scale))


def _ref_minmax_column(term, scale) -> tuple[float | None, float | None]:
    """Reference for a column-spine (objective) term: collect (col_id, coef)
    whole, apply col scale, reduce."""
    df = term.frame.lazy().select("col_id", "coef").collect()
    if df.height == 0:
        return None, None
    cids = df["col_id"].to_numpy().astype(np.int64)
    coef = df["coef"].to_numpy().astype(np.float64)
    rids = np.full(cids.size, -1, dtype=np.int64)
    return _reduce_abs(_scale_vals(rids, cids, coef, scale))


def _ref_histogram_constraint(term, over, scale, classify):
    """Whole-collect log2 histogram (sum_log2, count, min, max) per bucket
    key, over the full chain — the reference for Log2HistogramReducer."""
    over_rid = over.with_columns(
        _rid=pl.int_range(0, over.height, dtype=pl.Int64)
    )
    on = [d for d in term.dims if d in over.columns]
    df = (
        over_rid.lazy()
        .join(term.frame.lazy(), on=on, how="inner")
        .select("_rid", "col_id", "coef")
        .collect()
    )
    rids = df["_rid"].to_numpy().astype(np.int64)
    cids = df["col_id"].to_numpy().astype(np.int64)
    coef = df["coef"].to_numpy().astype(np.float64)
    vals = np.abs(_scale_vals(rids, cids, coef, scale))
    acc: dict = {}
    for v, c in zip(vals.tolist(), cids.tolist()):
        if not math.isfinite(v) or v <= 0:
            continue
        bkey = classify(int(c))
        if bkey is None:
            continue
        lv = math.log2(v)
        ps, pn, pmin, pmax = acc.get(bkey, (0.0, 0, math.inf, 0.0))
        acc[bkey] = (ps + lv, pn + 1, min(pmin, v), max(pmax, v))
    return acc


# ---------------------------------------------------------------------------
# Side-vector helper.


def _side_vectors(n_rows: int, n_cols: int):
    rf = np.array(
        [10.0 ** ((i % 5) - 2) for i in range(n_rows)], dtype=np.float64
    )
    cf = np.array(
        [10.0 ** ((i % 7) - 3) for i in range(n_cols)], dtype=np.float64
    )
    return rf, cf


BATCH_SIZES = [1, 2, 3, 7, 1_000_000]


# ---------------------------------------------------------------------------
# Shape 1: bare Var (no Param joins) on the LHS.


def test_bare_var_lhs():
    prob = Problem(dense_axes=("d", "t"))
    ds, ts = [10, 11, 12], [100, 101]
    cells = list(itertools.product(ds, ts))
    over = pl.DataFrame({"d": [c[0] for c in cells], "t": [c[1] for c in cells]})
    v = prob.add_var("v", ("d", "t"), over, lower=0.0, upper=1e6)
    prob.add_cstr("c", over=over, sense="<=", lhs_terms={"l": v * 3.5},
                  rhs_terms={"r": 0.0})
    term = prob._cstrs[0][1].expr.terms[0]
    rf, cf = _side_vectors(over.height, prob._next_col)
    scale = (rf, 0, cf)
    recipe = CoefWalkRecipe.from_term(term)
    ref = _ref_minmax_constraint(term, over, scale)
    for bs in BATCH_SIZES:
        (got,) = bounded_coefficient_walk(
            over, recipe, scale, [MinMaxAbsReducer(scale)],
            batch_rows=bs, dense_axes=("d", "t"),
        )
        assert got == ref, f"batch_rows={bs}: {got!r} != {ref!r}"


# ---------------------------------------------------------------------------
# Shape 2: Var x P x P dense LHS.


def _build_vpp():
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
    return prob, over


def test_vpp_dense_lhs():
    prob, over = _build_vpp()
    term = prob._cstrs[0][1].expr.terms[0]
    rf, cf = _side_vectors(over.height, prob._next_col)
    scale = (rf, 0, cf)
    recipe = CoefWalkRecipe.from_term(term)
    ref = _ref_minmax_constraint(term, over, scale)
    for bs in BATCH_SIZES:
        (got,) = bounded_coefficient_walk(
            over, recipe, scale, [MinMaxAbsReducer(scale)],
            batch_rows=bs, dense_axes=("d", "t"),
        )
        assert got == ref, f"batch_rows={bs}: {got!r} != {ref!r}"


def test_vpp_histogram_single_batch_exact():
    """Histogram is EXACT vs whole-collect for a single (n-row) batch."""
    prob, over = _build_vpp()
    term = prob._cstrs[0][1].expr.terms[0]
    rf, cf = _side_vectors(over.height, prob._next_col)
    scale = (rf, 0, cf)
    recipe = CoefWalkRecipe.from_term(term)

    # Two buckets by col_id parity.
    def classify(cid):
        return "even" if cid % 2 == 0 else "odd"

    ref = _ref_histogram_constraint(term, over, scale, classify)
    (got,) = bounded_coefficient_walk(
        over, recipe, scale, [Log2HistogramReducer(scale, classify)],
        batch_rows=over.height, dense_axes=("d", "t"),
    )
    assert set(got) == set(ref)
    for k in ref:
        rs, rn, rmin, rmax = ref[k]
        gs, gn, gmin, gmax = got[k]
        assert gn == rn
        assert gmin == rmin and gmax == rmax
        assert gs == rs  # single batch ⇒ identical summation order ⇒ exact


def test_vpp_histogram_batched_fp_tol():
    """Histogram across many small batches matches whole-collect within
    FP reassociation tolerance (count/min/max exact)."""
    prob, over = _build_vpp()
    term = prob._cstrs[0][1].expr.terms[0]
    rf, cf = _side_vectors(over.height, prob._next_col)
    scale = (rf, 0, cf)
    recipe = CoefWalkRecipe.from_term(term)

    def classify(cid):
        return cid % 3

    ref = _ref_histogram_constraint(term, over, scale, classify)
    for bs in (1, 2, 5):
        (got,) = bounded_coefficient_walk(
            over, recipe, scale, [Log2HistogramReducer(scale, classify)],
            batch_rows=bs, dense_axes=("d", "t"),
        )
        assert set(got) == set(ref)
        for k in ref:
            rs, rn, rmin, rmax = ref[k]
            gs, gn, gmin, gmax = got[k]
            assert gn == rn
            assert gmin == rmin and gmax == rmax
            assert gs == pytest.approx(rs, rel=1e-12, abs=1e-9)


# ---------------------------------------------------------------------------
# Shape 3: Sum (relabel) LHS — reduce_dims ⊆ var.dims.


def _build_sum_relabel():
    prob = Problem(dense_axes=("d", "t"))
    ps, ss, ds, ts = [0, 1], ["s0", "s1"], [10, 11], [100, 101, 102]
    rows = list(itertools.product(ps, ss, ds, ts))
    var_index = pl.DataFrame(
        {"p": [r[0] for r in rows], "s": [r[1] for r in rows],
         "d": [r[2] for r in rows], "t": [r[3] for r in rows]}
    )
    v = prob.add_var("v", ("p", "s", "d", "t"), var_index, lower=0.0, upper=1e6)
    P_unit = Param(("p",), pl.DataFrame({"p": ps, "value": [2.0, 30.0]}),
                   name="P_unit")
    dt = list(itertools.product(ds, ts))
    P_step = Param(("d", "t"), pl.DataFrame(
        {"d": [r[0] for r in dt], "t": [r[1] for r in dt],
         "value": np.linspace(0.5, 1.5, len(dt))}), name="P_step")
    nb = Sum(v * P_unit * P_step, over=("p", "s"))
    nb_over = nb.terms[0].frame.select(list(nb.terms[0].dims)).unique().sort(
        ["d", "t"]
    )
    prob.add_cstr("nb", over=nb_over, sense="<=",
                  lhs_terms={"l": nb}, rhs_terms={"r": 0.0})
    return prob, nb_over


def test_sum_relabel_lhs():
    prob, over = _build_sum_relabel()
    term = prob._cstrs[0][1].expr.terms[0]
    rf, cf = _side_vectors(over.height, prob._next_col)
    scale = (rf, 0, cf)
    recipe = CoefWalkRecipe.from_term(term)
    ref = _ref_minmax_constraint(term, over, scale)
    for bs in BATCH_SIZES:
        (got,) = bounded_coefficient_walk(
            over, recipe, scale, [MinMaxAbsReducer(scale)],
            batch_rows=bs, dense_axes=("d", "t"),
        )
        assert got == ref, f"batch_rows={bs}: {got!r} != {ref!r}"


# ---------------------------------------------------------------------------
# Shape 4: map-Where LHS (Sum over a map-introduced relabel).


def _build_map_where():
    prob = Problem(dense_axes=("d", "t"))
    ps, ss, ds, ts = [0, 1], ["s0", "s1"], [10, 11], [100, 101, 102]
    rows = list(itertools.product(ps, ss, ds, ts))
    var_index = pl.DataFrame(
        {"p": [r[0] for r in rows], "s": [r[1] for r in rows],
         "d": [r[2] for r in rows], "t": [r[3] for r in rows]}
    )
    v = prob.add_var("v", ("p", "s", "d", "t"), var_index, lower=0.0, upper=1e6)
    P_unit = Param(("p",), pl.DataFrame({"p": ps, "value": [2.0, 30.0]}),
                   name="P_unit")
    dt = list(itertools.product(ds, ts))
    P_step = Param(("d", "t"), pl.DataFrame(
        {"d": [r[0] for r in dt], "t": [r[1] for r in dt],
         "value": np.linspace(0.5, 1.5, len(dt))}), name="P_step")
    map_rows = list(itertools.product(ps, ss))
    map_to_n = pl.DataFrame(
        {"p": [r[0] for r in map_rows], "s": [r[1] for r in map_rows],
         "n": [f"n{(r[0] + (0 if r[1] == 's0' else 1)) % 2}" for r in map_rows]}
    )
    nb = Sum(Where(v * P_unit, map_to_n) * P_step, over=("p", "s"))
    nb_over = nb.terms[0].frame.select(list(nb.terms[0].dims)).unique().sort(
        ["n", "d", "t"]
    )
    prob.add_cstr("nb", over=nb_over, sense="<=",
                  lhs_terms={"l": nb}, rhs_terms={"r": 0.0})
    return prob, nb_over


def test_map_where_lhs():
    prob, over = _build_map_where()
    term = prob._cstrs[0][1].expr.terms[0]
    rf, cf = _side_vectors(over.height, prob._next_col)
    scale = (rf, 0, cf)
    recipe = CoefWalkRecipe.from_term(term)
    ref = _ref_minmax_constraint(term, over, scale)
    for bs in BATCH_SIZES:
        (got,) = bounded_coefficient_walk(
            over, recipe, scale, [MinMaxAbsReducer(scale)],
            batch_rows=bs, dense_axes=("d", "t"),
        )
        assert got == ref, f"batch_rows={bs}: {got!r} != {ref!r}"


# ---------------------------------------------------------------------------
# Shape 5: Sum-combining LHS (a reduced dim is NOT a Var dim).


def _build_sum_combining():
    """Var(p,d,t) fanned out to ``h`` via a map (p)->h, multiplied by a
    Param on (d,t) (so the map is NOT baked eagerly and ``h`` survives as a
    map extra), then Sum over ``h``.  ``h`` is map-introduced, NOT a Var
    dim, so the reduce genuinely COMBINES several coef terms per
    (p,d,t,col_id) kept group — exercising the combining builder arm.

    Sized ≥ the block-COO min-dense gate via the env override applied by
    the test so the combining classifier fires (rather than declining on
    the perf gate)."""
    prob = Problem(dense_axes=("d", "t"))
    ps, ds, ts = [0, 1, 2], [10, 11], [100, 101]
    cells = list(itertools.product(ps, ds, ts))
    var_index = pl.DataFrame(
        {"p": [c[0] for c in cells], "d": [c[1] for c in cells],
         "t": [c[2] for c in cells]}
    )
    v = prob.add_var("v", ("p", "d", "t"), var_index, lower=0.0, upper=1e6)
    P_unit = Param(("p",), pl.DataFrame({"p": ps, "value": [2.0, 5.0, 11.0]}),
                   name="P_unit")
    # Map p -> h (fan-out: each p maps to 2 distinct h values).
    map_ph = pl.DataFrame(
        {"p": [p for p in ps for _ in range(2)],
         "h": [h for _ in ps for h in ("h0", "h1")]}
    )
    dt = list(itertools.product(ds, ts))
    P_step = Param(("d", "t"), pl.DataFrame(
        {"d": [c[0] for c in dt], "t": [c[1] for c in dt],
         "value": np.linspace(0.5, 1.5, len(dt))}), name="P_step")
    # Multiplying by a (d,t) Param does NOT touch ``h``, so the map stays
    # deferred (where_map_frames survives) ⇒ the combining builder reduces
    # the fan-out over ``h``.
    expr = Sum(Where(v * P_unit, map_ph) * P_step, over=("h",))
    over = expr.terms[0].frame.select(list(expr.terms[0].dims)).unique().sort(
        ["p", "d", "t"]
    )
    prob.add_cstr("comb", over=over, sense="<=",
                  lhs_terms={"l": expr}, rhs_terms={"r": 0.0})
    return prob, over, expr.terms[0]


def test_sum_combining_lhs(monkeypatch):
    # Force the block-COO min-dense perf gate down so the (small) combining
    # shape is classified + reaches the combining builder arm.
    monkeypatch.setenv("POLAR_HIGH_BLOCK_COO_MIN_DENSE", "1")
    prob, over, term = _build_sum_combining()
    rf, cf = _side_vectors(over.height, prob._next_col)
    scale = (rf, 0, cf)
    recipe = CoefWalkRecipe.from_term(term)
    assert recipe.sum_block_meta is not None
    # Confirm this is genuinely the combining arm (a reduced dim outside
    # the Var dims AND the map survives unbaked).
    assert not set(recipe.sum_block_meta.reduce_dims).issubset(
        set(recipe.sum_block_meta.var_source.dims)
    )
    assert recipe.sum_block_meta.where_map_frames is not None
    ref = _ref_minmax_constraint(term, over, scale)
    # Combining arm processes whole-spine, so it is exact for every
    # batch_rows (the loop coalesces it to one batch).
    for bs in BATCH_SIZES:
        (got,) = bounded_coefficient_walk(
            over, recipe, scale, [MinMaxAbsReducer(scale)],
            batch_rows=bs, dense_axes=("d", "t"),
        )
        assert got == ref, f"batch_rows={bs}: {got!r} != {ref!r}"


# ---------------------------------------------------------------------------
# Shape 6: sparse / non-dense grid LHS (Param missing some (d,t) cells).


def _build_sparse():
    prob = Problem(dense_axes=("d", "t"))
    ps, ds, ts = [0, 1, 2], [10, 11, 12], [100, 101, 102]
    cells = list(itertools.product(ps, ds, ts))
    over = pl.DataFrame(
        {"p": [c[0] for c in cells], "d": [c[1] for c in cells],
         "t": [c[2] for c in cells]}
    )
    v = prob.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)
    # Pa is SPARSE on (d,t): drop a couple of cells so the dense
    # completeness guard fails ⇒ the joined / prune-down backstop fires.
    dt = list(itertools.product(ds, ts))
    dt_sparse = [c for i, c in enumerate(dt) if i not in (1, 4)]
    Pa = Param(("d", "t"), pl.DataFrame(
        {"d": [c[0] for c in dt_sparse], "t": [c[1] for c in dt_sparse],
         "value": np.linspace(1e-2, 50.0, len(dt_sparse))}), name="Pa")
    prob.add_cstr("sp", over=over, sense="<=",
                  lhs_terms={"l": v * Pa}, rhs_terms={"r": 0.0})
    return prob, over


def test_sparse_lhs():
    prob, over = _build_sparse()
    term = prob._cstrs[0][1].expr.terms[0]
    rf, cf = _side_vectors(over.height, prob._next_col)
    scale = (rf, 0, cf)
    recipe = CoefWalkRecipe.from_term(term)
    # The reference must use the SAME row set the chain produces (sparse
    # Pa drops cells), which _ref_minmax_constraint replicates via the
    # inner-join collect.
    ref = _ref_minmax_constraint(term, over, scale)
    for bs in BATCH_SIZES:
        (got,) = bounded_coefficient_walk(
            over, recipe, scale, [MinMaxAbsReducer(scale)],
            batch_rows=bs, dense_axes=("d", "t"),
        )
        assert got == ref, f"batch_rows={bs}: {got!r} != {ref!r}"


# ---------------------------------------------------------------------------
# Shape 7: Var on the RHS — scale-neutral, counted via the LHS.  The walk's
# column-mode reduction over an objective-style Var x Param chain is the
# closest analogue (no _rid attach); pin it byte-identical too.


def test_objective_column_mode():
    prob = Problem(dense_axes=("d", "t"))
    ps, ds, ts = [0, 1, 2], [10, 11, 12], [100, 101]
    cells = list(itertools.product(ps, ds, ts))
    idx = pl.DataFrame(
        {"p": [c[0] for c in cells], "d": [c[1] for c in cells],
         "t": [c[2] for c in cells]}
    )
    v = prob.add_var("v", ("p", "d", "t"), idx, lower=0.0, upper=1e6)
    dt = list(itertools.product(ds, ts))
    Pa = Param(("d", "t"), pl.DataFrame(
        {"d": [c[0] for c in dt], "t": [c[1] for c in dt],
         "value": np.linspace(1e-3, 1e2, len(dt))}), name="Pa")
    # Objective-style term: Var x Pa, NOT summed — column spine = Var.frame.
    term = (v * Pa).terms[0]
    # Column spine carries col_id.
    spine = v.frame
    n_cols = prob._next_col
    _, cf = _side_vectors(0, n_cols)
    scale = (None, 0, cf)  # no row factor for the objective
    recipe = CoefWalkRecipe.from_term(term)
    ref = _ref_minmax_column(term, scale)
    for bs in BATCH_SIZES:
        (got,) = bounded_coefficient_walk(
            spine, recipe, scale, [MinMaxAbsReducer(scale)],
            batch_rows=bs, dense_axes=("d", "t"),
        )
        assert got == ref, f"batch_rows={bs}: {got!r} != {ref!r}"


# ---------------------------------------------------------------------------
# Shape 8: Var-LESS RHS Param chain (param_only mode).  The bounded walk's
# (_rid, coef) stream — col_id absent — must match the whole-collect RHS
# range (over ⋈ rhs.lazy) byte-for-byte, across batch sizes, for dense AND
# non-dense / sparse chains.


def _build_rhs_chain_dense():
    """A DES ``profile_flow_upper_limit``-shaped RHS: a dense-complete
    3-Param composite chain over ``(p, s, d, t)`` with dense suffix
    ``(d, t)`` — exercises the param-only POSITIONAL fast path (lead+dense,
    lead-only, dense-only atomics)."""
    ps, ss, ds, ts = [0, 1, 2], ["s0", "s1"], [10, 11], [100, 101, 102, 103]
    rows = list(itertools.product(ps, ss, ds, ts))
    over = pl.DataFrame(
        {"p": [r[0] for r in rows], "s": [r[1] for r in rows],
         "d": [r[2] for r in rows], "t": [r[3] for r in rows]}
    )
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
    rhs = Pprofile * Pcount * Pavail
    return over, rhs


def _batch_sizes(n: int) -> list[int]:
    return sorted(set([1, max(1, n // 3), 1_000_000]))


def test_rhs_chain_dense_param_only():
    over, rhs = _build_rhs_chain_dense()
    n = over.height
    rf, cf = _side_vectors(n, 8)
    # Row factor present, NO col factor (RHS has none).
    scale = (rf, 0, None)
    recipe = CoefWalkRecipe.from_rhs_chain(rhs)
    assert recipe.param_only and recipe.var_source is None
    ref = _ref_minmax_rhs_chain(rhs, over, scale)
    for bs in _batch_sizes(n):
        (got,) = bounded_coefficient_walk(
            over, recipe, scale, [MinMaxAbsReducer(scale)],
            batch_rows=bs, dense_axes=("d", "t"),
        )
        assert got == ref, f"batch_rows={bs}: {got!r} != {ref!r}"


def test_rhs_chain_dense_param_only_base_row_offset():
    """A non-zero ``base_row`` offset must index the row factor correctly
    (mirrors a family that is not the first in the LP row order)."""
    over, rhs = _build_rhs_chain_dense()
    n = over.height
    base = 7
    rf, _cf = _side_vectors(n + base, 8)
    scale = (rf, base, None)
    recipe = CoefWalkRecipe.from_rhs_chain(rhs)
    ref = _ref_minmax_rhs_chain(rhs, over, scale)
    for bs in _batch_sizes(n):
        (got,) = bounded_coefficient_walk(
            over, recipe, scale, [MinMaxAbsReducer(scale)],
            batch_rows=bs, dense_axes=("d", "t"),
        )
        assert got == ref, f"batch_rows={bs}: {got!r} != {ref!r}"


def _build_rhs_chain_sparse():
    """A SPARSE RHS chain: the dense ``(d, t)`` atomic drops cells so the
    over grid is NOT dense-complete ⇒ the positional fast path declines and
    the param-only PRUNE-DOWN backstop fires (no dense suffix needed)."""
    ps, ds, ts = [0, 1, 2], [10, 11, 12], [100, 101, 102]
    rows = list(itertools.product(ps, ds, ts))
    over = pl.DataFrame(
        {"p": [r[0] for r in rows], "d": [r[1] for r in rows],
         "t": [r[2] for r in rows]}
    )
    # Pa SPARSE on (d,t): drop a couple of cells ⇒ left-join nulls ⇒ the
    # positional completeness / null guard declines ⇒ prune-down backstop.
    dt = list(itertools.product(ds, ts))
    dt_sparse = [c for i, c in enumerate(dt) if i not in (1, 4)]
    Pa = Param(("d", "t"), pl.DataFrame(
        {"d": [c[0] for c in dt_sparse], "t": [c[1] for c in dt_sparse],
         "value": np.linspace(1e-2, 50.0, len(dt_sparse))}), name="Pa")
    Pb = Param(("p",), pl.DataFrame({"p": ps, "value": [3.0, 40.0, 700.0]}),
               name="Pb")
    rhs = Pa * Pb
    return over, rhs


def test_rhs_chain_sparse_param_only():
    over, rhs = _build_rhs_chain_sparse()
    n = over.height
    rf, _cf = _side_vectors(n, 8)
    scale = (rf, 0, None)
    recipe = CoefWalkRecipe.from_rhs_chain(rhs)
    ref = _ref_minmax_rhs_chain(rhs, over, scale)
    for bs in _batch_sizes(n):
        (got,) = bounded_coefficient_walk(
            over, recipe, scale, [MinMaxAbsReducer(scale)],
            batch_rows=bs, dense_axes=("d", "t"),
        )
        assert got == ref, f"batch_rows={bs}: {got!r} != {ref!r}"


def test_rhs_chain_param_only_no_scale():
    """param_only with scale all-None reduces ``|coef|`` directly and still
    matches the whole-collect, dense AND sparse."""
    for builder in (_build_rhs_chain_dense, _build_rhs_chain_sparse):
        over, rhs = builder()
        n = over.height
        scale = (None, 0, None)
        recipe = CoefWalkRecipe.from_rhs_chain(rhs)
        ref = _ref_minmax_rhs_chain(rhs, over, scale)
        for bs in _batch_sizes(n):
            (got,) = bounded_coefficient_walk(
                over, recipe, scale, [MinMaxAbsReducer(scale)],
                batch_rows=bs, dense_axes=("d", "t"),
            )
            assert got == ref, f"{builder.__name__} batch_rows={bs}"


# ---------------------------------------------------------------------------
# No-scale path: scale all-None must still match a whole-collect |coef|.


def test_no_scale_minmax():
    prob, over = _build_vpp()
    term = prob._cstrs[0][1].expr.terms[0]
    scale = (None, 0, None)
    recipe = CoefWalkRecipe.from_term(term)
    ref = _ref_minmax_constraint(term, over, scale)
    for bs in (1, 4, 1_000_000):
        (got,) = bounded_coefficient_walk(
            over, recipe, scale, [MinMaxAbsReducer(scale)],
            batch_rows=bs, dense_axes=("d", "t"),
        )
        assert got == ref


# ---------------------------------------------------------------------------
# Memory bound: per-batch peak grows with batch_rows, NOT with the full
# product, on a wide dense shape.


def _build_wide():
    prob = Problem(dense_axes=("d", "t"))
    ps = list(range(40))
    ds = list(range(8))
    ts = list(range(60))
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
    return prob, over


def _peak_for_batch(over, recipe, scale, batch_rows) -> int:
    tracemalloc.start()
    tracemalloc.reset_peak()
    bounded_coefficient_walk(
        over, recipe, scale, [MinMaxAbsReducer(scale)],
        batch_rows=batch_rows, dense_axes=("d", "t"),
    )
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def test_memory_bounded_by_batch():
    prob, over = _build_wide()
    term = prob._cstrs[0][1].expr.terms[0]
    n = over.height
    rf, cf = _side_vectors(n, prob._next_col)
    scale = (rf, 0, cf)
    recipe = CoefWalkRecipe.from_term(term)

    # Correctness first: small batch == whole batch.
    ref = _ref_minmax_constraint(term, over, scale)
    (got_small,) = bounded_coefficient_walk(
        over, recipe, scale, [MinMaxAbsReducer(scale)],
        batch_rows=256, dense_axes=("d", "t"),
    )
    assert got_small == ref

    peak_small = _peak_for_batch(over, recipe, scale, 256)
    peak_whole = _peak_for_batch(over, recipe, scale, n)

    # The whole-batch peak must be materially larger than the small-batch
    # peak — proving the small batch never holds the full product.  Use a
    # generous margin to stay robust across polars allocator noise.
    assert peak_small * 3 < peak_whole, (
        f"small-batch peak {peak_small} not bounded below whole-batch "
        f"peak {peak_whole} (n={n})"
    )
