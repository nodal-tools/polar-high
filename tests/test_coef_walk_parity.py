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


def _ref_minmax_constraint(term, over: pl.DataFrame, scale) -> tuple[float | None, float | None]:
    """Reference: collect the term's merged lazy plan whole, attach _rid by
    inner-joining over, apply scale, reduce.  This is the whole-collect
    streaming readout the bounded path must match byte-for-byte."""
    over_rid = over.with_columns(_rid=pl.int_range(0, over.height, dtype=pl.Int64))
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


def _ref_minmax_rhs_chain(rhs, over: pl.DataFrame, scale) -> tuple[float | None, float | None]:
    """Reference for a Var-LESS RHS Param chain: collect the merged
    ``over ⋈ rhs.lazy`` product whole (the materialising path the bounded
    walk replaces), attach ``_rid``, apply the row-factor scale (NO col
    factor on the RHS), reduce.  Byte-identity target for the param-only
    walk mode."""
    over_rid = over.with_columns(_rid=pl.int_range(0, over.height, dtype=pl.Int64))
    on = list(rhs.dims)
    j = over_rid.lazy().join(rhs.lazy, on=on, how="left").select("_rid", "value").collect()
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


def _ref_histogram_column(term, scale, classify):
    """Whole-collect log2 histogram over a column-spine (objective) term —
    the reference for the column-mode :class:`Log2HistogramReducer`."""
    df = term.frame.lazy().select("col_id", "coef").collect()
    cids = df["col_id"].to_numpy().astype(np.int64)
    coef = df["coef"].to_numpy().astype(np.float64)
    rids = np.full(cids.size, -1, dtype=np.int64)
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


def _ref_histogram_constraint(term, over, scale, classify):
    """Whole-collect log2 histogram (sum_log2, count, min, max) per bucket
    key, over the full chain — the reference for Log2HistogramReducer."""
    over_rid = over.with_columns(_rid=pl.int_range(0, over.height, dtype=pl.Int64))
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
    rf = np.array([10.0 ** ((i % 5) - 2) for i in range(n_rows)], dtype=np.float64)
    cf = np.array([10.0 ** ((i % 7) - 3) for i in range(n_cols)], dtype=np.float64)
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
    prob.add_cstr("c", over=over, sense="<=", lhs_terms={"l": v * 3.5}, rhs_terms={"r": 0.0})
    term = prob._cstrs[0][1].expr.terms[0]
    rf, cf = _side_vectors(over.height, prob._next_col)
    scale = (rf, 0, cf)
    recipe = CoefWalkRecipe.from_term(term)
    ref = _ref_minmax_constraint(term, over, scale)
    for bs in BATCH_SIZES:
        (got,) = bounded_coefficient_walk(
            over,
            recipe,
            scale,
            [MinMaxAbsReducer(scale)],
            batch_rows=bs,
            dense_axes=("d", "t"),
        )
        assert got == ref, f"batch_rows={bs}: {got!r} != {ref!r}"


# ---------------------------------------------------------------------------
# Shape 2: Var x P x P dense LHS.


def _build_vpp():
    prob = Problem(dense_axes=("d", "t"))
    ps, ds, ts = [0, 1, 2], [10, 11, 12, 13], [100, 101, 102, 103, 104]
    cells = list(itertools.product(ps, ds, ts))
    over = pl.DataFrame(
        {"p": [c[0] for c in cells], "d": [c[1] for c in cells], "t": [c[2] for c in cells]}
    )
    v = prob.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)
    dt = list(itertools.product(ds, ts))
    Pa = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [c[0] for c in dt],
                "t": [c[1] for c in dt],
                "value": np.linspace(1e-3, 1e2, len(dt)),
            }
        ),
        name="Pa",
    )
    Pb = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [c[0] for c in dt],
                "t": [c[1] for c in dt],
                "value": np.linspace(2.0, 5e3, len(dt)),
            }
        ),
        name="Pb",
    )
    prob.add_cstr("vpp", over=over, sense="<=", lhs_terms={"l": v * Pa * Pb}, rhs_terms={"r": 0.0})
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
            over,
            recipe,
            scale,
            [MinMaxAbsReducer(scale)],
            batch_rows=bs,
            dense_axes=("d", "t"),
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
        over,
        recipe,
        scale,
        [Log2HistogramReducer(scale, classify)],
        batch_rows=over.height,
        dense_axes=("d", "t"),
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
            over,
            recipe,
            scale,
            [Log2HistogramReducer(scale, classify)],
            batch_rows=bs,
            dense_axes=("d", "t"),
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
        {
            "p": [r[0] for r in rows],
            "s": [r[1] for r in rows],
            "d": [r[2] for r in rows],
            "t": [r[3] for r in rows],
        }
    )
    v = prob.add_var("v", ("p", "s", "d", "t"), var_index, lower=0.0, upper=1e6)
    P_unit = Param(("p",), pl.DataFrame({"p": ps, "value": [2.0, 30.0]}), name="P_unit")
    dt = list(itertools.product(ds, ts))
    P_step = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [r[0] for r in dt],
                "t": [r[1] for r in dt],
                "value": np.linspace(0.5, 1.5, len(dt)),
            }
        ),
        name="P_step",
    )
    nb = Sum(v * P_unit * P_step, over=("p", "s"))
    nb_over = nb.terms[0].frame.select(list(nb.terms[0].dims)).unique().sort(["d", "t"])
    prob.add_cstr("nb", over=nb_over, sense="<=", lhs_terms={"l": nb}, rhs_terms={"r": 0.0})
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
            over,
            recipe,
            scale,
            [MinMaxAbsReducer(scale)],
            batch_rows=bs,
            dense_axes=("d", "t"),
        )
        assert got == ref, f"batch_rows={bs}: {got!r} != {ref!r}"


# ---------------------------------------------------------------------------
# Shape 4: map-Where LHS (Sum over a map-introduced relabel).


def _build_map_where():
    prob = Problem(dense_axes=("d", "t"))
    ps, ss, ds, ts = [0, 1], ["s0", "s1"], [10, 11], [100, 101, 102]
    rows = list(itertools.product(ps, ss, ds, ts))
    var_index = pl.DataFrame(
        {
            "p": [r[0] for r in rows],
            "s": [r[1] for r in rows],
            "d": [r[2] for r in rows],
            "t": [r[3] for r in rows],
        }
    )
    v = prob.add_var("v", ("p", "s", "d", "t"), var_index, lower=0.0, upper=1e6)
    P_unit = Param(("p",), pl.DataFrame({"p": ps, "value": [2.0, 30.0]}), name="P_unit")
    dt = list(itertools.product(ds, ts))
    P_step = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [r[0] for r in dt],
                "t": [r[1] for r in dt],
                "value": np.linspace(0.5, 1.5, len(dt)),
            }
        ),
        name="P_step",
    )
    map_rows = list(itertools.product(ps, ss))
    map_to_n = pl.DataFrame(
        {
            "p": [r[0] for r in map_rows],
            "s": [r[1] for r in map_rows],
            "n": [f"n{(r[0] + (0 if r[1] == 's0' else 1)) % 2}" for r in map_rows],
        }
    )
    nb = Sum(Where(v * P_unit, map_to_n) * P_step, over=("p", "s"))
    nb_over = nb.terms[0].frame.select(list(nb.terms[0].dims)).unique().sort(["n", "d", "t"])
    prob.add_cstr("nb", over=nb_over, sense="<=", lhs_terms={"l": nb}, rhs_terms={"r": 0.0})
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
            over,
            recipe,
            scale,
            [MinMaxAbsReducer(scale)],
            batch_rows=bs,
            dense_axes=("d", "t"),
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
        {"p": [c[0] for c in cells], "d": [c[1] for c in cells], "t": [c[2] for c in cells]}
    )
    v = prob.add_var("v", ("p", "d", "t"), var_index, lower=0.0, upper=1e6)
    P_unit = Param(("p",), pl.DataFrame({"p": ps, "value": [2.0, 5.0, 11.0]}), name="P_unit")
    # Map p -> h (fan-out: each p maps to 2 distinct h values).
    map_ph = pl.DataFrame(
        {"p": [p for p in ps for _ in range(2)], "h": [h for _ in ps for h in ("h0", "h1")]}
    )
    dt = list(itertools.product(ds, ts))
    P_step = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [c[0] for c in dt],
                "t": [c[1] for c in dt],
                "value": np.linspace(0.5, 1.5, len(dt)),
            }
        ),
        name="P_step",
    )
    # Multiplying by a (d,t) Param does NOT touch ``h``, so the map stays
    # deferred (where_map_frames survives) ⇒ the combining builder reduces
    # the fan-out over ``h``.
    expr = Sum(Where(v * P_unit, map_ph) * P_step, over=("h",))
    over = expr.terms[0].frame.select(list(expr.terms[0].dims)).unique().sort(["p", "d", "t"])
    prob.add_cstr("comb", over=over, sense="<=", lhs_terms={"l": expr}, rhs_terms={"r": 0.0})
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
            over,
            recipe,
            scale,
            [MinMaxAbsReducer(scale)],
            batch_rows=bs,
            dense_axes=("d", "t"),
        )
        assert got == ref, f"batch_rows={bs}: {got!r} != {ref!r}"


# ---------------------------------------------------------------------------
# Shape 6: sparse / non-dense grid LHS (Param missing some (d,t) cells).


def _build_sparse():
    prob = Problem(dense_axes=("d", "t"))
    ps, ds, ts = [0, 1, 2], [10, 11, 12], [100, 101, 102]
    cells = list(itertools.product(ps, ds, ts))
    over = pl.DataFrame(
        {"p": [c[0] for c in cells], "d": [c[1] for c in cells], "t": [c[2] for c in cells]}
    )
    v = prob.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)
    # Pa is SPARSE on (d,t): drop a couple of cells so the dense
    # completeness guard fails ⇒ the joined / prune-down backstop fires.
    dt = list(itertools.product(ds, ts))
    dt_sparse = [c for i, c in enumerate(dt) if i not in (1, 4)]
    Pa = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [c[0] for c in dt_sparse],
                "t": [c[1] for c in dt_sparse],
                "value": np.linspace(1e-2, 50.0, len(dt_sparse)),
            }
        ),
        name="Pa",
    )
    prob.add_cstr("sp", over=over, sense="<=", lhs_terms={"l": v * Pa}, rhs_terms={"r": 0.0})
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
            over,
            recipe,
            scale,
            [MinMaxAbsReducer(scale)],
            batch_rows=bs,
            dense_axes=("d", "t"),
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
        {"p": [c[0] for c in cells], "d": [c[1] for c in cells], "t": [c[2] for c in cells]}
    )
    v = prob.add_var("v", ("p", "d", "t"), idx, lower=0.0, upper=1e6)
    dt = list(itertools.product(ds, ts))
    Pa = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [c[0] for c in dt],
                "t": [c[1] for c in dt],
                "value": np.linspace(1e-3, 1e2, len(dt)),
            }
        ),
        name="Pa",
    )
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
            spine,
            recipe,
            scale,
            [MinMaxAbsReducer(scale)],
            batch_rows=bs,
            dense_axes=("d", "t"),
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
        {
            "p": [r[0] for r in rows],
            "s": [r[1] for r in rows],
            "d": [r[2] for r in rows],
            "t": [r[3] for r in rows],
        }
    )
    pdt = list(itertools.product(ps, ds, ts))
    Pprofile = Param(
        ("p", "d", "t"),
        pl.DataFrame(
            {
                "p": [c[0] for c in pdt],
                "d": [c[1] for c in pdt],
                "t": [c[2] for c in pdt],
                "value": np.linspace(1e-3, 5e2, len(pdt)),
            }
        ),
        name="Pprofile",
    )
    psl = list(itertools.product(ps, ss))
    Pcount = Param(
        ("p", "s"),
        pl.DataFrame(
            {
                "p": [c[0] for c in psl],
                "s": [c[1] for c in psl],
                "value": np.linspace(2.0, 4e3, len(psl)),
            }
        ),
        name="Pcount",
    )
    dt = list(itertools.product(ds, ts))
    Pavail = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [c[0] for c in dt],
                "t": [c[1] for c in dt],
                "value": np.linspace(0.4, 0.95, len(dt)),
            }
        ),
        name="Pavail",
    )
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
            over,
            recipe,
            scale,
            [MinMaxAbsReducer(scale)],
            batch_rows=bs,
            dense_axes=("d", "t"),
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
            over,
            recipe,
            scale,
            [MinMaxAbsReducer(scale)],
            batch_rows=bs,
            dense_axes=("d", "t"),
        )
        assert got == ref, f"batch_rows={bs}: {got!r} != {ref!r}"


def _build_rhs_chain_sparse():
    """A SPARSE RHS chain: the dense ``(d, t)`` atomic drops cells so the
    over grid is NOT dense-complete ⇒ the positional fast path declines and
    the param-only PRUNE-DOWN backstop fires (no dense suffix needed)."""
    ps, ds, ts = [0, 1, 2], [10, 11, 12], [100, 101, 102]
    rows = list(itertools.product(ps, ds, ts))
    over = pl.DataFrame(
        {"p": [r[0] for r in rows], "d": [r[1] for r in rows], "t": [r[2] for r in rows]}
    )
    # Pa SPARSE on (d,t): drop a couple of cells ⇒ left-join nulls ⇒ the
    # positional completeness / null guard declines ⇒ prune-down backstop.
    dt = list(itertools.product(ds, ts))
    dt_sparse = [c for i, c in enumerate(dt) if i not in (1, 4)]
    Pa = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [c[0] for c in dt_sparse],
                "t": [c[1] for c in dt_sparse],
                "value": np.linspace(1e-2, 50.0, len(dt_sparse)),
            }
        ),
        name="Pa",
    )
    Pb = Param(("p",), pl.DataFrame({"p": ps, "value": [3.0, 40.0, 700.0]}), name="Pb")
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
            over,
            recipe,
            scale,
            [MinMaxAbsReducer(scale)],
            batch_rows=bs,
            dense_axes=("d", "t"),
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
                over,
                recipe,
                scale,
                [MinMaxAbsReducer(scale)],
                batch_rows=bs,
                dense_axes=("d", "t"),
            )
            assert got == ref, f"{builder.__name__} batch_rows={bs}"


# ---------------------------------------------------------------------------
# Shape 9: SINGLE frame-built RHS Param (``_sources is None``, non-empty
# dims).  ``CoefWalkRecipe.from_rhs_param`` wraps it as a 1-element
# param_only chain; the walk's (_rid, coef) stream must match the
# whole-collect RHS range (``over ⋈ rhs.lazy`` left-join, fill_null(0.0))
# byte-for-byte, across batch sizes — the DES ``maxToSink`` shape.


def _build_rhs_frame_param():
    """A single frame-constructed ``Param`` RHS over ``(p, d, t)`` with the
    declared dense suffix ``(d, t)`` — dense-complete, so the param-only
    POSITIONAL fast path can fire.  No ``_sources`` (a directly-built frame
    Param), one ``value`` column, one row per over-row — NO deep product."""
    ps, ds, ts = [0, 1, 2], [10, 11], [100, 101, 102, 103]
    rows = list(itertools.product(ps, ds, ts))
    over = pl.DataFrame(
        {"p": [r[0] for r in rows], "d": [r[1] for r in rows], "t": [r[2] for r in rows]}
    )
    rhs = Param(
        ("p", "d", "t"),
        pl.DataFrame(
            {
                "p": [r[0] for r in rows],
                "d": [r[1] for r in rows],
                "t": [r[2] for r in rows],
                "value": np.linspace(1e-3, 5e2, len(rows)),
            }
        ),
        name="maxToSink",
    )
    return over, rhs


def _build_rhs_frame_param_sparse():
    """A single frame-constructed ``Param`` RHS that is SPARSE on ``(d, t)``
    (drops cells), so the over grid is NOT dense-complete ⇒ the param-only
    positional fast path declines and the PRUNE-DOWN backstop fires (which
    must reproduce the left-join ``fill_null(0.0)`` byte-identically)."""
    ps, ds, ts = [0, 1, 2], [10, 11, 12], [100, 101, 102]
    rows = list(itertools.product(ps, ds, ts))
    over = pl.DataFrame(
        {"p": [r[0] for r in rows], "d": [r[1] for r in rows], "t": [r[2] for r in rows]}
    )
    # Drop a few (p, d, t) cells from the RHS frame so the left-join surfaces
    # nulls (filled to 0.0) and the dense-completeness guard fails.
    keep = [c for i, c in enumerate(rows) if i % 7 != 0]
    rhs = Param(
        ("p", "d", "t"),
        pl.DataFrame(
            {
                "p": [r[0] for r in keep],
                "d": [r[1] for r in keep],
                "t": [r[2] for r in keep],
                "value": np.linspace(1e-2, 9e2, len(keep)),
            }
        ),
        name="maxToSink_sparse",
    )
    return over, rhs


def test_from_rhs_param_unit_builds_recipe():
    """``from_rhs_param`` on a frame Param (``_sources is None``, non-empty
    dims) builds a Var-less param_only recipe wrapping the single Param as a
    1-element chain with ``coef_scalar == rhs._value_scalar``."""
    _over, rhs = _build_rhs_frame_param()
    assert rhs._sources is None  # genuinely a frame Param
    recipe = CoefWalkRecipe.from_rhs_param(rhs)
    assert recipe.param_only is True
    assert recipe.var_source is None
    assert recipe.sum_block_meta is None
    assert len(recipe.param_sources) == 1
    atomic, direction = recipe.param_sources[0]
    assert atomic is rhs
    assert direction == 1
    assert recipe.coef_scalar == rhs._value_scalar == 1.0


def test_from_rhs_param_rejects_composite_chain():
    """A composite ``_sources`` chain (an algebra result) must route through
    ``from_rhs_chain``, not ``from_rhs_param`` — the latter rejects it."""
    _over, rhs = _build_rhs_chain_dense()
    assert isinstance(rhs._sources, list)
    with pytest.raises(ValueError, match="from_rhs_chain"):
        CoefWalkRecipe.from_rhs_param(rhs)


def test_from_rhs_param_rejects_dimless():
    """A dimless scalar Param has no over-row alignment — rejected (the
    caller's scalar-broadcast branch handles it)."""
    rhs = Param((), pl.DataFrame({"value": [3.5]}))
    with pytest.raises(ValueError, match="non-empty dims"):
        CoefWalkRecipe.from_rhs_param(rhs)


def test_rhs_frame_param_dense_param_only():
    """Frame-Param RHS, dense-complete (positional fast path): byte-identical
    to the whole-collect across batch sizes, with the row factor applied."""
    over, rhs = _build_rhs_frame_param()
    n = over.height
    rf, _cf = _side_vectors(n, 8)
    scale = (rf, 0, None)  # row factor on, NO col factor (RHS has none)
    recipe = CoefWalkRecipe.from_rhs_param(rhs)
    ref = _ref_minmax_rhs_chain(rhs, over, scale)
    assert ref != (None, None)
    for bs in _batch_sizes(n):
        (got,) = bounded_coefficient_walk(
            over,
            recipe,
            scale,
            [MinMaxAbsReducer(scale)],
            batch_rows=bs,
            dense_axes=("d", "t"),
        )
        assert got == ref, f"batch_rows={bs}: {got!r} != {ref!r}"


def test_rhs_frame_param_sparse_param_only():
    """Frame-Param RHS, SPARSE (prune-down backstop): the walk must apply the
    SAME ``fill_null(0.0)`` the whole-collect left-join applies — byte-
    identical across batch sizes (this is the byte-identity guarantee for the
    frame-Param mask/fill that the un-gated ranges-pre RHS readout relies on)."""
    over, rhs = _build_rhs_frame_param_sparse()
    n = over.height
    rf, _cf = _side_vectors(n, 8)
    scale = (rf, 0, None)
    recipe = CoefWalkRecipe.from_rhs_param(rhs)
    ref = _ref_minmax_rhs_chain(rhs, over, scale)
    assert ref != (None, None)
    for bs in _batch_sizes(n):
        (got,) = bounded_coefficient_walk(
            over,
            recipe,
            scale,
            [MinMaxAbsReducer(scale)],
            batch_rows=bs,
            dense_axes=("d", "t"),
        )
        assert got == ref, f"batch_rows={bs}: {got!r} != {ref!r}"


def test_rhs_frame_param_no_scale():
    """Frame-Param RHS with scale all-None (the ranges-PRE pass) reduces raw
    ``|value|`` and still matches the whole-collect, dense AND sparse."""
    for builder in (_build_rhs_frame_param, _build_rhs_frame_param_sparse):
        over, rhs = builder()
        n = over.height
        scale = (None, 0, None)
        recipe = CoefWalkRecipe.from_rhs_param(rhs)
        ref = _ref_minmax_rhs_chain(rhs, over, scale)
        assert ref != (None, None), builder.__name__
        for bs in _batch_sizes(n):
            (got,) = bounded_coefficient_walk(
                over,
                recipe,
                scale,
                [MinMaxAbsReducer(scale)],
                batch_rows=bs,
                dense_axes=("d", "t"),
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
            over,
            recipe,
            scale,
            [MinMaxAbsReducer(scale)],
            batch_rows=bs,
            dense_axes=("d", "t"),
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
        {"p": [c[0] for c in cells], "d": [c[1] for c in cells], "t": [c[2] for c in cells]}
    )
    v = prob.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)
    dt = list(itertools.product(ds, ts))
    Pa = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [c[0] for c in dt],
                "t": [c[1] for c in dt],
                "value": np.linspace(1e-3, 1e2, len(dt)),
            }
        ),
        name="Pa",
    )
    Pb = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [c[0] for c in dt],
                "t": [c[1] for c in dt],
                "value": np.linspace(2.0, 5e3, len(dt)),
            }
        ),
        name="Pb",
    )
    prob.add_cstr("vpp", over=over, sense="<=", lhs_terms={"l": v * Pa * Pb}, rhs_terms={"r": 0.0})
    return prob, over


def _peak_for_batch(over, recipe, scale, batch_rows) -> int:
    tracemalloc.start()
    tracemalloc.reset_peak()
    bounded_coefficient_walk(
        over,
        recipe,
        scale,
        [MinMaxAbsReducer(scale)],
        batch_rows=batch_rows,
        dense_axes=("d", "t"),
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
        over,
        recipe,
        scale,
        [MinMaxAbsReducer(scale)],
        batch_rows=256,
        dense_axes=("d", "t"),
    )
    assert got_small == ref

    peak_small = _peak_for_batch(over, recipe, scale, 256)
    peak_whole = _peak_for_batch(over, recipe, scale, n)

    # The whole-batch peak must be materially larger than the small-batch
    # peak — proving the small batch never holds the full product.  Use a
    # generous margin to stay robust across polars allocator noise.
    assert peak_small * 3 < peak_whole, (
        f"small-batch peak {peak_small} not bounded below whole-batch peak {peak_whole} (n={n})"
    )


# ---------------------------------------------------------------------------
# D1: post-Sum Expr-algebra now FORWARDS the recipe, so these shapes route
# to the WALK (recipe present) instead of the collect fallback.


def _build_neg_sum_relabel():
    """A ``-Sum(v * P_unit * P_step, over=('p','s'))`` LHS — the headline
    ``-Sum(...)`` production constraint shape.  Before D1 the negation
    dropped the recipe; now it rides through with ``coef_scalar`` negated."""
    prob = Problem(dense_axes=("d", "t"))
    ps, ss, ds, ts = [0, 1], ["s0", "s1"], [10, 11], [100, 101, 102]
    rows = list(itertools.product(ps, ss, ds, ts))
    var_index = pl.DataFrame(
        {
            "p": [r[0] for r in rows],
            "s": [r[1] for r in rows],
            "d": [r[2] for r in rows],
            "t": [r[3] for r in rows],
        }
    )
    v = prob.add_var("v", ("p", "s", "d", "t"), var_index, lower=0.0, upper=1e6)
    P_unit = Param(("p",), pl.DataFrame({"p": ps, "value": [2.0, 30.0]}), name="P_unit")
    dt = list(itertools.product(ds, ts))
    P_step = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [r[0] for r in dt],
                "t": [r[1] for r in dt],
                "value": np.linspace(0.5, 1.5, len(dt)),
            }
        ),
        name="P_step",
    )
    nb = -Sum(v * P_unit * P_step, over=("p", "s"))
    nb_over = nb.terms[0].frame.select(list(nb.terms[0].dims)).unique().sort(["d", "t"])
    prob.add_cstr("nb", over=nb_over, sense="<=", lhs_terms={"l": nb}, rhs_terms={"r": 0.0})
    return prob, nb_over


def test_neg_sum_relabel_parity():
    prob, over = _build_neg_sum_relabel()
    term = prob._cstrs[0][1].expr.terms[0]
    rf, cf = _side_vectors(over.height, prob._next_col)
    scale = (rf, 0, cf)
    recipe = CoefWalkRecipe.from_term(term)
    # D1: the negated Sum routes to the WALK (recipe present), not collect.
    assert recipe.sum_block_meta is not None
    assert recipe.var_source is not None
    ref = _ref_minmax_constraint(term, over, scale)
    for bs in BATCH_SIZES:
        (got,) = bounded_coefficient_walk(
            over,
            recipe,
            scale,
            [MinMaxAbsReducer(scale)],
            batch_rows=bs,
            dense_axes=("d", "t"),
        )
        assert got == ref, f"batch_rows={bs}: {got!r} != {ref!r}"


def test_double_sum_objective_parity():
    """``Sum(Sum(v * P, over=(...)), over=None)`` — the objective collapse-
    all shape ``set_objective`` builds.  The outer no-op collapse forwards
    the recipe, so the column-mode walk routes to the WALK; assert log2
    histogram parity to the whole-collect reference."""
    prob = Problem(dense_axes=("d", "t"))
    ps, ds, ts = [0, 1, 2], [10, 11, 12], [100, 101]
    cells = list(itertools.product(ps, ds, ts))
    idx = pl.DataFrame(
        {"p": [c[0] for c in cells], "d": [c[1] for c in cells], "t": [c[2] for c in cells]}
    )
    v = prob.add_var("v", ("p", "d", "t"), idx, lower=0.0, upper=1e6)
    Pcost = Param(
        ("p", "d", "t"),
        pl.DataFrame(
            {
                "p": [c[0] for c in cells],
                "d": [c[1] for c in cells],
                "t": [c[2] for c in cells],
                "value": np.linspace(1e-3, 1e2, len(cells)),
            }
        ),
        name="Pcost",
    )
    # Inner Sum collapses every Var dim -> a scalar (dims == ()) objective
    # term; the outer Sum(over=None) is the no-op collapse-all relabel.
    inner = Sum(v * Pcost, over=("p", "d", "t"))
    assert inner.terms[0].dims == ()
    obj = Sum(inner, over=None)
    term = obj.terms[0]
    assert term.dims == ()
    # D1: collapse-all forwarded the recipe -> column-mode walk path.
    recipe = CoefWalkRecipe.from_term(term)
    assert recipe.sum_block_meta is not None
    assert recipe.var_source is not None

    # Column spine = the Var grid (one row per col_id).
    spine = v.frame
    n_cols = prob._next_col
    _, cf = _side_vectors(0, n_cols)
    scale = (None, 0, cf)  # objective: col factor only

    def classify(cid):
        return "even" if cid % 2 == 0 else "odd"

    ref = _ref_histogram_column(term, scale, classify)
    # The walk routes the forwarded recipe through the in-block rebuild,
    # whose summation order differs from the whole-collect reference — so
    # count / min / max are EXACT and the log2 sum matches within FP
    # reassociation tolerance (mirrors test_vpp_histogram_batched_fp_tol).
    (got,) = bounded_coefficient_walk(
        spine,
        recipe,
        scale,
        [Log2HistogramReducer(scale, classify)],
        batch_rows=spine.height,
        dense_axes=("d", "t"),
    )
    assert set(got) == set(ref)
    for k in ref:
        rs, rn, rmin, rmax = ref[k]
        gs, gn, gmin, gmax = got[k]
        assert gn == rn
        assert gmin == rmin and gmax == rmax
        assert gs == pytest.approx(rs, rel=1e-12, abs=1e-9)


def test_where_after_sum_parity():
    """``Where(Sum(v * P_unit * P_step, over=('p','s')), filter)`` — a
    pure-filter Where AFTER a Sum forwards the recipe with the filter
    recorded into ``where_frames``; the walk must match the whole-collect
    reference byte-for-byte across every batch size and route to the WALK
    (recipe present), not the collect fallback."""
    prob = Problem(dense_axes=("d", "t"))
    ps, ss, ds, ts = [0, 1], ["s0", "s1"], [10, 11, 12], [100, 101, 102]
    rows = list(itertools.product(ps, ss, ds, ts))
    var_index = pl.DataFrame(
        {
            "p": [r[0] for r in rows],
            "s": [r[1] for r in rows],
            "d": [r[2] for r in rows],
            "t": [r[3] for r in rows],
        }
    )
    v = prob.add_var("v", ("p", "s", "d", "t"), var_index, lower=0.0, upper=1e6)
    P_unit = Param(("p",), pl.DataFrame({"p": ps, "value": [2.0, 30.0]}), name="P_unit")
    dt = list(itertools.product(ds, ts))
    P_step = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [r[0] for r in dt],
                "t": [r[1] for r in dt],
                "value": np.linspace(0.5, 1.5, len(dt)),
            }
        ),
        name="P_step",
    )
    nb = Sum(v * P_unit * P_step, over=("p", "s"))
    # Pure-filter Where AFTER the Sum: keep only d in {10, 11} (shared dim,
    # no map extras) — recorded into the recipe's ``where_frames``.
    filt = pl.DataFrame({"d": [10, 11]})
    nbw = Where(nb, filt)
    term = nbw.terms[0]
    # D1: the pure-filter Where forwarded the recipe with the filter
    # recorded into the recipe's ``where_frames``.
    assert term.sum_block_meta is not None
    assert len(term.sum_block_meta.where_frames or ()) == 1

    over = term.frame.select(list(term.dims)).unique().sort(["d", "t"])
    rf, cf = _side_vectors(over.height, prob._next_col)
    scale = (rf, 0, cf)
    recipe = CoefWalkRecipe.from_term(term)
    # D1: the Where-after-Sum term routes to the WALK (recipe present).
    assert recipe.sum_block_meta is not None
    assert recipe.var_source is not None
    ref = _ref_minmax_constraint(term, over, scale)
    for bs in BATCH_SIZES:
        (got,) = bounded_coefficient_walk(
            over,
            recipe,
            scale,
            [MinMaxAbsReducer(scale)],
            batch_rows=bs,
            dense_axes=("d", "t"),
        )
        assert got == ref, f"batch_rows={bs}: {got!r} != {ref!r}"


def _build_map_where_after_sum():
    """``Where(Sum(v * P_unit * P_step, over=('s',)), p_to_n)`` — a
    MAP-EFFECT Where applied AFTER a Sum.  The Sum keeps ``(p, d, t)``; the
    post-Sum Where maps the kept dim ``p`` to a NEW node dim ``n`` (the
    ``nodeBalance`` ``(source,sink)→n`` shape, but with the map introduced
    after the reduction — the D1 forwarding path).

    The map's introduced dim ``n`` is DEFERRED in ``meta.where_map_frames``
    and NOT physically carried by the reduced ``term.lazy``
    (``(p, d, t, col_id, coef)``).  The Var grid (36 rows) sits below the
    default block-COO min-dense gate (100), so the Sum block-COO classifier
    DECLINES for every batch and the walk routes through the reduced-lazy
    fallback (:func:`_reduced_lazy_collect`) — the path that must bake the
    deferred map to materialise ``n`` before the ``_rid`` join.  Returns the
    term, the ``(n, d, t)`` constraint grid, and the map frame (for the
    reference build)."""
    prob = Problem(dense_axes=("d", "t"))
    ps, ss, ds, ts = [0, 1, 2], ["s0", "s1"], [10, 11], [100, 101, 102]
    rows = list(itertools.product(ps, ss, ds, ts))
    var_index = pl.DataFrame(
        {
            "p": [r[0] for r in rows],
            "s": [r[1] for r in rows],
            "d": [r[2] for r in rows],
            "t": [r[3] for r in rows],
        }
    )
    v = prob.add_var("v", ("p", "s", "d", "t"), var_index, lower=0.0, upper=1e6)
    P_unit = Param(("p",), pl.DataFrame({"p": ps, "value": [2.0, 30.0, 7.0]}), name="P_unit")
    dt = list(itertools.product(ds, ts))
    P_step = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [r[0] for r in dt],
                "t": [r[1] for r in dt],
                "value": np.linspace(0.5, 1.5, len(dt)),
            }
        ),
        name="P_step",
    )
    # Sum over ``s`` only → keep (p, d, t).
    nb = Sum(v * P_unit * P_step, over=("s",))
    # Map the kept dim ``p`` to a new node dim ``n`` (fan-in: p0,p2 -> nA).
    p_to_n = pl.DataFrame({"p": ps, "n": ["nA", "nB", "nA"]})
    nbw = Where(nb, p_to_n)
    term = nbw.terms[0]
    # The constraint grid is the post-map (n, d, t) cells.
    over = (
        term.lazy.join(p_to_n.lazy(), on="p", how="inner")
        .select("n", "d", "t")
        .unique()
        .sort(["n", "d", "t"])
        .collect()
    )
    return prob, term, over, p_to_n


def _ref_minmax_map_after_sum(term, over, p_to_n, scale):
    """Reference for a map-after-Sum term: materialise the deferred map
    (inner-join the reduced ``term.lazy`` to ``p_to_n`` to introduce ``n``),
    attach ``_rid`` via the (n, d, t) grid, apply scale, reduce — the
    whole-collect the bounded walk's reduced-lazy fallback must match."""
    over_rid = over.with_columns(_rid=pl.int_range(0, over.height, dtype=pl.Int64))
    materialised = term.lazy.join(p_to_n.lazy(), on="p", how="inner")
    on = [d for d in term.dims if d in over.columns]
    df = (
        over_rid.lazy()
        .join(materialised, on=on, how="inner")
        .select("_rid", "col_id", "coef")
        .collect()
    )
    if df.height == 0:
        return None, None
    rids = df["_rid"].to_numpy().astype(np.int64)
    cids = df["col_id"].to_numpy().astype(np.int64)
    coef = df["coef"].to_numpy().astype(np.float64)
    return _reduce_abs(_scale_vals(rids, cids, coef, scale))


def test_map_where_after_sum_parity(recwarn):
    """A MAP-EFFECT ``Where`` AFTER a ``Sum`` introduces a new dim (``n``)
    via a mapping frame (the ``nodeBalance`` shape).  The walk routes the
    forwarded recipe through the reduced-lazy fallback, which must bake the
    deferred map to materialise ``n`` before the ``_rid`` join.

    Min/max is order-independent (:class:`MinMaxAbsReducer`), so the
    reduced-lazy fallback reproduces the deferred-map support EXACTLY — we
    assert BYTE-IDENTITY to the whole-collect reference across EVERY
    ``batch_rows`` (including the smallest), proving the fallback path is
    exercised and correct at every batch size, with NO warning or crash."""
    prob, term, over, p_to_n = _build_map_where_after_sum()
    # D1: the map-after-Sum term routes to the WALK with the recipe present
    # and the introduced dim DEFERRED (not physically in the reduced lazy).
    recipe = CoefWalkRecipe.from_term(term)
    assert recipe.sum_block_meta is not None
    assert recipe.var_source is not None
    assert recipe.sum_block_meta.where_map_frames is not None
    assert "n" in recipe.reduced_dims
    assert "n" not in recipe.reduced_lazy.collect_schema().names()

    rf, cf = _side_vectors(over.height, prob._next_col)
    scale = (rf, 0, cf)
    ref = _ref_minmax_map_after_sum(term, over, p_to_n, scale)
    # The reference must be a real (non-empty) reduction — guard against a
    # vacuously-passing (None, None) on both sides.
    assert ref != (None, None)
    for bs in BATCH_SIZES:
        (got,) = bounded_coefficient_walk(
            over,
            recipe,
            scale,
            [MinMaxAbsReducer(scale)],
            batch_rows=bs,
            dense_axes=("d", "t"),
        )
        assert got == ref, f"batch_rows={bs}: {got!r} != {ref!r}"
    # No warning surfaced by the fallback bake.
    assert len(recwarn) == 0, [str(w.message) for w in recwarn]


def test_map_where_after_sum_histogram_parity():
    """Log2-histogram parity for the map-after-Sum reduced-lazy fallback.
    Count / min / max combine exactly; the log2 sum matches the
    whole-collect within FP reassociation (the same tolerance the other
    batched-histogram tests use) — confirming the materialised-``n`` support
    is the right one for the bucketed readout too."""
    prob, term, over, p_to_n = _build_map_where_after_sum()
    recipe = CoefWalkRecipe.from_term(term)
    rf, cf = _side_vectors(over.height, prob._next_col)
    scale = (rf, 0, cf)

    def classify(cid):
        return cid % 3

    # Reference histogram over the materialised-n whole-collect.
    over_rid = over.with_columns(_rid=pl.int_range(0, over.height, dtype=pl.Int64))
    materialised = term.lazy.join(p_to_n.lazy(), on="p", how="inner")
    on = [d for d in term.dims if d in over.columns]
    df = (
        over_rid.lazy()
        .join(materialised, on=on, how="inner")
        .select("_rid", "col_id", "coef")
        .collect()
    )
    rids = df["_rid"].to_numpy().astype(np.int64)
    cids = df["col_id"].to_numpy().astype(np.int64)
    coef = df["coef"].to_numpy().astype(np.float64)
    vals = np.abs(_scale_vals(rids, cids, coef, scale))
    ref: dict = {}
    for vval, c in zip(vals.tolist(), cids.tolist()):
        if not math.isfinite(vval) or vval <= 0:
            continue
        bkey = classify(int(c))
        ps_, pn_, pmin_, pmax_ = ref.get(bkey, (0.0, 0, math.inf, 0.0))
        ref[bkey] = (ps_ + math.log2(vval), pn_ + 1, min(pmin_, vval), max(pmax_, vval))
    assert ref  # non-empty reference

    for bs in BATCH_SIZES:
        (got,) = bounded_coefficient_walk(
            over,
            recipe,
            scale,
            [Log2HistogramReducer(scale, classify)],
            batch_rows=bs,
            dense_axes=("d", "t"),
        )
        assert set(got) == set(ref), f"batch_rows={bs}"
        for k in ref:
            rs, rn, rmin, rmax = ref[k]
            gs, gn, gmin, gmax = got[k]
            assert gn == rn, f"batch_rows={bs} bucket={k}"
            assert gmin == rmin and gmax == rmax, f"batch_rows={bs} bucket={k}"
            assert gs == pytest.approx(rs, rel=1e-12, abs=1e-9), f"batch_rows={bs} bucket={k}"
