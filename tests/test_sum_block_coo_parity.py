"""Parity tests for the Sum-block-COO LHS arm (Phase C-3a).

Background
----------
Phase C-3a teaches block-COO to evaluate a ``Sum``-wrapped
``Var × Param-chain`` term by rebuilding the unreduced product from the
:class:`polar_high.engine.SumBlockMeta` recipe captured at Sum-time
(Phase C-2) and reducing it IN-BLOCK — without going through polars'
join + group_by.  It is wired at Site 1 only
(:meth:`Problem._build_canonical_matrix`) as a sibling to the non-Sum
block-COO arm.

The headline target is ``nodeBalance_eq``:

    Sum(Where(v(p,s,d,t) * P_unit(p), map_(p,s)->n) * P_step(d,t),
        over=("p","s"))

with ``dense_axes=("d","t")``.  ``Where(_, map_)`` introduces node ``n``
(a MAP-effect Where, deferred into ``where_map_frames`` by C-1); C-2
captured the FULL recipe.  Block-COO rebuilds from the recipe and
produces the reduced LP coefficients.

Bit-identity vs bit-equivalence
-------------------------------
Where each ``(keep…, col_id)`` group reduces to a SINGLE row (e.g.
nodeBalance — every flow is a distinct ``col_id`` mapping to one node),
the reduction is a 1-element sum ⇒ BIT-IDENTICAL to polars' group_by
(``test_node_balance_shape_bit_identical``).  When multiple unreduced
rows fall into the same ``(keep…, col_id)`` group (true coef combining),
the block-COO numpy reduction sums in a different order than polars'
hash-group ⇒ bit-EQUIVALENT within ``rtol=1e-9``
(``test_coef_combining_within_tolerance``).

The off switch is ``POLAR_HIGH_DISABLE_BLOCK_COO=1`` (default ON, D5).
These tests need a small dense suffix, so they lower the perf gate via
``POLAR_HIGH_BLOCK_COO_MIN_DENSE=1`` — the gate is perf-only and never
affects correctness.
"""

from __future__ import annotations

import io
import itertools
import os
import sys

import numpy as np
import polars as pl

from polar_high.engine import Param, Problem, Sum, Where

# --------------------------------------------------------------------- #
# Env-guard helpers (mirror tests/test_block_coo_parity.py)             #
# --------------------------------------------------------------------- #


def _clear_guard() -> None:
    os.environ.pop("POLAR_HIGH_DISABLE_BLOCK_COO", None)
    os.environ.pop("POLAR_HIGH_ENABLE_BLOCK_COO", None)
    os.environ.pop("POLAR_HIGH_DISABLE_PRUNE_DOWN", None)
    os.environ.pop("POLAR_HIGH_DISABLE_WHERE_PUSHDOWN", None)
    os.environ.pop("POLAR_HIGH_BLOCK_COO_MIN_DENSE", None)
    os.environ.pop("POLAR_HIGH_BLOCK_COO_PROFILE", None)


def _matrix_arrays(m) -> tuple[list, list, list]:
    """Sorted (col, row, val) triple comparison key + row_lb / row_ub."""
    val = np.asarray(m.val, dtype=np.float64)
    row_idx = np.asarray(m.row_idx, dtype=np.int64)
    col_ptr = np.asarray(m.col_ptr, dtype=np.int64)
    cols = np.repeat(
        np.arange(m.n_cols, dtype=np.int64), np.diff(col_ptr).astype(np.int64)
    )
    order = np.lexsort((row_idx, cols))
    return (
        list(val[order]),
        list(np.asarray(m.row_lb, dtype=np.float64)),
        list(np.asarray(m.row_ub, dtype=np.float64)),
    )


def _snapshot(builder, *, disable: bool, min_dense: int = 1) -> tuple[list, list, list]:
    _clear_guard()
    try:
        if disable:
            os.environ["POLAR_HIGH_DISABLE_BLOCK_COO"] = "1"
        os.environ["POLAR_HIGH_BLOCK_COO_MIN_DENSE"] = str(min_dense)
        prob = builder()
        return _matrix_arrays(prob._build_canonical_matrix())
    finally:
        _clear_guard()


def _sum_block_profile(builder, *, min_dense: int = 1) -> str:
    """Capture the block-COO profile stream for a build under the DEFAULT
    (block-COO ON).  Returns the full stderr text so callers can assert the
    Sum arm fired (``kind=sum``) AND which path it took
    (``path=relabel`` vs ``path=combining``)."""
    _clear_guard()
    os.environ["POLAR_HIGH_BLOCK_COO_PROFILE"] = "1"
    os.environ["POLAR_HIGH_BLOCK_COO_MIN_DENSE"] = str(min_dense)
    buf = io.StringIO()
    old = sys.stderr
    try:
        sys.stderr = buf
        prob = builder()
        prob._build_canonical_matrix()
    finally:
        sys.stderr = old
        _clear_guard()
    return buf.getvalue()


def _sum_block_fires(builder, *, min_dense: int = 1) -> bool:
    """True iff the Sum-block-COO arm fired (``kind=sum`` in the profile
    stream) under the DEFAULT (block-COO ON)."""
    return "kind=sum" in _sum_block_profile(builder, min_dense=min_dense)


# --------------------------------------------------------------------- #
# Builders                                                              #
# --------------------------------------------------------------------- #


def _node_balance_builder() -> Problem:
    """nodeBalance-shaped Problem: ``Sum(Where(v*P_unit, map)*P_step,
    over=("p","s"))`` with ``dense_axes=("d","t")``, dense-complete.

    Each flow (p,s,d,t) is a distinct ``col_id`` mapping (via the map) to
    one node ``n`` — so every ``(d,t,n,col_id)`` reduce group is a single
    element ⇒ block-COO is BIT-IDENTICAL to polars here.
    """
    p_idx = [0, 1]
    s_idx = ["s0", "s1"]
    d_idx = [10, 11]
    t_idx = [100, 101, 102]

    prob = Problem(dense_axes=("d", "t"))
    rows = list(itertools.product(p_idx, s_idx, d_idx, t_idx))
    var_index = pl.DataFrame(
        {
            "p": [r[0] for r in rows],
            "s": [r[1] for r in rows],
            "d": [r[2] for r in rows],
            "t": [r[3] for r in rows],
        }
    )
    v = prob.add_var("v", ("p", "s", "d", "t"), var_index, lower=0.0, upper=1e6)
    P_unit = Param(
        ("p",), pl.DataFrame({"p": p_idx, "value": [2.0, 3.0]}), name="P_unit"
    )
    dt_rows = list(itertools.product(d_idx, t_idx))
    P_step = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [r[0] for r in dt_rows],
                "t": [r[1] for r in dt_rows],
                "value": np.linspace(0.5, 1.5, len(dt_rows)),
            }
        ),
        name="P_step",
    )
    map_rows = list(itertools.product(p_idx, s_idx))
    map_to_n = pl.DataFrame(
        {
            "p": [r[0] for r in map_rows],
            "s": [r[1] for r in map_rows],
            "n": [f"n{(r[0] + (0 if r[1] == 's0' else 1)) % 2}" for r in map_rows],
        }
    )
    lhs = Sum(Where(v * P_unit, map_to_n) * P_step, over=("p", "s"))
    over_frame = lhs.terms[0].frame.select(list(lhs.terms[0].dims)).unique()
    prob.add_cstr(
        "nb",
        over=over_frame,
        sense="<=",
        lhs_terms={"lhs": lhs},
        rhs_terms={"rhs": 0.0},
    )
    return prob


def _coef_combining_builder() -> Problem:
    """A Sum whose reduce produces MULTI-element ``(keep…, col_id)`` groups
    so coefficients actually SUM.

    ``v(g, d, t)`` with a map ``g -> h`` that is ONE-TO-MANY: each ``g``
    maps to three ``h`` rows.  We Sum over ``("g", "h")`` keeping
    ``(d, t)``.  Because the map duplicates each var cell into three ``h``
    rows that are ALL reduced away, the same ``col_id`` (one per (g,d,t)
    cell) lands in the same ``(d, t, col_id)`` group three times — so
    block-COO performs a genuine MULTI-element reduce (``np.add.reduceat``
    over the 3-row group) rather than a 1-element copy.  This exercises the
    reduce-aggregation path; the result is within ``rtol=1e-9`` of polars'
    group_by (bit-EQUIVALENT — a different summation order; here the three
    summands are equal so it is in fact identical, but the SUMMATION PATH
    is the multi-element one).  ``dense_axes=("d","t")``.

    NOTE: a Param keyed on the map-introduced dim ``h`` cannot appear here
    — multiplying such a Param bakes the deferred map eagerly (see
    ``_bake_map_before_mul``), which makes the recipe carry a Param dim
    outside ``var.dims ∪ map_extras`` and the classifier (correctly)
    DECLINES.  So under the firing contract the only multi-element groups
    are equal-valued fan-out duplicates; the reduce still runs the
    multi-element ``reduceat`` branch, which is what this test pins.
    """
    g_idx = [0, 1, 2]
    d_idx = [10, 11]
    t_idx = [100, 101, 102, 103]

    prob = Problem(dense_axes=("d", "t"))
    rows = list(itertools.product(g_idx, d_idx, t_idx))
    var_index = pl.DataFrame(
        {
            "g": [r[0] for r in rows],
            "d": [r[1] for r in rows],
            "t": [r[2] for r in rows],
        }
    )
    v = prob.add_var("v", ("g", "d", "t"), var_index, lower=0.0, upper=1e6)
    P_g = Param(
        ("g",),
        pl.DataFrame({"g": g_idx, "value": [2.0, 3.0, 5.0]}),
        name="P_g",
    )
    dt_rows = list(itertools.product(d_idx, t_idx))
    P_step = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [r[0] for r in dt_rows],
                "t": [r[1] for r in dt_rows],
                "value": np.linspace(0.5, 1.5, len(dt_rows)),
            }
        ),
        name="P_step",
    )
    # One-to-many map: each g -> three distinct h.  All h summed out.
    map_rows = []
    for g in g_idx:
        map_rows.append((g, g * 10))
        map_rows.append((g, g * 10 + 1))
        map_rows.append((g, g * 10 + 2))
    map_g_to_h = pl.DataFrame(
        {"g": [r[0] for r in map_rows], "h": [r[1] for r in map_rows]}
    )
    lhs = Sum(Where(v * P_g, map_g_to_h) * P_step, over=("g", "h"))
    over_frame = lhs.terms[0].frame.select(list(lhs.terms[0].dims)).unique()
    prob.add_cstr(
        "cc",
        over=over_frame,
        sense="<=",
        lhs_terms={"lhs": lhs},
        rhs_terms={"rhs": 0.0},
    )
    return prob


def _non_suffix_builder() -> Problem:
    """A shape the Sum-block-COO classifier DECLINES: the Var dims do NOT
    end in the declared dense suffix ``(d, t)``.

    ``v(d, t, k)`` — dense axes ``(d, t)`` are NOT the trailing dims (``k``
    is) — so the suffix contract fails and the arm must not fire.  Result
    is still correct (it reads the reduced ``term.lazy``).
    """
    d_idx = [10, 11]
    t_idx = [100, 101, 102]
    k_idx = ["a", "b"]

    prob = Problem(dense_axes=("d", "t"))
    rows = list(itertools.product(d_idx, t_idx, k_idx))
    var_index = pl.DataFrame(
        {
            "d": [r[0] for r in rows],
            "t": [r[1] for r in rows],
            "k": [r[2] for r in rows],
        }
    )
    v = prob.add_var("v", ("d", "t", "k"), var_index, lower=0.0, upper=1e6)
    P_k = Param(
        ("k",), pl.DataFrame({"k": k_idx, "value": [2.0, 3.0]}), name="P_k"
    )
    dt_rows = list(itertools.product(d_idx, t_idx))
    P_step = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [r[0] for r in dt_rows],
                "t": [r[1] for r in dt_rows],
                "value": np.linspace(0.5, 1.5, len(dt_rows)),
            }
        ),
        name="P_step",
    )
    lhs = Sum(v * P_k * P_step, over=("k",))
    over_frame = lhs.terms[0].frame.select(list(lhs.terms[0].dims)).unique()
    prob.add_cstr(
        "nf",
        over=over_frame,
        sense="<=",
        lhs_terms={"lhs": lhs},
        rhs_terms={"rhs": 0.0},
    )
    return prob


# --------------------------------------------------------------------- #
# Tests                                                                 #
# --------------------------------------------------------------------- #


def test_node_balance_shape_bit_identical():
    """nodeBalance: ``reduce_dims=(p,s) ⊆ var.dims`` ⇒ single-element reduce
    groups ⇒ the RELABEL fast-path fires (``path=relabel``) and is
    BIT-IDENTICAL to the block-COO-off (reduced term.lazy) path."""
    snap_off = _snapshot(_node_balance_builder, disable=True)
    snap_on = _snapshot(_node_balance_builder, disable=False)
    assert snap_on == snap_off, (
        "nodeBalance Sum-block-COO must be byte-identical to the "
        "block-COO-off (reduced term.lazy) path"
    )
    prof = _sum_block_profile(_node_balance_builder)
    assert "kind=sum" in prof, (
        "the Sum-block-COO arm must fire for the nodeBalance shape"
    )
    assert "path=relabel" in prof, (
        "nodeBalance (reduce_dims ⊆ var.dims) must take the RELABEL "
        "fast-path, not the materialize-then-reduce combining path"
    )
    assert "path=combining" not in prof


def test_coef_combining_within_tolerance():
    """Multi-element reduce groups (true coef combining) ⇒ bit-EQUIVALENT
    ON vs OFF within rtol=1e-9 (different summation order than polars)."""
    snap_off = _snapshot(_coef_combining_builder, disable=True)
    snap_on = _snapshot(_coef_combining_builder, disable=False)
    prof = _sum_block_profile(_coef_combining_builder)
    assert "kind=sum" in prof, (
        "the Sum-block-COO arm must fire for the coef-combining shape"
    )
    assert "path=combining" in prof, (
        "coef-combining (reduce dim h ∉ var.dims) must take the "
        "materialize-then-reduce path, not the relabel fast-path"
    )
    # row_lb / row_ub must match exactly (they don't go through the reduce).
    assert snap_on[1] == snap_off[1]
    assert snap_on[2] == snap_off[2]
    # Coefficient vectors: same length, equal within rtol=1e-9.
    assert len(snap_on[0]) == len(snap_off[0])
    np.testing.assert_allclose(
        np.asarray(snap_on[0]), np.asarray(snap_off[0]), rtol=1e-9, atol=0.0
    )


def test_sum_block_falls_back():
    """A Var whose dims do NOT end in the dense suffix ⇒ the arm declines
    (does NOT fire) and the result is still correct (reduced term.lazy)."""
    assert not _sum_block_fires(_non_suffix_builder), (
        "non-suffix Var must NOT fire the Sum-block-COO arm"
    )
    snap_off = _snapshot(_non_suffix_builder, disable=True)
    snap_on = _snapshot(_non_suffix_builder, disable=False)
    assert snap_on == snap_off, (
        "declined Sum term must read its reduced term.lazy verbatim "
        "(byte-identical to block-COO-off)"
    )


def _wide_relabel_builder() -> Problem:
    """A LARGE relabel-shaped Problem whose map FANS each flow OUT to many
    rows, over a LONG dense axis — so the unreduced product the
    materialize-then-reduce path builds + sorts (the map fan-out applied
    BEFORE the coef chain) is much larger than the relabel fast-path's
    per-factor numpy buffers (coef computed on the SMALL pre-map Var seed,
    fanned out only at emit).

    ``v(p, s, d, t)`` with ``dense_axes=("d","t")``; the map ``(p,s)->n``
    fans each ``(p,s)`` flow to ``FANOUT`` distinct nodes.  Every fanned
    row is a DISTINCT ``(n, d, t, col_id)`` group ⇒ still single-element ⇒
    still a pure RELABEL (no summation).  ``Sum(over=("p","s"))`` ⇒
    ``reduce_dims=(p,s) ⊆ var.dims``.

    The asymmetry this pins:
      * relabel  — coef chain on the seed (n_p*n_s*n_d*n_t rows), fan-out
        applied once at emit;
      * combining — fan-out applied FIRST, coef chain + full sort + reduceat
        on the fanned (FANOUT× larger) product.
    """
    n_p = 30
    n_s = 2
    n_d = 2
    n_t = 48
    fanout = 6

    p_idx = list(range(n_p))
    s_idx = [f"s{j}" for j in range(n_s)]
    d_idx = list(range(10, 10 + n_d))
    t_idx = list(range(100, 100 + n_t))

    prob = Problem(dense_axes=("d", "t"))
    rows = list(itertools.product(p_idx, s_idx, d_idx, t_idx))
    var_index = pl.DataFrame(
        {
            "p": [r[0] for r in rows],
            "s": [r[1] for r in rows],
            "d": [r[2] for r in rows],
            "t": [r[3] for r in rows],
        }
    )
    v = prob.add_var("v", ("p", "s", "d", "t"), var_index, lower=0.0, upper=1e6)
    P_unit = Param(
        ("p",),
        pl.DataFrame({"p": p_idx, "value": [1.0 + (i % 5) for i in p_idx]}),
        name="P_unit",
    )
    dt_rows = list(itertools.product(d_idx, t_idx))
    P_step = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [r[0] for r in dt_rows],
                "t": [r[1] for r in dt_rows],
                "value": np.linspace(0.5, 1.5, len(dt_rows)),
            }
        ),
        name="P_step",
    )
    # Fan-out map: each (p, s) flow -> ``fanout`` distinct nodes.  Each
    # fanned row keeps the flow's distinct col_id ⇒ single-element group.
    map_rows = []
    for p, s in itertools.product(p_idx, s_idx):
        for k in range(fanout):
            map_rows.append((p, s, f"n{p}_{s}_{k}"))
    map_to_n = pl.DataFrame(
        {
            "p": [r[0] for r in map_rows],
            "s": [r[1] for r in map_rows],
            "n": [r[2] for r in map_rows],
        }
    )
    lhs = Sum(Where(v * P_unit, map_to_n) * P_step, over=("p", "s"))
    over_frame = lhs.terms[0].frame.select(list(lhs.terms[0].dims)).unique()
    prob.add_cstr(
        "nb",
        over=over_frame,
        sense="<=",
        lhs_terms={"lhs": lhs},
        rhs_terms={"rhs": 0.0},
    )
    return prob


def _capture_sum_block_args(builder):
    """Build the matrix once with profiling ON, wrapping the relabel builder
    to capture the exact ``(row_index_lf, axis_cols, meta, on, dense_spec)``
    the dispatch site passes — so the memory test can drive both the relabel
    fast-path and the materialize-then-reduce path on the SAME recipe."""
    import polar_high.engine as eng

    captured: dict = {}
    orig = eng._build_sum_block_coo_relabel

    def _spy(row_index_lf, axis_cols, meta, on, dense_spec, keep_dims=None):
        captured["args"] = (row_index_lf, axis_cols, meta, on, dense_spec, keep_dims)
        return orig(row_index_lf, axis_cols, meta, on, dense_spec, keep_dims)

    _clear_guard()
    os.environ["POLAR_HIGH_BLOCK_COO_MIN_DENSE"] = "1"
    eng._build_sum_block_coo_relabel = _spy
    try:
        prob = builder()
        prob._build_canonical_matrix()
    finally:
        eng._build_sum_block_coo_relabel = orig
        _clear_guard()
    assert "args" in captured, "relabel builder did not fire"
    return captured["args"]


def test_builder_peak_memory_bounded_per_block():
    """RELABEL fast-path memory win — ACHIEVED (the C-3a strict-xfail is now
    a passing test for the relabel case).

    For a relabel-shaped term (``reduce_dims ⊆ var.dims``) every reduce
    group is single-element, so the fast-path computes coef POSITIONALLY on
    the pre-sorted Var seed (peak bounded by the seed + per-factor numpy
    buffers) and emits via streaming joins — it NEVER sorts the full
    unreduced product nor builds the struct-shift mask + ``np.add.reduceat``
    boundary arrays the materialize-then-reduce path needs.

    We drive BOTH paths on the SAME captured recipe and measure peak
    tracemalloc.  The relabel path's peak must be materially below the
    materialize-then-reduce path's (the combining body, forced here by
    flipping only the branch predicate via a phantom reduce dim — the
    combining body groups by ``(*keep, col_id)`` regardless, so the produced
    matrix is identical, only the memory profile differs).

    The genuine-combining path (reduce dim ∉ var.dims) remains
    materialize-then-reduce; that is acceptable because there the reduce is
    a real summation that needs the contiguous-group sort, and it is pinned
    correct by ``test_coef_combining_within_tolerance``.
    """
    import tracemalloc

    import polar_high.engine as eng

    row_index_lf, axis_cols, meta, on, dense_spec, keep_dims = (
        _capture_sum_block_args(_wide_relabel_builder)
    )

    # reduce_dims ⊆ var.dims here — confirm we are measuring the relabel case.
    assert set(dense_spec["reduce_dims"]).issubset(set(meta.var_source.dims))

    # --- Relabel fast-path peak.
    tracemalloc.start()
    out_relabel = eng._build_sum_block_coo_relabel(
        row_index_lf, axis_cols, meta, on, dense_spec, keep_dims
    )
    _cur, peak_relabel = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- Materialize-then-reduce (combining) peak on the SAME recipe.  Flip
    # ONLY the branch predicate by appending a phantom reduce dim absent from
    # var.dims; the combining body never reads reduce_dims (it groups on
    # (*keep, col_id)), so the produced matrix is identical.
    combining_spec = dict(dense_spec)
    combining_spec["reduce_dims"] = list(dense_spec["reduce_dims"]) + ["__phantom"]
    assert not set(combining_spec["reduce_dims"]).issubset(
        set(meta.var_source.dims)
    )
    tracemalloc.start()
    out_combining = eng._build_sum_block_coo_plan(
        row_index_lf, axis_cols, meta, on, combining_spec, keep_dims
    )
    _cur, peak_combining = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Same coefficients (as a set of (_rid, col_id, coef) triples) — the two
    # paths are equivalent here; only the memory profile differs.
    key = ["_rid", "col_id"]
    a = out_relabel.sort(key)
    b = out_combining.sort(key)
    assert a.height == b.height
    np.testing.assert_allclose(
        a["coef"].to_numpy(), b["coef"].to_numpy(), rtol=0.0, atol=0.0
    )

    # The memory win: relabel peak is materially below materialize-then-
    # reduce.  Use a robust 0.75x margin (the avoided full-product sort +
    # struct-shift mask + reduceat boundary arrays are several buffers of the
    # full product; relabel keeps only the seed coef buffer + the small map /
    # row_index join outputs).
    assert peak_relabel < 0.75 * peak_combining, (
        f"relabel peak {peak_relabel} not materially below combining peak "
        f"{peak_combining} (memory win not realized)"
    )
