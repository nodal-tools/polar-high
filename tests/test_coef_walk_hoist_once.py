"""Pin the O(n) hoisting of the bounded coefficient walk.

The walk (:func:`polar_high.autoscale._coef_walk.bounded_coefficient_walk`)
iterates the spine in row-batches.  The batch-INVARIANT, full-frame work —
the dense-axis sort verification (``_verify_dense_sorted``: a whole-Var
collect + struct + ``is_sorted`` scan), the dense-only Param value vectors
(``atomic.lazy.collect().sort()``), and (column / objective mode) the whole
``(col_id, coef)`` product — MUST be computed ONCE per term, not once per
batch.  Doing it per batch is O(n²) (full-frame cost × n_batches).

These tests drive a genuinely MULTI-batch walk (small ``batch_rows`` over a
many-row spine) and assert:

* ``_verify_dense_sorted`` fires EXACTLY ONCE regardless of how many batches
  the walk runs (1 vs many) — the single biggest O(n²) term.
* the dense-only Param vectors are collected ONCE (``_dense_param_vectors``
  called once) and threaded into the engine builders (the cache is
  non-empty), so the builders never re-collect them per batch.
* (column mode) the whole-frame product is computed ONCE
  (``_column_whole_product`` called once) and each batch positionally maps
  into it — no per-batch ``var.frame`` semi-join + collect.
* more than one batch is actually exercised (the per-batch builder runs
  ``>1`` times), so the once-per-term assertions are not vacuous.
* the batched result stays BYTE-IDENTICAL to the single-whole-batch result
  (the hoist computes the same numbers once, not different ones).
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl

import polar_high.autoscale._coef_walk as cw
import polar_high.engine as eng
from polar_high.autoscale._coef_walk import (
    CoefWalkRecipe,
    MinMaxAbsReducer,
    bounded_coefficient_walk,
)
from polar_high.engine import Param, Problem


def _side_vectors(n_rows: int, n_cols: int):
    rf = np.array([10.0 ** ((i % 5) - 2) for i in range(n_rows)], dtype=np.float64)
    cf = np.array([10.0 ** ((i % 7) - 3) for i in range(n_cols)], dtype=np.float64)
    return rf, cf


def _build_vpp_wide():
    """``Var(p,d,t) × Pa(d,t) × Pb(d,t)`` — a dense LHS with a dense-only
    Param chain and enough rows that small ``batch_rows`` yields many
    batches."""
    prob = Problem(dense_axes=("d", "t"))
    ps = [0, 1, 2, 3]
    ds = [10, 11, 12, 13]
    ts = [100, 101, 102, 103, 104, 105]
    cells = list(itertools.product(ps, ds, ts))
    over = pl.DataFrame(
        {
            "p": [c[0] for c in cells],
            "d": [c[1] for c in cells],
            "t": [c[2] for c in cells],
        }
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
    prob.add_cstr(
        "vpp",
        over=over,
        sense="<=",
        lhs_terms={"l": v * Pa * Pb},
        rhs_terms={"r": 0.0},
    )
    return prob, over


def _build_obj_vpp():
    """An OBJECTIVE (column-spine) ``Var(p,d,t) × Pa(d,t)`` term — drives the
    column-mode whole-product hoist."""
    prob = Problem(dense_axes=("d", "t"))
    ps = [0, 1, 2, 3]
    ds = [10, 11, 12, 13]
    ts = [100, 101, 102, 103, 104, 105]
    cells = list(itertools.product(ps, ds, ts))
    frame = pl.DataFrame(
        {
            "p": [c[0] for c in cells],
            "d": [c[1] for c in cells],
            "t": [c[2] for c in cells],
        }
    )
    v = prob.add_var("v", ("p", "d", "t"), frame, lower=0.0, upper=1e6)
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
    return prob, v, Pa


class _Counter:
    """Wrap a callable, counting invocations + recording non-empty returns,
    while delegating verbatim."""

    def __init__(self, fn):
        self._fn = fn
        self.n = 0
        self.any_nonempty = False

    def __call__(self, *a, **k):
        self.n += 1
        out = self._fn(*a, **k)
        if out:
            self.any_nonempty = True
        return out


def test_verify_dense_sorted_hoisted_once_constraint(monkeypatch):
    """``_verify_dense_sorted`` fires EXACTLY ONCE per constraint-mode walk —
    independent of the batch count — and the per-batch builder runs many
    times (so the assertion is not vacuous)."""
    prob, over = _build_vpp_wide()
    term = prob._cstrs[0][1].expr.terms[0]
    rf, cf = _side_vectors(over.height, prob._next_col)
    scale = (rf, 0, cf)
    recipe = CoefWalkRecipe.from_term(term)

    # Reference (single whole-spine batch) and its verify-count baseline.
    n = over.height
    assert n > 10  # genuinely multi-batch at batch_rows=3

    whole = _run_counted(monkeypatch, over, recipe, scale, batch_rows=n, dense_axes=("d", "t"))
    assert whole["verify_n"] == 1, whole
    assert whole["build_n"] == 1  # one whole batch

    small = _run_counted(monkeypatch, over, recipe, scale, batch_rows=3, dense_axes=("d", "t"))
    # The crux: many batches, but the full-frame verify still ran ONCE.
    assert small["build_n"] > 1, "test did not exercise multiple batches"
    assert small["verify_n"] == 1, (
        "verify ran per batch — O(n^2) regression: "
        f"{small['verify_n']} verifies for {small['build_n']} batches"
    )
    # Dense-only Param vectors hoisted once and actually threaded.
    assert small["dpv_n"] == 1
    assert small["dpv_nonempty"], "dense-only Param vectors not hoisted"
    # Byte-identical result regardless of batch count.
    assert small["got"] == whole["got"]


def test_verify_dense_sorted_hoisted_once_column(monkeypatch):
    """Column / objective mode: the whole-frame product + verify are computed
    ONCE; each batch positionally maps into the product."""
    prob, v, Pa = _build_obj_vpp()
    spine = v.frame  # the objective column spine carries col_id
    n = spine.height
    assert n > 10

    term = (v * Pa).terms[0]
    recipe = CoefWalkRecipe.from_term(term)
    _rf, cf = _side_vectors(1, prob._next_col)
    scale = (None, 0, cf)

    whole = _run_counted(monkeypatch, spine, recipe, scale, batch_rows=n, dense_axes=("d", "t"))
    assert whole["verify_n"] == 1
    assert whole["colprod_n"] == 1

    small = _run_counted(monkeypatch, spine, recipe, scale, batch_rows=4, dense_axes=("d", "t"))
    assert small["build_n"] > 1, "test did not exercise multiple batches"
    assert small["verify_n"] == 1, (
        f"verify ran per batch: {small['verify_n']} for {small['build_n']}"
    )
    assert small["colprod_n"] == 1, (
        "whole-frame column product rebuilt per batch — O(n^2) regression: "
        f"{small['colprod_n']} builds for {small['build_n']} batches"
    )
    assert small["got"] == whole["got"]


def _run_counted(monkeypatch, spine, recipe, scale, *, batch_rows, dense_axes):
    """Run one walk with spies on the batch-invariant full-frame ops + the
    per-batch builders, returning the counts and the reducer result."""
    # Force the block-COO path (the dense-card secondary gate defaults to 100;
    # the test grids are smaller) so the hoisted verify / dense-vector work is
    # genuinely exercised — that is the path whose O(n^2) we are pinning.
    monkeypatch.setenv("POLAR_HIGH_BLOCK_COO_MIN_DENSE", "1")
    verify_spy = _Counter(eng._verify_dense_sorted)
    dpv_spy = _Counter(cw._dense_param_vectors)
    colprod_spy = _Counter(cw._column_whole_product)
    con_spy = _Counter(cw._build_constraint_batch_triple)
    col_spy = _Counter(cw._build_column_batch_triple)
    po_spy = _Counter(cw._build_param_only_batch_triple)

    # Patch the names the walk module resolves at call time.
    monkeypatch.setattr(eng, "_verify_dense_sorted", verify_spy)
    monkeypatch.setattr(cw, "_verify_dense_sorted", verify_spy)
    monkeypatch.setattr(cw, "_dense_param_vectors", dpv_spy)
    monkeypatch.setattr(cw, "_column_whole_product", colprod_spy)
    monkeypatch.setattr(cw, "_build_constraint_batch_triple", con_spy)
    monkeypatch.setattr(cw, "_build_column_batch_triple", col_spy)
    monkeypatch.setattr(cw, "_build_param_only_batch_triple", po_spy)

    (got,) = bounded_coefficient_walk(
        spine,
        recipe,
        scale,
        [MinMaxAbsReducer(scale)],
        batch_rows=batch_rows,
        dense_axes=dense_axes,
    )
    return {
        "got": got,
        "verify_n": verify_spy.n,
        "dpv_n": dpv_spy.n,
        "dpv_nonempty": dpv_spy.any_nonempty,
        "colprod_n": colprod_spy.n,
        "build_n": con_spy.n + col_spy.n + po_spy.n,
    }
