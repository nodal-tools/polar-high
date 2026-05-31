"""Memory-bounding pin for the inline block-COO LHS range readout.

Background
----------
:func:`polar_high.autoscale._ranges._ranges_via_streaming` reads
``(min, max)`` of ``|post-Layer-2 LHS coef|`` per constraint term to pick
the autoscale exponents.  The inline block-COO fast path (Phase D-1) used to
build the WHOLE family's ``(_rid, col_id, coef)`` triple in a single
``_build_block_coo_plan`` / ``_build_sum_block_coo_plan`` collect just to
read a min/max — for a wide family (e.g. the FlexTool DES LP's
``profile_flow_upper_limit``, ~10⁵ rows) that single collect materialised
tens of GB at once and OOM'd a DES run.

The bounding fix routes the inline block-evaluable LHS terms through the
already-bounded :func:`bounded_coefficient_walk`: the SAME block-COO
classify+build now runs per ``batch_rows``-row slice of the family ``over``
spine, folding each batch's magnitude into a running min/max and freeing the
batch before the next.  Peak is bounded by ``batch_rows`` (× the per-factor
numpy buffers), NEVER the full family product.

This module PINS the bounding two ways, with NO tolerance:

* **byte-identical** — the ``RangeReport`` from the bounded path (default
  block-COO ON) is byte-for-byte equal to the streaming-collect reference
  (``POLAR_HIGH_DISABLE_BLOCK_COO=1``), so the batching changed nothing the
  scaler sees (min/max is order-free); and
* **actually batched** — for a family WIDER than the walk's batch size the
  per-batch builder (``_build_constraint_batch_triple``) is invoked MORE
  THAN ONCE (so the whole-family triple was never materialised), and the
  largest single batch handed to it never exceeds the batch-size cap.  A
  reference run on a family NARROWER than the cap takes exactly one batch —
  proving the multi-batch count is the wide family being sliced, not a
  fixed artefact.
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import polars as pl

import polar_high.autoscale._coef_walk as _cw
from polar_high.autoscale import ScalingConfig
from polar_high.autoscale._ranges import _ranges_via_streaming
from polar_high.engine import Param, Problem, Sum


def _config() -> ScalingConfig:
    return ScalingConfig(
        threshold_decades=9.0,
        user_bound_scale=None,
        report_yaml_path=None,
    )


def _clear_guard() -> None:
    os.environ.pop("POLAR_HIGH_DISABLE_BLOCK_COO", None)
    os.environ.pop("POLAR_HIGH_ENABLE_BLOCK_COO", None)
    os.environ.pop("POLAR_HIGH_BLOCK_COO_MIN_DENSE", None)
    os.environ.pop("POLAR_HIGH_BLOCK_COO_PROFILE", None)
    os.environ.pop("POLAR_HIGH_RANGES_PROFILE", None)
    os.environ.pop("POLAR_HIGH_RANGES_MAX_FAMILY_ROWS", None)


def _build_problem(n_p: int) -> Problem:
    """One non-Sum block-evaluable LHS family ``Var(p,d,t) × Pa(d,t) ×
    Pb(d,t)`` with ``n_p`` leading-blocks (so the family has
    ``n_p * n_d * n_t`` rows).  Dense suffix ``(d, t)``; frames built in
    ``itertools.product`` order so the dense_axes sort contract holds."""
    n_d, n_t = 4, 5
    ps, ds, ts = list(range(n_p)), list(range(n_d)), list(range(n_t))
    cells = list(itertools.product(ps, ds, ts))
    w_over = pl.DataFrame(
        {
            "p": [c[0] for c in cells],
            "d": [c[1] for c in cells],
            "t": [c[2] for c in cells],
        }
    )

    prob = Problem(dense_axes=("d", "t"))
    w = prob.add_var("w", ("p", "d", "t"), w_over, lower=0.0, upper=1e6)
    dt2 = list(itertools.product(ds, ts))
    Pa = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [c[0] for c in dt2],
                "t": [c[1] for c in dt2],
                "value": np.linspace(1e-3, 1e2, len(dt2)),
            }
        ),
        name="Pa",
    )
    Pb = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [c[0] for c in dt2],
                "t": [c[1] for c in dt2],
                "value": np.linspace(2.0, 5e3, len(dt2)),
            }
        ),
        name="Pb",
    )
    prob.add_cstr(
        "vpp",
        over=w_over,
        sense="<=",
        lhs_terms={"lhs": w * Pa * Pb},
        rhs_terms={"rhs": 0.0},
    )
    prob.set_objective(Sum(w), sense="min")
    return prob


def _install_side_vectors(prob: Problem) -> None:
    """Layer-2 row/col factors with a multi-decade spread so the readout's
    side-vector multiply is non-trivial."""
    n_rows = sum(over.height for _c, _p, over in prob._cstrs)
    n_cols = int(prob._next_col)
    rf = np.array(
        [10.0 ** ((i % 5) - 2) for i in range(n_rows)], dtype=np.float64
    )
    cf = np.array(
        [10.0 ** ((i % 7) - 3) for i in range(n_cols)], dtype=np.float64
    )
    prob._layer2_row_factor = rf
    prob._layer2_col_factor = cf


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


class _BatchSpy:
    """Wrap ``bounded_coefficient_walk`` to FORCE a tiny ``batch_rows`` and
    record the size of every per-batch ``(_rid, col_id, coef)`` build, so a
    test can assert the family was sliced into >1 bounded batches and that no
    single batch ever exceeded the forced cap (i.e. the WHOLE family triple
    was never built at once)."""

    def __init__(self, batch_rows: int) -> None:
        self.batch_rows = batch_rows
        self.batch_sizes: list[int] = []
        self._orig_walk = _cw.bounded_coefficient_walk
        self._orig_constraint = _cw._build_constraint_batch_triple

    def __enter__(self) -> _BatchSpy:
        spy = self

        def _walk(spine, recipe, scale, reducers, *, batch_rows=1_000_000,
                  dense_axes=None):
            # Force the small batch size regardless of the caller's request
            # so a modest test family is genuinely sliced.
            return spy._orig_walk(
                spine, recipe, scale, reducers,
                batch_rows=spy.batch_rows, dense_axes=dense_axes,
            )

        def _constraint(batch_over, axis_cols, recipe, dense_axes):
            spy.batch_sizes.append(int(batch_over.height))
            return spy._orig_constraint(
                batch_over, axis_cols, recipe, dense_axes
            )

        # The walk is bound into ``_ranges`` via a local import of the
        # ``_coef_walk`` symbol, so patching the module attribute is enough
        # ONLY if the import resolves at call time.  ``_ranges`` imports the
        # symbol once inside ``_ranges_via_streaming``; patch the source
        # module attribute BEFORE that import runs (we do — the patch is
        # active for the whole ``with`` block).
        _cw.bounded_coefficient_walk = _walk
        _cw._build_constraint_batch_triple = _constraint
        return self

    def __exit__(self, *exc) -> None:
        _cw.bounded_coefficient_walk = self._orig_walk
        _cw._build_constraint_batch_triple = self._orig_constraint


def _streaming_reference(prob: Problem, cfg: ScalingConfig) -> tuple:
    """RangeReport via the streaming-collect path (block-COO forced OFF) —
    the byte-identical reference the bounded path must reproduce.  Runs on
    the SAME Problem instance so the ``over`` row order (and hence the
    position-indexed side vectors) is identical."""
    os.environ["POLAR_HIGH_DISABLE_BLOCK_COO"] = "1"
    try:
        return _report_tuple(_ranges_via_streaming(prob, cfg))
    finally:
        os.environ.pop("POLAR_HIGH_DISABLE_BLOCK_COO", None)


def test_inline_block_coo_lhs_is_batched_not_whole_collect() -> None:
    """A WIDE block-evaluable LHS family is read in bounded batches via the
    coefficient walk — the inline whole-family collect is never taken — AND
    the min/max it folds is byte-identical to the streaming-collect
    reference."""
    cfg = _config()
    _clear_guard()
    try:
        os.environ["POLAR_HIGH_BLOCK_COO_MIN_DENSE"] = "1"

        # Family = 40 leading blocks × (4×5 dense) = 800 rows.  Force a
        # 64-row batch so the walk MUST slice it into multiple batches.
        prob = _build_problem(n_p=40)
        _install_side_vectors(prob)
        family_rows = prob._cstrs[0][2].height
        assert family_rows == 800, family_rows

        # Streaming reference FIRST (same Problem, block-COO OFF) — no spy.
        ref = _streaming_reference(prob, cfg)

        # Bounded block-COO path with a forced tiny batch size + a spy on the
        # per-batch builder.
        batch_rows = 64
        with _BatchSpy(batch_rows) as spy:
            rep = _ranges_via_streaming(prob, cfg)
        got = _report_tuple(rep)
    finally:
        _clear_guard()

    # (1) Byte-identical to the streaming reference — batching changed
    # nothing the scaler sees.
    assert got == ref, (
        "bounded-batched block-COO range readout diverged from the "
        f"streaming reference:\n  bounded = {got}\n  stream  = {ref}"
    )

    # (2) The family was actually sliced into MORE THAN ONE bounded batch
    # (the whole-family triple was never materialised).
    assert len(spy.batch_sizes) > 1, (
        "expected the wide LHS family to be walked in multiple bounded "
        f"batches; the per-batch builder was called {len(spy.batch_sizes)} "
        "time(s) — the inline whole-family collect was taken instead of the "
        "bounded walk."
    )
    # (3) No single batch ever exceeded the forced cap — peak is bounded by
    # batch_rows, never the 800-row family.
    assert max(spy.batch_sizes) <= batch_rows, (
        "a per-batch build exceeded the batch-row cap "
        f"({max(spy.batch_sizes)} > {batch_rows}) — the slice is not bounded."
    )
    # (4) Every row of the family was covered exactly once across batches
    # (no rows dropped / double-counted by the slicing).
    assert sum(spy.batch_sizes) == family_rows, (
        f"batch sizes {spy.batch_sizes} sum to {sum(spy.batch_sizes)}, "
        f"expected the full family ({family_rows} rows)."
    )


def test_narrow_family_takes_single_batch() -> None:
    """Control: a family NARROWER than the batch cap is read in exactly ONE
    bounded batch — proving the multi-batch count above is the wide family
    being sliced, not a fixed artefact of routing through the walk."""
    cfg = _config()
    _clear_guard()
    try:
        os.environ["POLAR_HIGH_BLOCK_COO_MIN_DENSE"] = "1"
        # 2 leading blocks × (4×5) = 40 rows < the 64-row cap.
        prob = _build_problem(n_p=2)
        _install_side_vectors(prob)
        family_rows = prob._cstrs[0][2].height
        assert family_rows == 40, family_rows

        with _BatchSpy(64) as spy:
            _ranges_via_streaming(prob, cfg)
    finally:
        _clear_guard()

    assert len(spy.batch_sizes) == 1, (
        "a family smaller than the batch cap must take exactly one batch; "
        f"got {len(spy.batch_sizes)} batches of sizes {spy.batch_sizes}."
    )
    assert spy.batch_sizes[0] == family_rows
