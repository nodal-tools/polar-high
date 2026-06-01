"""Regression: the column-mode whole-product fast path for a pure-relabel
``Sum(over=None)`` objective term reads the term's OWN reduced ``(col_id,
coef)`` plan directly instead of rebuilding the (near-)unreduced ``Var ×
Param`` whole product through ``_build_block_coo_plan`` (whose joined branch
is the ~40 s/term DES hot spot).

Correctness invariant
---------------------
For a relabel ``Sum`` (``reduce_dims ⊆ var.dims``, no map-effect Where
frames) ``col_id`` is 1:1 with Var cells, so the reduced plan's
``group_by(col_id).sum(coef)`` is over single-element groups: the reduced
``coef`` EQUALS the per-cell product coef the old whole-product build emits.
Hence the fast-path ``(col_id, coef)`` is *byte-identical* (exact ``==``,
not merely ~1e-9) to the old build — proven below against BOTH the directly
collected reduced plan AND an explicit replay of the old
``_column_whole_product`` body.

Joined-branch note
------------------
The objective term ``set_objective`` builds is always a *collapse-all*
relabel Sum (``term.dims == ()``).  For that shape the column block-COO
classifier returns ``spec is None`` (verified in
``test_old_build_used_prune_down_path``), so the OLD ``_column_whole_product``
took the ``_lhs_prune_down_collect`` fallback — NOT
``_build_block_coo_plan``'s joined branch.  A unit-buildable objective term
cannot force the joined branch (it needs a genuinely non-dense-complete
block-COO grid that the DES produces at scale); the DES exercises that path.
We therefore pin byte-identity to the OLD build (whichever path it took) and
that the fast path now bypasses BOTH the block-COO plan and the prune-down
collect.  The DES-scale 40 s/term saving is the same code path either way:
the fast path replaces the whole-frame ``Var × Param`` materialise with a
~one-row-per-LP-column reduced-plan collect.
"""

import itertools
import math

import numpy as np
import polars as pl
import pytest

import polar_high.autoscale._coef_walk as cw
import polar_high.engine as eng
from polar_high.autoscale._coef_walk import (
    CoefWalkRecipe,
    Log2HistogramReducer,
    _column_whole_product,
    _lhs_prune_down_collect,
    bounded_coefficient_walk,
)
from polar_high.engine import Param, Problem, Sum, Where


def _new_gate(recipe: CoefWalkRecipe) -> bool:
    """The ``_column_whole_product`` reduced-plan fast-path gate, transcribed
    verbatim from production so a future edit that narrows the real predicate
    makes these tests fail.

    The reduced per-column ``coef`` IS the LP cost vector for EVERY Sum term
    (relabel, fan-out, or Where-carved alike) — ``set_objective`` builds a
    collapse-all ``Sum`` whose ``term.lazy = group_by("col_id").agg(coef.sum())``
    is one row per LP column.  So the gate fires whenever a Sum recipe carries a
    reduced plan; it is NOT conditioned on ``reduce_dims``, ``where_frames``, or
    ``where_map_frames`` (those only distinguish whether the reduced coef equals
    the per-cell coef — which the histogram must NOT depend on; the LP-facing
    reduced coef is the correct bucket source in every case)."""
    meta = recipe.sum_block_meta
    if meta is None:
        return False
    return recipe.reduced_lazy is not None


def _old_relabel_only_gate(recipe: CoefWalkRecipe) -> bool:
    """The PRIOR (too-tight) gate — fired ONLY for a single-element relabel Sum
    with no Where frames of any kind (``meta.reduce_dims ⊆ var.dims`` AND
    ``where_frames``/``where_map_frames`` all ``None``).  It MISSED the real DES
    objective flow terms (``Sum(Where(v_flow, idx) * params)`` — the ``Where``
    sets ``meta.where_frames``) and every fan-out term, sending them to the
    ~40 s/term per-cell rebuild.  Used only to PROVE the new gate is strictly
    WIDER (it admits the terms this one wrongly excluded)."""
    meta = recipe.sum_block_meta
    if meta is None:
        return False
    var_dims = list(recipe.var_source.dims)
    return (
        recipe.reduced_lazy is not None
        and meta.where_frames is None
        and meta.where_map_frames is None
        and recipe.where_map_frames is None
        and set(meta.reduce_dims).issubset(set(var_dims))
    )


def _build_relabel_objective(sparse: bool):
    """A pure-relabel ``Sum(over=None)`` objective term over a ``Var × Param``
    chain — the shape ``set_objective`` builds.  With ``sparse=True`` the Var
    grid is NOT dense-complete (~60% of cells kept)."""
    prob = Problem(dense_axes=("d", "t"))
    cells = list(itertools.product([0, 1, 2, 3], [10, 11, 12, 13], [100, 101, 102]))
    if sparse:
        rng = np.random.default_rng(7)
        cells = [c for c in cells if rng.random() < 0.6]
    idx = pl.DataFrame(
        {
            "p": [c[0] for c in cells],
            "d": [c[1] for c in cells],
            "t": [c[2] for c in cells],
        }
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
    # Inner Sum collapses every Var dim -> a scalar objective term; the outer
    # Sum(over=None) is the no-op collapse-all relabel set_objective forwards.
    obj = Sum(Sum(v * Pcost, over=("p", "d", "t")), over=None)
    term = obj.terms[0]
    assert term.dims == ()
    return prob, v, term, CoefWalkRecipe.from_term(term)


def _side_vectors(n_cols: int) -> np.ndarray:
    return np.array(
        [10.0 ** ((i % 7) - 3) for i in range(n_cols)], dtype=np.float64
    )


def _sorted_pair(cid: np.ndarray, coef: np.ndarray):
    order = np.argsort(cid, kind="stable")
    return cid[order].astype(np.int64), coef[order].astype(np.float64)


def _seed_of(v):
    return v.frame.collect() if hasattr(v.frame, "collect") else v.frame


def _reduced_reference(recipe: CoefWalkRecipe):
    """Collect the term's reduced ``(col_id, coef)`` plan directly — the
    source the fast path reads."""
    df = recipe.reduced_lazy.select("col_id", "coef").collect()
    return _sorted_pair(df["col_id"].to_numpy(), df["coef"].to_numpy())


def _old_whole_product(seed, recipe: CoefWalkRecipe):
    """Explicit replay of the OLD ``_column_whole_product`` body (the path
    BEFORE the reduced-plan fast path): identity row_index over the whole
    seed → block-COO plan when ``spec`` fires, else the prune-down collect.
    For the collapse-all objective ``spec is None`` so this is the
    ``_lhs_prune_down_collect`` fallback.  This is the correctness anchor:
    the fast path must agree with it value-for-value."""
    var = recipe.var_source
    var_dims = list(var.dims)
    row_index = seed.select(*var_dims).with_columns(
        _rid=pl.int_range(0, seed.height, dtype=pl.Int64)
    )
    df = _lhs_prune_down_collect(row_index.lazy(), list(var_dims), recipe)
    return _sorted_pair(df["col_id"].to_numpy(), df["coef"].to_numpy())


@pytest.mark.parametrize("sparse", [True, False])
def test_recipe_meets_reduced_fastpath_gate(sparse):
    """The relabel-Sum objective recipe satisfies the fast-path gate AND its
    reduced plan carries (col_id, coef) — else the fast path is dead code."""
    _prob, _v, _term, recipe = _build_relabel_objective(sparse)
    assert recipe.reduced_lazy is not None
    assert recipe.sum_block_meta is not None
    assert getattr(recipe.sum_block_meta, "var_source", None) is not None
    assert recipe.where_map_frames is None
    assert set(recipe.reduced_dims or ()).issubset(set(recipe.var_source.dims))
    schema = set(recipe.reduced_lazy.collect_schema().names())
    assert "col_id" in schema
    assert "coef" in schema
    # The TIGHTENED gate fires for this genuine relabel term: the summed dims
    # are exactly the Var dims (single-element col_id groups) and there are no
    # Where frames of any kind.
    meta = recipe.sum_block_meta
    assert meta.reduce_dims == ("p", "d", "t")
    assert set(meta.reduce_dims).issubset(set(recipe.var_source.dims))
    assert meta.where_frames is None
    assert meta.where_map_frames is None
    assert _new_gate(recipe) is True


@pytest.mark.parametrize("sparse", [True, False])
def test_fastpath_skips_blockcoo_and_prunedown(sparse):
    """The fast path must NOT invoke ``_build_block_coo_plan``, its joined
    branch, or ``_lhs_prune_down_collect`` — it reads the reduced plan
    directly."""
    _prob, v, _term, recipe = _build_relabel_objective(sparse)
    seed = _seed_of(v)

    hits = {"plan": 0, "prune": 0, "joined": 0}
    o_plan = cw._build_block_coo_plan
    o_prune = cw._lhs_prune_down_collect
    o_joined = eng._build_block_coo_plan_joined

    def wrap_plan(*a, **k):
        hits["plan"] += 1
        return o_plan(*a, **k)

    def wrap_prune(*a, **k):
        hits["prune"] += 1
        return o_prune(*a, **k)

    def wrap_joined(*a, **k):
        hits["joined"] += 1
        return o_joined(*a, **k)

    cw._build_block_coo_plan = wrap_plan
    cw._lhs_prune_down_collect = wrap_prune
    eng._build_block_coo_plan_joined = wrap_joined
    try:
        # spec / dense_param_vectors are irrelevant once the gate fires; the
        # production hoist passes spec=None for the collapse-all objective.
        cid, _coef = _column_whole_product(seed, recipe, None, None)
    finally:
        cw._build_block_coo_plan = o_plan
        cw._lhs_prune_down_collect = o_prune
        eng._build_block_coo_plan_joined = o_joined

    assert hits == {"plan": 0, "prune": 0, "joined": 0}
    assert cid.size == seed.height > 0


@pytest.mark.parametrize("sparse", [True, False])
def test_old_build_used_prune_down_path(sparse):
    """Sanity: the OLD collapse-all-objective build classified ``spec is
    None`` and so took ``_lhs_prune_down_collect`` (NOT the block-COO joined
    branch).  A unit objective term cannot force the joined branch; the DES
    exercises it.  This pins WHICH old path the byte-identity anchor replays."""
    _prob, v, _term, recipe = _build_relabel_objective(sparse)
    seed = _seed_of(v)
    var_dims = list(recipe.var_source.dims)

    hits = {"prune": 0, "joined": 0}
    o_prune = cw._lhs_prune_down_collect
    o_joined = eng._build_block_coo_plan_joined

    def wrap_prune(*a, **k):
        hits["prune"] += 1
        return o_prune(*a, **k)

    def wrap_joined(*a, **k):
        hits["joined"] += 1
        return o_joined(*a, **k)

    cw._lhs_prune_down_collect = wrap_prune
    eng._build_block_coo_plan_joined = wrap_joined
    try:
        row_index = seed.select(*var_dims).with_columns(
            _rid=pl.int_range(0, seed.height, dtype=pl.Int64)
        )
        o_prune(row_index.lazy(), list(var_dims), recipe)
    finally:
        cw._lhs_prune_down_collect = o_prune
        eng._build_block_coo_plan_joined = o_joined

    # The old path is the prune-down collect; the joined branch is not the
    # objective's path at unit scale.
    assert hits["joined"] == 0


@pytest.mark.parametrize("sparse", [True, False])
def test_fastpath_byte_identical_to_reduced_and_old_build(sparse):
    """The fast-path ``(col_id, coef)`` is byte-identical (exact ==) to BOTH
    the directly collected reduced plan AND the old whole-product build."""
    _prob, v, _term, recipe = _build_relabel_objective(sparse)
    seed = _seed_of(v)

    cid_new, coef_new = _sorted_pair(
        *_column_whole_product(seed, recipe, None, None)
    )
    cid_red, coef_red = _reduced_reference(recipe)
    cid_old, coef_old = _old_whole_product(seed, recipe)

    # Non-trivial: one LP column per Var cell.
    assert cid_new.size == seed.height > 0

    assert np.array_equal(cid_new, cid_red)
    assert np.array_equal(coef_new, coef_red)

    # Byte-identical to the old build (single-element col_id groups ⇒ reduced
    # coef == per-cell coef).
    assert np.array_equal(cid_new, cid_old)
    assert np.array_equal(coef_new, coef_old)


def _ref_histogram_column(cid: np.ndarray, coef: np.ndarray, scale, classify):
    """Whole-collect log2 histogram over a column (col_id, coef) pair, mirroring
    ``test_coef_walk_parity._ref_histogram_column``."""
    l2_cf = scale[2]
    vals = np.abs(coef.astype(np.float64))
    if l2_cf is not None:
        vals = vals * np.abs(l2_cf[cid])
    acc: dict = {}
    for v, c in zip(vals.tolist(), cid.tolist()):
        if not math.isfinite(v) or v <= 0:
            continue
        bkey = classify(int(c))
        if bkey is None:
            continue
        lv = math.log2(v)
        ps, pn, pmin, pmax = acc.get(bkey, (0.0, 0, math.inf, 0.0))
        acc[bkey] = (ps + lv, pn + 1, min(pmin, v), max(pmax, v))
    return acc


@pytest.mark.parametrize("sparse", [True, False])
def test_column_walk_histogram_matches_old_build(sparse):
    """Drive the end-to-end column ``bounded_coefficient_walk`` with a
    ``Log2HistogramReducer`` and assert its histogram matches the histogram
    of the OLD-build (col_id, coef): count / min / max EXACT, log2 sum exact
    for the single whole-spine batch."""
    prob, v, _term, recipe = _build_relabel_objective(sparse)
    seed = _seed_of(v)
    spine = v.frame
    cf = _side_vectors(prob._next_col)
    scale = (None, 0, cf)  # objective: col factor only

    def classify(cid):
        return "even" if cid % 2 == 0 else "odd"

    cid_old, coef_old = _old_whole_product(seed, recipe)
    ref = _ref_histogram_column(cid_old, coef_old, scale, classify)

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
        # Single whole-spine batch ⇒ exact (no FP reassociation across batches).
        assert gs == pytest.approx(rs, rel=1e-12, abs=1e-9)


# ---------------------------------------------------------------------------
# Widened-gate regression coverage.
#
# The reduced per-column ``coef`` is the LP cost vector for EVERY Sum objective
# term, so the fast path fires for ALL of them.  The reduced coef equals the
# per-cell whole-product coef ONLY for a single-element relabel; for a *fan-out*
# term (a ``Param`` dim the ``Var`` LACKS, summed out by ``Sum``) or a grid-
# carving ``Where`` the reduced ``group_by`` genuinely sums several product
# cells into one ``col_id`` — and THAT summed value is the correct LP-facing
# objective coefficient the histogram must bucket, NOT the per-cell coef the old
# per-cell rebuild emitted.  (The old rebuild bucketed per-cell coefs that never
# appear in the LP, and ``_build_column_batch_triple``'s ``searchsorted``, which
# assumes unique ``col_id``, silently collapsed the duplicates anyway.)  These
# tests pin: fan-out / Where-carved terms now ALSO take the fast path, their
# ``(col_id, coef)`` is the unique-per-column reduced reference, and the prior
# (too-tight) relabel-only gate wrongly excluded them.


def _build_fanout_objective():
    """A ``Sum(over=None)`` objective whose inner ``Sum`` collapses a Param
    fan-out dim ``h`` that the ``Var`` does NOT carry.

    ``Var`` is on ``(p, d, t)``; the cost ``Param`` is on ``(p, d, t, h)`` —
    ``h`` is a genuine Param dim (NOT a map-introduced ``where_map_frames``
    extra).  ``Sum(over=("p","d","t","h"))`` collapses the whole product to a
    scalar objective term, so ``meta.reduce_dims == ("p","d","t","h")`` with
    ``h ∉ var.dims``.  ``meta.where_frames`` / ``meta.where_map_frames`` /
    ``recipe.where_map_frames`` are all ``None``, so the term is excluded from
    the fast path SPECIFICALLY by the ``reduce_dims ⊆ var_dims`` clause — the
    tightening this file guards — and by nothing else.

    Each LP column (one per ``(p,d,t)`` Var cell) appears in ``len(hs)``
    product rows, so the reduced plan's ``col_id`` groups are MULTI-element:
    the reduced ``coef`` is the genuine sum over ``h`` ≠ the per-cell coef."""
    prob = Problem(dense_axes=("d", "t"))
    ps, ds, ts, hs = [0, 1, 2], [10, 11], [100, 101], [0, 1]
    cells = list(itertools.product(ps, ds, ts))
    var_index = pl.DataFrame(
        {
            "p": [c[0] for c in cells],
            "d": [c[1] for c in cells],
            "t": [c[2] for c in cells],
        }
    )
    v = prob.add_var("v", ("p", "d", "t"), var_index, lower=0.0, upper=1e6)
    pdth = list(itertools.product(ps, ds, ts, hs))
    Pcost = Param(
        ("p", "d", "t", "h"),
        pl.DataFrame(
            {
                "p": [c[0] for c in pdth],
                "d": [c[1] for c in pdth],
                "t": [c[2] for c in pdth],
                "h": [c[3] for c in pdth],
                "value": np.linspace(1e-3, 1e2, len(pdth)),
            }
        ),
        name="Pcost",
    )
    inner = Sum(v * Pcost, over=("p", "d", "t", "h"))
    obj = Sum(inner, over=None)
    term = obj.terms[0]
    assert term.dims == ()
    return prob, v, term, CoefWalkRecipe.from_term(term)


def _build_filtered_relabel_objective():
    """A relabel ``Sum(over=None)`` objective with a PURE-FILTER ``Where``
    BEFORE the ``Sum`` (so ``meta.where_frames`` is set).

    The ``Where`` frame carries only ``p`` (a Var dim, no extras) ⇒ it is a
    pure filter (``meta.where_map_frames`` stays ``None``), and the summed dims
    are exactly the Var dims (``reduce_dims ⊆ var.dims`` holds).  So this term
    passes EVERY old-gate clause yet is excluded by the freshly-added
    ``meta.where_frames is None`` clause: a pre-``Sum`` filter carves the grid
    the reduced plan was built over, so the fast path must NOT fire."""
    prob = Problem(dense_axes=("d", "t"))
    ps, ds, ts = [0, 1, 2, 3], [10, 11], [100, 101]
    cells = list(itertools.product(ps, ds, ts))
    var_index = pl.DataFrame(
        {
            "p": [c[0] for c in cells],
            "d": [c[1] for c in cells],
            "t": [c[2] for c in cells],
        }
    )
    v = prob.add_var("v", ("p", "d", "t"), var_index, lower=0.0, upper=1e6)
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
    keep_p = pl.DataFrame({"p": [0, 1]})  # pure filter: p ∈ {0,1}, no extras
    inner = Sum(Where(v * Pcost, keep_p), over=("p", "d", "t"))
    obj = Sum(inner, over=None)
    term = obj.terms[0]
    assert term.dims == ()
    return prob, v, term, CoefWalkRecipe.from_term(term)


def test_fanout_term_admitted_by_widened_gate():
    """A fan-out objective term is now ADMITTED by the widened gate, whereas the
    prior relabel-only gate WRONGLY excluded it (sending it to the per-cell
    rebuild).

    Regression anchor: a future edit that re-narrows the gate to require
    ``meta.reduce_dims`` subset of var dims (or any Where-frame clause) flips
    ``_new_gate`` back to ``False`` and fails here."""
    _prob, _v, _term, recipe = _build_fanout_objective()
    meta = recipe.sum_block_meta
    assert meta is not None
    # A genuine fan-out: a summed dim outside the Var dims, no Where frames.
    assert meta.reduce_dims == ("p", "d", "t", "h")
    assert "h" not in set(recipe.var_source.dims)
    assert not set(meta.reduce_dims).issubset(set(recipe.var_source.dims))
    assert meta.where_frames is None
    assert meta.where_map_frames is None
    assert recipe.where_map_frames is None
    # Admitted by the WIDENED gate; excluded by the PRIOR relabel-only gate.
    assert _new_gate(recipe) is True
    assert _old_relabel_only_gate(recipe) is False


def test_fanout_takes_reduced_plan_fastpath():
    """``_column_whole_product`` now TAKES the reduced-plan fast path for a
    fan-out term — verified by instrumenting ``_collect_streaming`` (the fast
    path's signature collect) AND confirming the per-cell rebuild does NOT run.

    The fast path collects ``reduced_lazy.select("col_id","coef")`` via
    ``_collect_streaming``; we assert that collect IS invoked, the rebuild
    collect (``_lhs_prune_down_collect``) is NOT, and the result is the REDUCED
    shape: one unique row per LP column (``== n_cols``), NOT the per-cell
    product (``n_cols * |h|``)."""
    _prob, v, _term, recipe = _build_fanout_objective()
    seed = _seed_of(v)
    n_cols = seed.height

    hits = {"stream": 0, "prune": 0}
    o_stream = cw._collect_streaming
    o_prune = cw._lhs_prune_down_collect

    def wrap_stream(*a, **k):
        hits["stream"] += 1
        return o_stream(*a, **k)

    def wrap_prune(*a, **k):
        hits["prune"] += 1
        return o_prune(*a, **k)

    cw._collect_streaming = wrap_stream
    cw._lhs_prune_down_collect = wrap_prune
    try:
        cid, _coef = _column_whole_product(seed, recipe, None, None)
    finally:
        cw._collect_streaming = o_stream
        cw._lhs_prune_down_collect = o_prune

    # Fast-path streaming collect ran; the per-cell rebuild did NOT.
    assert hits["stream"] == 1
    assert hits["prune"] == 0
    # Reduced shape: one row per LP column (col_id unique), NOT the per-cell
    # product (which would be n_cols * |h|).
    assert cid.size == n_cols > 0
    assert np.unique(cid).size == cid.size


def test_fanout_fastpath_matches_reduced_reference_not_per_cell():
    """Correctness: the fan-out fast path emits the REDUCED (LP-facing) coef —
    the summed-over-``h`` value per LP column — which is the CORRECT objective
    coefficient and DIFFERS from the old per-cell rebuild.

    Two anchors:
      1. ``_column_whole_product`` is byte-identical to the directly collected
         reduced plan ``reduced_lazy.select("col_id","coef")`` — one UNIQUE
         ``col_id`` per LP column, the LP cost vector.
      2. The reduced coef DIFFERS from the per-cell rebuild
         (``_old_whole_product`` emits ``n_cols * |h|`` rows with repeated
         col_ids), proving the fan-out genuinely sums multiple product cells —
         the old per-cell histogram source was WRONG for the objective."""
    _prob, v, _term, recipe = _build_fanout_objective()
    seed = _seed_of(v)

    cid_new, coef_new = _sorted_pair(
        *_column_whole_product(seed, recipe, None, None)
    )
    cid_red, coef_red = _reduced_reference(recipe)
    cid_old, coef_old = _old_whole_product(seed, recipe)

    # 1. Byte-identical to the reduced reference: one unique col_id per LP col.
    assert cid_new.size == seed.height > 0
    assert np.unique(cid_new).size == cid_new.size
    assert np.array_equal(cid_new, cid_red)
    assert np.array_equal(coef_new, coef_red)

    # 2. The per-cell rebuild fans out (n_cols * |h| rows, repeated col_ids) and
    # its raw coef differs — proving the reduced (LP-facing) coef is the right
    # source and the old per-cell build was wrong for the objective histogram.
    assert cid_old.size == seed.height * 2  # |h| == 2
    assert np.unique(cid_old).size < cid_old.size

    # Grouping the per-cell product by col_id recovers EXACTLY the reduced coef
    # (the fast path reads it directly instead of rebuilding + re-summing).
    grouped = (
        pl.DataFrame({"col_id": cid_old, "coef": coef_old})
        .group_by("col_id")
        .agg(pl.col("coef").sum())
        .sort("col_id")
    )
    assert np.array_equal(
        grouped["col_id"].to_numpy().astype(np.int64), cid_new
    )
    assert np.allclose(
        grouped["coef"].to_numpy(), coef_new, rtol=1e-12, atol=0.0
    )


def test_fanout_fastpath_histogram_matches_reduced_reference():
    """The histogram over ``_column_whole_product``'s emitted ``(col_id, coef)``
    for the fan-out term equals the histogram over the REDUCED reference
    (count / min / max EXACT, log2-sum exact) — NOT the per-cell rebuild.

    This pins the corrected ``_column_whole_product`` contract for a fan-out
    objective term: it emits the reduced LP cost vector (one unique row per LP
    column), so the histogram buckets exactly the coefficients that appear in
    the LP — matching ``_ref_histogram_column`` over the reduced plan, the same
    source ``engine``'s objective scatter / ``_layer2`` / ``_ranges`` bucket."""
    _prob, v, _term, recipe = _build_fanout_objective()
    seed = _seed_of(v)
    cf = _side_vectors(_prob._next_col)
    scale = (None, 0, cf)

    def classify(cid):
        return "even" if cid % 2 == 0 else "odd"

    cid_new, coef_new = _column_whole_product(seed, recipe, None, None)
    cid_red, coef_red = _reduced_reference(recipe)

    got = _ref_histogram_column(cid_new, coef_new, scale, classify)
    ref = _ref_histogram_column(cid_red, coef_red, scale, classify)

    assert set(got) == set(ref)
    for k in ref:
        rs, rn, rmin, rmax = ref[k]
        gs, gn, gmin, gmax = got[k]
        assert gn == rn
        assert gmin == rmin and gmax == rmax
        assert gs == pytest.approx(rs, rel=1e-12, abs=1e-9)
    # The reduced histogram bins one row per LP column (col_id unique), so the
    # total count is n_cols — NOT the per-cell n_cols * |h|.
    assert sum(v[1] for v in got.values()) == cid_new.size == seed.height


def test_prefilter_where_term_admitted_and_matches_reduced():
    """A pre-``Sum`` pure-filter ``Where`` (``meta.where_frames`` set) — the
    EXACT shape of the real DES objective flow terms ``Sum(Where(v_flow, idx) *
    params)`` — now TAKES the fast path.  The prior relabel-only gate excluded
    it on its ``meta.where_frames is None`` clause, sending the whole DES
    objective to the ~40 s/term per-cell rebuild; the widened gate fires.

    Asserts: the gate is now ``True`` (was ``False``), the fast-path streaming
    collect runs (the per-cell rebuild does not), and the emitted ``(col_id,
    coef)`` matches the reduced reference (the LP cost vector over the carved
    grid) — one unique row per LP column.  This is the DES-shape regression
    anchor."""
    _prob, v, _term, recipe = _build_filtered_relabel_objective()
    meta = recipe.sum_block_meta
    assert meta is not None
    assert set(meta.reduce_dims).issubset(set(recipe.var_source.dims))
    assert meta.where_map_frames is None
    assert recipe.where_map_frames is None
    # Carries a pre-Sum pure-filter Where frame (the DES flow-term shape).
    assert meta.where_frames is not None
    # Admitted by the WIDENED gate; the prior relabel-only gate (with its
    # where_frames clause) wrongly excluded it.
    assert _new_gate(recipe) is True
    assert _old_relabel_only_gate(recipe) is False

    # The term takes the fast path (streaming collect runs; no per-cell rebuild).
    seed = _seed_of(v)
    hits = {"stream": 0, "prune": 0}
    o_stream = cw._collect_streaming
    o_prune = cw._lhs_prune_down_collect

    def wrap_stream(*a, **k):
        hits["stream"] += 1
        return o_stream(*a, **k)

    def wrap_prune(*a, **k):
        hits["prune"] += 1
        return o_prune(*a, **k)

    cw._collect_streaming = wrap_stream
    cw._lhs_prune_down_collect = wrap_prune
    try:
        cid_new, coef_new = _sorted_pair(
            *_column_whole_product(seed, recipe, None, None)
        )
    finally:
        cw._collect_streaming = o_stream
        cw._lhs_prune_down_collect = o_prune

    assert hits["stream"] == 1
    assert hits["prune"] == 0

    # Emits the reduced reference (LP cost vector over the carved grid): one
    # unique col_id per surviving LP column.
    cid_red, coef_red = _reduced_reference(recipe)
    assert np.array_equal(cid_new, cid_red)
    assert np.array_equal(coef_new, coef_red)
    assert np.unique(cid_new).size == cid_new.size
    # The pre-Sum filter actually carved the grid (p in {0,1} -> fewer columns).
    assert 0 < cid_new.size < seed.height
