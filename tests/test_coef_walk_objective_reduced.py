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
    """The TIGHTENED ``_column_whole_product`` reduced-plan fast-path gate,
    transcribed verbatim from production so a future edit that loosens the
    real predicate makes these tests fail.

    The two load-bearing tightenings vs the prior version: it checks
    ``meta.reduce_dims`` (the dims actually SUMMED) — NOT ``recipe.reduced_dims``
    (the post-Sum ``keep``, which is ``()`` for an objective ⇒ vacuously true) —
    and it requires ``meta.where_frames is None`` (a pure-filter ``Where`` before
    the ``Sum`` carves the grid, so the reduced coef is not the per-cell coef)."""
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


def _old_keep_gate(recipe: CoefWalkRecipe) -> bool:
    """The PRIOR (loose) gate — keyed on ``recipe.reduced_dims`` (the
    post-Sum ``keep``) instead of ``meta.reduce_dims`` (the summed dims), and
    WITHOUT the ``meta.where_frames is None`` clause.  For a collapse-all
    objective ``keep == ()`` so the ``issubset`` is vacuously true: this gate
    WRONGLY admits a fan-out / pure-filter term to the fast path.  Used only to
    PROVE the new gate is strictly tighter (the regression this file guards)."""
    meta = recipe.sum_block_meta
    if meta is None:
        return False
    var_dims = list(recipe.var_source.dims)
    return (
        recipe.reduced_lazy is not None
        and meta.where_map_frames is None
        and recipe.where_map_frames is None
        and set(recipe.reduced_dims or ()).issubset(set(var_dims))
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
# Tightened-gate regression coverage.
#
# The reduced-plan fast path is byte-identical to the per-cell whole-product
# build ONLY when every ``col_id`` group in the reduced plan is single-element.
# A *fan-out* term — a ``Param`` carrying a dim the ``Var`` LACKS, summed out by
# ``Sum`` — collapses MANY product cells into one ``col_id``, so the reduced
# ``coef`` is a genuine SUM ≠ the per-cell coef ``_column_whole_product`` must
# return.  The tightened gate (``meta.reduce_dims ⊆ var.dims``) EXCLUDES such
# terms; the prior keep-based gate (``recipe.reduced_dims ⊆ var.dims``, with
# ``keep == ()`` for an objective) WRONGLY admitted them.  These tests pin both
# the exclusion AND that the excluded term still gets a correct per-cell result
# via the rebuild path.


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


def test_fanout_term_excluded_by_tightened_gate():
    """The key new assertion: a fan-out objective term is EXCLUDED by the
    tightened gate, while the prior keep-based gate WRONGLY admitted it.

    This is the regression anchor: a future edit that reverts the gate to key
    on ``recipe.reduced_dims`` (the empty ``keep``) — or drops the
    ``meta.reduce_dims`` check — flips ``_new_gate`` back to ``True`` and fails
    here."""
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
    # Excluded by the TIGHTENED gate; admitted by the PRIOR (vacuous-keep) gate.
    assert _new_gate(recipe) is False
    assert recipe.reduced_dims == ()  # the post-Sum keep is empty …
    assert _old_keep_gate(recipe) is True  # … so the old gate was vacuously true


def test_fanout_skips_reduced_plan_fastpath():
    """``_column_whole_product`` must NOT take the reduced-plan fast path for a
    fan-out term — verified by instrumenting ``_collect_streaming`` (the fast
    path's signature collect) AND confirming the rebuild path runs instead.

    The fast path collects ``reduced_lazy.select("col_id","coef")`` via
    ``_collect_streaming``; the rebuild path collects the per-cell product via
    ``_lhs_prune_down_collect`` (``spec=None`` ⇒ no block-COO plan).  We assert
    the fast-path collect is NOT invoked and the rebuild collect IS, and that
    the result has one row per *product* cell (``> n_cols``), not one per LP
    column (the reduced fast-path shape)."""
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

    # Fast-path streaming collect NEVER ran; the rebuild collect DID.
    assert hits["stream"] == 0
    assert hits["prune"] == 1
    # Rebuild emits the UNREDUCED per-cell product (one row per (p,d,t,h)),
    # which is strictly more rows than the n_cols the fast path would emit.
    assert cid.size > n_cols > 0
    assert cid.size == n_cols * 2  # |h| == 2


def test_fanout_rebuild_matches_per_cell_reference_and_genuine_objective():
    """Correctness: the fan-out term still gets the CORRECT per-cell result via
    the rebuild path.

    Two anchors:
      1. ``_column_whole_product`` is byte-identical to the per-cell rebuild
         reference ``_old_whole_product`` (one row per product cell, the
         documented ``_column_whole_product`` contract).
      2. Grouping that per-cell product by ``col_id`` and summing recovers the
         GENUINE reduced objective coefficient (``reduced_lazy`` == one summed
         coef per LP column) — proving the rebuild carries the full coefficient
         support a fast-path raw-reduced collect would have *also* produced but
         in the WRONG (already-summed) per-row shape."""
    _prob, v, _term, recipe = _build_fanout_objective()
    seed = _seed_of(v)

    cid_new, coef_new = _sorted_pair(
        *_column_whole_product(seed, recipe, None, None)
    )
    cid_old, coef_old = _old_whole_product(seed, recipe)

    # 1. Per-cell byte-identity to the rebuild reference.
    assert cid_new.size == seed.height * 2 > 0  # |h| product fan-out
    assert np.array_equal(cid_new, cid_old)
    assert np.array_equal(coef_new, coef_old)

    # The per-cell product genuinely fans out: col_ids repeat (NOT 1:1 with
    # LP columns), so a raw reduced-plan fast path would have been wrong here.
    assert np.unique(cid_new).size == seed.height
    assert cid_new.size > np.unique(cid_new).size

    # 2. Grouping the per-cell product by col_id recovers the genuine reduced
    # objective coefficient (the summed-over-h coef per LP column).
    grouped = (
        pl.DataFrame({"col_id": cid_new, "coef": coef_new})
        .group_by("col_id")
        .agg(pl.col("coef").sum())
        .sort("col_id")
    )
    reduced = (
        recipe.reduced_lazy.select("col_id", "coef").collect().sort("col_id")
    )
    assert np.array_equal(
        grouped["col_id"].to_numpy().astype(np.int64),
        reduced["col_id"].to_numpy().astype(np.int64),
    )
    assert np.allclose(
        grouped["coef"].to_numpy(),
        reduced["coef"].to_numpy(),
        rtol=1e-12,
        atol=0.0,
    )


def test_fanout_column_product_histogram_matches_per_cell_reference():
    """The histogram over ``_column_whole_product``'s emitted ``(col_id,
    coef)`` for the fan-out term equals the histogram over the per-cell rebuild
    reference (count / min / max EXACT, log2-sum exact).

    This pins ``_column_whole_product``'s CONTRACT for an excluded term: it
    emits the per-cell product (one row per ``(p,d,t,h)``), so each per-cell
    magnitude bins separately — the reference is the per-cell rebuild, NOT the
    already-summed reduced plan.  (The downstream column-mode ``searchsorted``
    lookup in ``bounded_coefficient_walk`` assumes col_id is 1:1 with LP
    columns and so only supports relabel objective terms — the production
    objective path only routes 1:1 terms through the column walk — which is why
    this asserts at the ``_column_whole_product`` boundary the gate governs,
    not the full walk.)"""
    _prob, v, _term, recipe = _build_fanout_objective()
    seed = _seed_of(v)
    cf = _side_vectors(_prob._next_col)
    scale = (None, 0, cf)

    def classify(cid):
        return "even" if cid % 2 == 0 else "odd"

    cid_new, coef_new = _column_whole_product(seed, recipe, None, None)
    cid_old, coef_old = _old_whole_product(seed, recipe)

    got = _ref_histogram_column(cid_new, coef_new, scale, classify)
    ref = _ref_histogram_column(cid_old, coef_old, scale, classify)

    assert set(got) == set(ref)
    for k in ref:
        rs, rn, rmin, rmax = ref[k]
        gs, gn, gmin, gmax = got[k]
        assert gn == rn
        assert gmin == rmin and gmax == rmax
        assert gs == pytest.approx(rs, rel=1e-12, abs=1e-9)
    # The per-cell histogram bins EVERY product row (one per (p,d,t,h)), so
    # each bucket's count is the per-cell count, NOT the reduced-column count.
    assert sum(v[1] for v in got.values()) == cid_new.size


def test_prefilter_where_term_excluded_by_where_frames_clause():
    """A pre-``Sum`` pure-filter ``Where`` sets ``meta.where_frames`` while the
    summed dims stay ⊆ Var dims and no map frames exist — so it passes the
    ``reduce_dims`` / ``where_map_frames`` clauses yet is excluded by the
    freshly-added ``meta.where_frames is None`` clause.

    Also confirms the excluded term still gets the correct per-cell result via
    the rebuild path (no fast-path streaming collect)."""
    _prob, v, _term, recipe = _build_filtered_relabel_objective()
    meta = recipe.sum_block_meta
    assert meta is not None
    # Passes the reduce_dims subset check and has no map frames …
    assert set(meta.reduce_dims).issubset(set(recipe.var_source.dims))
    assert meta.where_map_frames is None
    assert recipe.where_map_frames is None
    # … but carries a pre-Sum pure-filter Where frame.
    assert meta.where_frames is not None
    # Excluded by the new where_frames clause; the old gate (no such clause)
    # would have admitted it.
    assert _new_gate(recipe) is False
    assert _old_keep_gate(recipe) is True

    # The excluded term takes the rebuild path (no fast-path streaming collect)
    # and is byte-identical to the per-cell rebuild reference.
    seed = _seed_of(v)
    hits = {"stream": 0}
    o_stream = cw._collect_streaming

    def wrap_stream(*a, **k):
        hits["stream"] += 1
        return o_stream(*a, **k)

    cw._collect_streaming = wrap_stream
    try:
        cid_new, coef_new = _sorted_pair(
            *_column_whole_product(seed, recipe, None, None)
        )
    finally:
        cw._collect_streaming = o_stream

    assert hits["stream"] == 0
    cid_old, coef_old = _old_whole_product(seed, recipe)
    assert np.array_equal(cid_new, cid_old)
    assert np.array_equal(coef_new, coef_old)
    # The pre-Sum filter actually carved the grid (p ∈ {0,1} ⇒ fewer columns).
    assert cid_new.size > 0
    assert cid_new.size < seed.height
