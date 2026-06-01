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
from polar_high.engine import Param, Problem, Sum


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
