"""A recipe-less dim-bound LHS term keeps the materialising collect.

Regression for the Layer-3 (``_ranges_via_streaming``) range-readout crash on
``family=nodeBalance_eq term_idx=4`` (confirmed from a real DES traceback):

    File ".../polar_high/autoscale/_ranges.py", in _ranges_via_streaming
        recipe = _CoefWalkRecipe.from_term(term)
    File ".../polar_high/autoscale/_coef_walk.py", in __init__
        raise TypeError("CoefWalkRecipe.var_source must be a Var; got NoneType")

A fully-collapsed ``Sum(over=ALL)`` LHS term ends up with BOTH ``var_source``
and ``sum_block_meta`` cleared (no block-COO recipe can be built) yet keeps a
non-empty ``term.dims`` (the constraint's broadcast axes) and a real ``over``
grid; its ``.lazy`` is the already-reduced ``(*dims, col_id, coef)`` constant
broadcast across the family rows.  The matrix-LHS bounded-coefficient-walk
fallback called ``_CoefWalkRecipe.from_term(term)`` WITHOUT first checking the
term could produce a recipe, so it raised.  Layer 3 catches that and SILENTLY
reverts the solve to an un-scaled LP — a real correctness degradation (only
L1/L2 applied).

The fix mirrors the FlexTool consumer's ``routable`` predicate
(``_layer2.bucket_coefficients``): a non-routable dim-bound term (``var_source``
None AND ``sum_block_meta`` None) skips the walk and falls through to a bounded
materialising-collect backstop that STILL folds its coefficient magnitude into
the matrix range, byte-identically to the pre-block-COO dim-bound readout.

Constructing the genuine shape
------------------------------
The polar-high DSL keeps ``sum_block_meta`` populated for the simple ``Sum``
reductions reachable in a unit test, so to pin the EXACT field combination the
DES run hit (BOTH ``None``, non-empty dims, reduced ``.lazy``) these tests take
a real ``Sum``-reduced term and clear both source slots — exactly the
post-reduction state the crashing DES term carried (the engine itself clears
both slots on terms it cannot block-evaluate, e.g. ``_Term.sum_block_meta =
None`` after certain rebuilds).  ``from_term`` is asserted to raise on that
state (the crash precondition), then the readout is asserted to (a) NOT raise
and (b) report a matrix range byte-identical to an independent reference
reduction of the term's ``.lazy`` with the SAME side-vector scale.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import polars as pl
import pytest

from polar_high.autoscale._coef_walk import CoefWalkRecipe
from polar_high.autoscale._coef_walk import CoefWalkRecipe as _CWR
from polar_high.autoscale._config import ScalingConfig
from polar_high.autoscale._ranges import _ranges_via_streaming
from polar_high.engine import Param, Problem, Sum, Where
from polar_high.engine import _align_enum_join_keys as _align


def _build_problem() -> Problem:
    """A nodeBalance-shaped constraint over ``(d, t, n)`` whose single LHS
    term is a map-relabel ``Sum`` (non-empty dims, real ``over``, reduced
    ``.lazy`` carrying ``(*dims, col_id, coef)``)."""
    p_idx = [0, 1, 2]
    s_idx = ["s0", "s1"]
    d_idx = [10, 11]
    t_idx = [100, 101]
    prob = Problem(dense_axes=("d", "t"))
    vr = list(itertools.product(p_idx, s_idx, d_idx, t_idx))
    vidx = pl.DataFrame(
        {
            "p": [r[0] for r in vr],
            "s": [r[1] for r in vr],
            "d": [r[2] for r in vr],
            "t": [r[3] for r in vr],
        }
    )
    v = prob.add_var("v", ("p", "s", "d", "t"), vidx)
    P = Param(
        ("p",),
        pl.DataFrame({"p": p_idx, "value": [2.0, 30.0, 400.0]}),
        name="P",
    )
    mrows = list(itertools.product(p_idx, s_idx))
    mp = pl.DataFrame(
        {
            "p": [r[0] for r in mrows],
            "s": [r[1] for r in mrows],
            "n": [f"n{(r[0]) % 2}" for r in mrows],
        }
    )
    nb = Sum(Where(v * P, mp), over=("p", "s"))
    nb_over = nb.terms[0].frame.select(list(nb.terms[0].dims)).unique().sort(["n", "d", "t"])
    prob.add_cstr(
        "nodeBalance_eq",
        over=nb_over,
        sense="==",
        lhs_terms={"e": nb},
        rhs_terms={"rhs": 0.0},
    )
    return prob


def _force_recipe_less(prob: Problem) -> None:
    """Clear BOTH source slots on the LHS term — the post-reduction state of
    the crashing DES ``Sum(over=ALL)`` term (recipe-less, non-empty dims)."""
    term = prob._cstrs[0][1].expr.terms[0]
    assert tuple(term.dims), "fixture term must keep non-empty dims"
    # Clear both source slots in place — the post-reduction state the engine
    # leaves on a term it cannot block-evaluate (it does ``t.sum_block_meta =
    # None`` / ``var_source`` was never set on a Sum-reduced term).  These are
    # plain instance attributes on ``_Term``; assigning directly is the same
    # mutation the engine performs.
    term.var_source = None
    term.sum_block_meta = None


def _rows(prob: Problem) -> int:
    return sum(1 if over is None else int(over.height) for _c, _p, over in prob._cstrs)


def _install_layer2(prob: Problem, *, col_factor: float) -> None:
    """Install Layer-2 side vectors (production mode): uniform power-of-two
    ``col_factor`` so the col-factor multiply is exercised and stays exact in
    IEEE; distinct power-of-two per-row factors so the row-factor scale is
    non-trivial."""
    prob._layer2_col_factor = np.full(prob._next_col, col_factor, dtype=np.float64)
    prob._layer2_row_factor = 2.0 ** np.arange(_rows(prob), dtype=np.float64)


def _reference_matrix_range(prob: Problem) -> tuple[float, float]:
    """The PRE-block-COO side-vectors-on dim-bound readout, recomputed
    independently: align the ``_rid`` row index to the term's reduced
    ``.lazy`` on the shared axis dims, inner-join to attach ``_rid``, then
    reduce ``|coef| * |rf[base_row+_rid]| * |cf[col_id]|`` over finite
    non-zero entries.  ``base_row`` is 0 (single constraint family)."""
    cf = prob._layer2_col_factor
    rf = prob._layer2_row_factor
    ((_cname, proto, over),) = prob._cstrs
    term = proto.expr.terms[0]
    on = [d for d in term.dims if d in over.columns]
    ri = over.with_columns(_rid=pl.int_range(0, over.height, dtype=pl.Int64)).lazy()
    rl_a, tl_a = _align(ri, term.lazy, on)
    j = rl_a.join(tl_a, on=on, how="inner").select("_rid", "col_id", "coef").collect()
    rids = j["_rid"].to_numpy().astype(np.int64)
    cids = j["col_id"].to_numpy().astype(np.int64)
    vals = j["coef"].to_numpy().astype(np.float64)
    vals = vals * np.abs(rf[rids]) * np.abs(cf[cids])
    a = np.abs(vals)
    m = a[np.isfinite(a) & (a != 0)]
    return float(m.min()), float(m.max())


def test_from_term_raises_on_recipe_less_shape() -> None:
    """Sanity / crash precondition: ``from_term`` raises the documented
    ``TypeError`` on a term with both source slots cleared — so the guard is
    load-bearing, not vacuous.
    """
    prob = _build_problem()
    _force_recipe_less(prob)
    term = prob._cstrs[0][1].expr.terms[0]
    assert getattr(term, "var_source", None) is None
    assert getattr(term, "sum_block_meta", None) is None
    assert set(term.dims) == {"n", "d", "t"}
    try:
        CoefWalkRecipe.from_term(term)
    except TypeError as exc:
        assert "var_source must be a Var" in str(exc)
    else:
        raise AssertionError(
            "from_term must raise on a recipe-less term; the regression guard "
            "would be vacuous otherwise"
        )


def test_recipe_less_lhs_term_does_not_raise() -> None:
    """The routability guard keeps the recipe-less dim-bound term off the
    walk so ``from_term`` is never reached → no ``TypeError`` (the crash is
    gone) and the term still contributes a finite matrix range.
    """
    prob = _build_problem()
    _force_recipe_less(prob)
    _install_layer2(prob, col_factor=0.5)
    rep = _ranges_via_streaming(prob, ScalingConfig())
    assert rep is not None
    assert np.isfinite(rep.matrix[0]) and np.isfinite(rep.matrix[1])


def test_recipe_less_lhs_term_matrix_range_byte_identical() -> None:
    """The reported matrix range equals an independent reference reduction of
    the term's ``.lazy`` with the SAME scale — proving the non-routable term
    is still COUNTED into the range, byte-for-byte, not dropped.
    """
    prob = _build_problem()
    _force_recipe_less(prob)
    _install_layer2(prob, col_factor=0.5)
    expected = _reference_matrix_range(prob)
    rep = _ranges_via_streaming(prob, ScalingConfig())
    assert rep.matrix == expected, (rep.matrix, expected)
    assert expected[0] > 0.0 and np.isfinite(expected[1])


def test_recipe_less_lhs_term_counts_col_factor() -> None:
    """Halving the column factor halves the reported magnitude — confirms the
    backstop folds ``|_l2_cf[col_id]|`` (not a no-op collect that drops it).
    """
    p_a = _build_problem()
    _force_recipe_less(p_a)
    _install_layer2(p_a, col_factor=1.0)
    rep_a = _ranges_via_streaming(p_a, ScalingConfig())

    p_b = _build_problem()
    _force_recipe_less(p_b)
    _install_layer2(p_b, col_factor=0.5)
    rep_b = _ranges_via_streaming(p_b, ScalingConfig())

    assert rep_b.matrix[0] == rep_a.matrix[0] * 0.5, (rep_a.matrix, rep_b.matrix)
    assert rep_b.matrix[1] == rep_a.matrix[1] * 0.5, (rep_a.matrix, rep_b.matrix)


# ===CONTRACT_AND_SHAPE_B_TESTS===
# Anti-regression for the PRECISE routability gap the first fix missed: the
# matrix/LHS guard must use ``CoefWalkRecipe.is_buildable`` (the deep mirror
# of ``from_term``'s precondition), NOT a shallow ``var_source is not None OR
# sum_block_meta is not None`` (which lets a ``Sum`` term with
# ``meta.var_source is None`` through, then crashes inside ``from_term``).


class _DuckMeta:
    """Minimal duck-typed ``SumBlockMeta`` (non-frozen, so the test can vary
    ``var_source``) exposing exactly what ``from_term`` / ``is_buildable``
    read off a meta."""

    def __init__(self, var_source):
        self.var_source = var_source
        self.param_sources = ()
        self.coef_scalar = 1.0
        self.where_frames = None
        self.where_map_frames = None
        self.reduce_dims = ()
        self.keep = ()


class _DuckTerm:
    """Minimal duck-typed ``_Term`` exposing exactly the attributes
    ``from_term`` / ``is_buildable`` read."""

    def __init__(self, var_source, sum_block_meta):
        self.var_source = var_source
        self.sum_block_meta = sum_block_meta
        self.param_sources = ()
        self.coef_scalar = 1.0
        self.where_frames = None
        self.where_map_frames = None
        self.dims = ("d",)
        # ``term.lazy`` carries ``(*dims, col_id, coef)`` so the meta-branch
        # ``from_term`` (which captures it as ``reduced_lazy``) builds cleanly.
        self.lazy = pl.DataFrame({"d": ["a", "b"], "col_id": [0, 1], "coef": [1.0, 2.0]}).lazy()


def _make_real_var():
    """A genuine (frozen) ``Var`` so the non-collapsed combinations build."""
    p = Problem()
    return p.add_var("v", ("d",), pl.DataFrame({"d": ["a", "b"]}), lower=0.0, upper=1.0)


def _from_term_succeeds(term) -> bool:
    """Ground truth: does ``from_term`` succeed for a NON-param_only term?
    It raises ``TypeError`` from ``__init__`` when the selected Var is None."""
    try:
        _CWR.from_term(term)
        return True
    except TypeError:
        return False


@pytest.mark.parametrize("meta_kind", ["none", "meta_var_none", "meta_var_real"])
@pytest.mark.parametrize("term_has_var", [False, True])
def test_is_buildable_matches_from_term_exactly(meta_kind, term_has_var):
    """For EVERY (sum_block_meta in {None, meta var None, meta var real}) x
    (term.var_source in {None, real Var}) the predicate must EQUAL whether
    ``from_term`` succeeds — including the meta-present-var-None shape that
    crashed DES (which a shallow ``var or meta`` check gets WRONG)."""
    if meta_kind == "none":
        meta = None
    elif meta_kind == "meta_var_none":
        meta = _DuckMeta(var_source=None)
    else:
        meta = _DuckMeta(var_source=_make_real_var())
    term_var = _make_real_var() if term_has_var else None
    term = _DuckTerm(var_source=term_var, sum_block_meta=meta)

    expected = _from_term_succeeds(term)
    got = _CWR.is_buildable(term)
    assert got is expected, (
        f"is_buildable(meta={meta_kind}, term_var={term_has_var})={got} "
        f"but from_term-succeeds={expected}"
    )


def _force_meta_present_var_none(prob: Problem) -> None:
    """Manufacture SHAPE (b): ``term.sum_block_meta`` PRESENT but
    ``meta.var_source`` None (and the term's own ``var_source`` cleared) — the
    exact field combination that crashed DES Layer 3.  ``SumBlockMeta`` is a
    FROZEN dataclass, so rebuild it via ``dataclasses.replace(meta,
    var_source=None)`` and assign back to the (non-frozen) ``_Term`` — the
    post-build state of a Sum term whose Var seed is no longer reconstructable
    from the meta, where the reduced ``term.lazy`` is the only source of truth
    (precisely what the non-buildable collect path consumes)."""
    term = prob._cstrs[0][1].expr.terms[0]
    assert term.sum_block_meta is not None, "fixture term must be Sum-wrapped"
    assert tuple(term.dims), "fixture term must keep non-empty dims"
    term.var_source = None
    term.sum_block_meta = dataclasses.replace(term.sum_block_meta, var_source=None)


def test_meta_present_var_none_is_not_buildable():
    """Shape (b) precondition: meta present + ``meta.var_source`` None is NOT
    buildable (the SHALLOW guard WOULD have passed it — meta is not None — then
    crashed in ``from_term``); and ``from_term`` genuinely raises on it, so the
    guard is load-bearing, not vacuous."""
    prob = _build_problem()
    _force_meta_present_var_none(prob)
    term = prob._cstrs[0][1].expr.terms[0]
    assert term.sum_block_meta is not None
    assert term.sum_block_meta.var_source is None
    assert _CWR.is_buildable(term) is False
    try:
        _CWR.from_term(term)
    except TypeError as exc:
        assert "var_source must be a Var" in str(exc)
    else:
        raise AssertionError("from_term must raise on meta-present-var-None")


def test_meta_present_var_none_readout_byte_identical():
    """End-to-end SHAPE (b): a dim-bound ``Sum`` LHS term with
    ``sum_block_meta`` present but ``meta.var_source`` None routes to the
    non-buildable collect — the readout must NOT raise and the matrix range
    must be BYTE-IDENTICAL to an independent materialise-and-reduce of the
    term's ``.lazy`` with the SAME side-vector scale."""
    prob = _build_problem()
    _force_meta_present_var_none(prob)
    _install_layer2(prob, col_factor=0.5)
    expected = _reference_matrix_range(prob)
    rep = _ranges_via_streaming(prob, ScalingConfig())  # must NOT raise
    assert rep is not None
    assert rep.matrix == expected, (rep.matrix, expected)
    assert expected[0] > 0.0 and np.isfinite(expected[1])
