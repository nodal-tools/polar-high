"""Phase C-2: capture an INERT ``SumBlockMeta`` reconstruction recipe.

When :func:`polar_high.engine.Sum` reduces a block-eligible term it
clears ``var_source`` and survivor-filters ``param_sources`` on the
returned term, discarding the information a future block-COO classifier
(Phase C-3) needs to rebuild the pre-Sum ``row_index → Var → P1 → P2 …``
chain and reduce it in-block.  Phase C-2 snapshots that pre-Sum state
into the new :class:`SumBlockMeta` on ``_Term.sum_block_meta``.

This change is INERT: nothing reads ``sum_block_meta`` yet.  These
tests pin (a) the capture content for a nodeBalance-shaped Sum, (b) the
None default for a non-block-eligible Sum, and (c) byte-identity of the
canonical matrix with vs without the captured recipe.
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
import pytest

from polar_high.engine import Param, Problem, Sum, SumBlockMeta, Where


def _node_balance_terms():
    """Build a nodeBalance-shaped LHS and return ``(expr, v, P_unit,
    P_step)``.

    Shape: ``Sum(Where(v(p,s,d,t) * P_unit(p), map_to_n) * P_step(d,t),
    over=("p","s"))``.  The map frame ``map_to_n`` introduces a new dim
    ``n`` from ``(p, s)``; ``over=("p","s")`` collapses p and s, so the
    surviving open dims are the post-map dims ``(d, t, n)``.
    """
    p_idx = [0, 1]
    s_idx = ["s0", "s1"]
    d_idx = [10, 11]
    t_idx = [100, 101, 102]

    prob = Problem()
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

    # P_unit keyed on p only — its dim 'p' is summed OUT by ``over``, so
    # the survivor filter drops it from the returned term yet block-COO
    # still needs it: SumBlockMeta must carry it FULL.
    P_unit = Param(
        ("p",),
        pl.DataFrame({"p": p_idx, "value": [2.0, 3.0]}),
        name="P_unit",
    )
    # P_step keyed on (d, t) — survives ``over`` (neither dim summed).
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
    # Map frame: (p, s) -> n.  Two nodes.
    map_rows = list(itertools.product(p_idx, s_idx))
    map_to_n = pl.DataFrame(
        {
            "p": [r[0] for r in map_rows],
            "s": [r[1] for r in map_rows],
            "n": [f"n{(r[0] + (0 if r[1] == 's0' else 1)) % 2}" for r in map_rows],
        }
    )

    expr = Sum(Where(v * P_unit, map_to_n) * P_step, over=("p", "s"))
    return expr, v, P_unit, P_step


def test_sum_block_meta_captures_node_balance_recipe():
    expr, v, P_unit, P_step = _node_balance_terms()
    assert len(expr.terms) == 1
    term = expr.terms[0]

    meta = term.sum_block_meta
    assert isinstance(meta, SumBlockMeta), "block-eligible Sum must capture a recipe"

    # var_source is the originating Var (cleared on the returned term).
    assert term.var_source is None, "Sum still clears var_source on the term"
    assert meta.var_source is v

    # reduce_dims == the Sum's ``over``; keep == the returned term's dims.
    assert meta.reduce_dims == ("p", "s")
    assert meta.keep == term.dims

    # FULL (un-filtered) param_sources: BOTH P_unit and P_step present,
    # even though P_unit's only dim ('p') is summed out by ``over``.
    captured_params = [p for (p, _dir) in meta.param_sources]
    assert P_unit in captured_params, "P_unit (dim summed out) must still be captured"
    assert P_step in captured_params
    assert len(meta.param_sources) == 2

    # The returned term's own (survivor-filtered) param_sources must have
    # dropped P_unit — confirming the recipe carries strictly MORE.
    survivor_params = [p for (p, _dir) in (term.param_sources or [])]
    assert P_unit not in survivor_params
    assert P_step in survivor_params


def test_plain_sum_no_var_source_gets_none():
    """A Sum of a bare expression with no Var (and no param chain) must
    leave ``sum_block_meta`` None — not block-eligible."""
    prob = Problem()
    x_idx = list(range(6))
    v = prob.add_var("v", ("x",), pl.DataFrame({"x": x_idx}), lower=0.0)
    # Sum of a plain Var.to_expr() -> term has var_source set but EMPTY
    # param_sources (no Param multiply), so it is NOT block-eligible.
    expr = Sum(v, over=("x",))
    assert expr.terms[0].sum_block_meta is None

    # And a Sum applied to an already-reduced term (nested) -> None too.
    P = Param(
        ("x",),
        pl.DataFrame({"x": x_idx, "value": np.linspace(0.1, 0.6, len(x_idx))}),
        name="P",
    )
    inner = Sum(v * P, over=())  # over=() -> not block-eligible (empty over)
    assert inner.terms[0].sum_block_meta is None


def test_nested_sum_does_not_carry_stale_recipe():
    """A re-Sum on an already-reduced (block-eligible) term must NOT
    carry a stale recipe — the new term's ``sum_block_meta`` is None."""
    prob = Problem()
    rows = list(itertools.product(range(3), range(4)))
    over_xt = pl.DataFrame({"x": [r[0] for r in rows], "t": [r[1] for r in rows]})
    v = prob.add_var("v", ("x", "t"), over_xt, lower=0.0, upper=1e6)
    P = Param(
        ("x", "t"),
        pl.DataFrame(
            {
                "x": [r[0] for r in rows],
                "t": [r[1] for r in rows],
                "value": np.linspace(0.1, 1.0, len(rows)),
            }
        ),
        name="P",
    )
    first = Sum(v * P, over=("t",))
    assert first.terms[0].sum_block_meta is not None  # block-eligible reduction
    # Re-Sum the already-reduced term (var_source is None now anyway, but
    # the guard also blocks stale recipes if a future op preserved it).
    second = Sum(first, over=("x",))
    assert second.terms[0].sum_block_meta is None


# ---------------------------------------------------------------------------
# D1: the recipe is now FORWARDED through post-Sum Expr-algebra ops.


def _sum_relabel():
    """A block-eligible ``Sum(v * P_unit * P_step, over=('p','s'))`` whose
    surviving open dims are ``(d, t)`` — the plain relabel shape (no map
    effect).  Returns ``(sum_expr, v, P_unit, P_step)``."""
    ps, ss, ds, ts = [0, 1], ["s0", "s1"], [10, 11], [100, 101, 102]
    rows = list(itertools.product(ps, ss, ds, ts))
    prob = Problem()
    var_index = pl.DataFrame(
        {
            "p": [r[0] for r in rows],
            "s": [r[1] for r in rows],
            "d": [r[2] for r in rows],
            "t": [r[3] for r in rows],
        }
    )
    v = prob.add_var("v", ("p", "s", "d", "t"), var_index, lower=0.0, upper=1e6)
    P_unit = Param(("p",), pl.DataFrame({"p": ps, "value": [2.0, 3.0]}), name="P_unit")
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
    sum_expr = Sum(v * P_unit * P_step, over=("p", "s"))
    assert sum_expr.terms[0].sum_block_meta is not None
    return sum_expr, v, P_unit, P_step


def test_neg_sum_carries_meta():
    """``-Sum(v*P, over=...)`` must carry the recipe with ``coef_scalar``
    negated and ``reduce_dims`` / ``keep`` unchanged."""
    sum_expr, v, P_unit, P_step = _sum_relabel()
    base = sum_expr.terms[0].sum_block_meta

    neg = (-sum_expr).terms[0]
    meta = neg.sum_block_meta
    assert isinstance(meta, SumBlockMeta), "-Sum must FORWARD the recipe"
    assert meta.coef_scalar == -base.coef_scalar
    assert meta.reduce_dims == base.reduce_dims == ("p", "s")
    assert meta.keep == base.keep
    assert meta.var_source is v
    # Param chain unchanged by a bare negation.
    assert [p for (p, _d) in meta.param_sources] == [p for (p, _d) in base.param_sources]


def test_scalar_mul_sum_carries_meta():
    """``Sum(...) * 3.0`` must carry the recipe with ``coef_scalar`` ×3."""
    sum_expr, _v, _P_unit, _P_step = _sum_relabel()
    base = sum_expr.terms[0].sum_block_meta

    scaled = (sum_expr * 3.0).terms[0]
    meta = scaled.sum_block_meta
    assert isinstance(meta, SumBlockMeta)
    assert meta.coef_scalar == base.coef_scalar * 3.0
    assert meta.reduce_dims == base.reduce_dims
    assert meta.keep == base.keep

    # And the same through scalar division.
    halved = (sum_expr / 2.0).terms[0]
    assert halved.sum_block_meta is not None
    assert halved.sum_block_meta.coef_scalar == base.coef_scalar / 2.0


def test_sub_sum_carries_meta_negated():
    """``other_expr - Sum(...)`` negates the subtracted side's recipe
    ``coef_scalar``; the self side is untouched.  This is the headline
    ``-Sum(...)`` production constraint shape."""
    sum_expr, v, _P_unit, _P_step = _sum_relabel()
    base = sum_expr.terms[0].sum_block_meta

    # A bare-Var Expr on the self side, minus the Sum term (the subtracted,
    # negated side carries the recipe).
    other = v.to_expr()
    diff = other - sum_expr
    metas = [t.sum_block_meta for t in diff.terms if t.sum_block_meta is not None]
    assert len(metas) == 1, "only the subtracted Sum term carries a recipe"
    assert metas[0].coef_scalar == -base.coef_scalar
    assert metas[0].keep == base.keep
    assert metas[0].reduce_dims == base.reduce_dims
    # The self-side bare-Var term is untouched (no recipe, positive coef).
    self_terms = [t for t in diff.terms if t.sum_block_meta is None]
    assert len(self_terms) == 1
    assert self_terms[0].coef_scalar == 1.0


def test_param_mul_after_sum_existing_dim_carries():
    """Post-Sum ``* P`` where ``P.dims ⊆ keep`` (and disjoint from
    ``reduce_dims``) carries the recipe with the Param appended to
    ``param_sources`` and its constant folded into ``coef_scalar``."""
    sum_expr, _v, _P_unit, _P_step = _sum_relabel()
    base = sum_expr.terms[0].sum_block_meta
    assert base.keep == ("d", "t")

    dt = list(itertools.product([10, 11], [100, 101, 102]))
    # P_extra keyed on (d, t) — subset of keep, disjoint from reduce_dims.
    P_extra = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [r[0] for r in dt],
                "t": [r[1] for r in dt],
                "value": np.linspace(1.0, 2.0, len(dt)),
            }
        ),
        name="P_extra",
    )

    out = (sum_expr * P_extra).terms[0]
    meta = out.sum_block_meta
    assert isinstance(meta, SumBlockMeta), "SAFE post-Sum Param mul must carry"
    captured = [p for (p, _d) in meta.param_sources]
    assert P_extra in captured, "the multiplied Param must be appended"
    assert len(meta.param_sources) == len(base.param_sources) + 1
    # Direction +1 for multiply.
    assert meta.param_sources[-1] == (P_extra, 1)
    assert meta.keep == base.keep and meta.reduce_dims == base.reduce_dims


def test_param_div_after_sum_existing_dim_carries_negated_dir():
    """Post-Sum ``/ P`` (SAFE) appends the Param with direction -1."""
    sum_expr, _v, _P_unit, _P_step = _sum_relabel()
    base = sum_expr.terms[0].sum_block_meta
    dt = list(itertools.product([10, 11], [100, 101, 102]))
    P_extra = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [r[0] for r in dt],
                "t": [r[1] for r in dt],
                "value": np.linspace(1.0, 2.0, len(dt)),
            }
        ),
        name="P_extra",
    )

    out = (sum_expr / P_extra).terms[0]
    meta = out.sum_block_meta
    assert isinstance(meta, SumBlockMeta)
    assert meta.param_sources[-1] == (P_extra, -1)
    assert len(meta.param_sources) == len(base.param_sources) + 1


def test_param_mul_after_sum_new_dim_declines():
    """Post-Sum ``* P`` where ``P`` introduces a dim NOT in ``keep`` must
    DECLINE: ``sum_block_meta is None`` AND a loud marker is emitted."""
    sum_expr, _v, _P_unit, _P_step = _sum_relabel()
    # P_new keyed on a brand-new dim 'z' (absent from keep=(d,t)).
    P_new = Param(("z",), pl.DataFrame({"z": [0, 1], "value": [4.0, 5.0]}), name="P_new")

    with pytest.warns(UserWarning, match=r"sum-block-meta DECLINE"):
        out = (sum_expr * P_new).terms[0]
    assert out.sum_block_meta is None, "introducing a new dim must DECLINE"


def test_param_mul_after_sum_reintroduces_summed_dim_declines():
    """Post-Sum ``* P`` whose dim re-introduces a SUMMED dim
    (``∩ reduce_dims``) must DECLINE."""
    sum_expr, _v, _P_unit, _P_step = _sum_relabel()
    # 'p' is in reduce_dims=(p,s) — re-introducing it must decline.
    P_p = Param(("p",), pl.DataFrame({"p": [0, 1], "value": [7.0, 9.0]}), name="P_p")
    with pytest.warns(UserWarning, match=r"sum-block-meta DECLINE"):
        out = (sum_expr * P_p).terms[0]
    assert out.sum_block_meta is None


def test_where_pure_after_sum_carries_meta():
    """A pure-filter ``Where`` after a Sum carries the recipe with the
    filter recorded into the recipe's ``where_frames`` (no dim change)."""
    sum_expr, _v, _P_unit, _P_step = _sum_relabel()
    base = sum_expr.terms[0].sum_block_meta
    base_wf = len(base.where_frames or ())

    # Pure filter on a kept dim (d) — shared, no extras.
    filt = pl.DataFrame({"d": [10]})
    out = Where(sum_expr, filt).terms[0]
    meta = out.sum_block_meta
    assert isinstance(meta, SumBlockMeta), "pure-filter Where must carry"
    assert meta.keep == base.keep and meta.reduce_dims == base.reduce_dims
    assert len(meta.where_frames or ()) == base_wf + 1


def test_where_map_after_sum_carries_meta():
    """A map-effect ``Where`` after a Sum carries the recipe with the
    map frame recorded and ``keep`` grown by the introduced dim."""
    sum_expr, _v, _P_unit, _P_step = _sum_relabel()
    base = sum_expr.terms[0].sum_block_meta
    assert base.keep == ("d", "t")
    base_wmf = len(base.where_map_frames or ())

    # Map frame (d) -> e introduces a new open dim 'e'.
    map_de = pl.DataFrame({"d": [10, 11], "e": ["e0", "e1"]})
    out = Where(sum_expr, map_de).terms[0]
    meta = out.sum_block_meta
    assert isinstance(meta, SumBlockMeta), "map-effect Where must carry"
    assert "e" in meta.keep, "keep must grow by the introduced dim"
    assert set(meta.keep) == {"d", "t", "e"}
    assert len(meta.where_map_frames or ()) == base_wmf + 1
    # Frame entry shape matches the (LazyFrame, frozenset) classifier
    # contract.
    last_frame, last_extras = meta.where_map_frames[-1]
    assert isinstance(last_extras, frozenset)
    assert last_extras == frozenset({"e"})
    assert meta.reduce_dims == base.reduce_dims


def test_double_sum_collapse_all_forwards_meta():
    """``Sum(Sum(v*P, over=(...all...)), over=None)`` — the outer collapse-
    all over an already-collapsed term forwards the recipe verbatim."""
    ps, ds, ts = [0, 1], [10, 11], [100, 101]
    rows = list(itertools.product(ps, ds, ts))
    prob = Problem()
    var_index = pl.DataFrame(
        {"p": [r[0] for r in rows], "d": [r[1] for r in rows], "t": [r[2] for r in rows]}
    )
    v = prob.add_var("v", ("p", "d", "t"), var_index, lower=0.0, upper=1e6)
    P = Param(
        ("p", "d", "t"),
        pl.DataFrame(
            {
                "p": [r[0] for r in rows],
                "d": [r[1] for r in rows],
                "t": [r[2] for r in rows],
                "value": np.linspace(0.1, 1.0, len(rows)),
            }
        ),
        name="P",
    )
    # Inner Sum collapses EVERY open dim -> scalar term (dims == ()).
    inner = Sum(v * P, over=("p", "d", "t"))
    assert inner.terms[0].dims == ()
    assert inner.terms[0].sum_block_meta is not None
    # Outer collapse-all over an already-collapsed term.
    outer = Sum(inner, over=None)
    out_term = outer.terms[0]
    assert out_term.dims == ()
    assert out_term.sum_block_meta is not None, "collapse-all must FORWARD"
    assert out_term.sum_block_meta.keep == ()


def _matrix_arrays(m):
    val = np.asarray(m.val, dtype=np.float64)
    row_idx = np.asarray(m.row_idx, dtype=np.int64)
    col_ptr = np.asarray(m.col_ptr, dtype=np.int64)
    cols = np.repeat(np.arange(m.n_cols, dtype=np.int64), np.diff(col_ptr).astype(np.int64))
    order = np.lexsort((row_idx, cols))
    return (
        list(val[order]),
        list(np.asarray(m.row_lb, dtype=np.float64)),
        list(np.asarray(m.row_ub, dtype=np.float64)),
    )


def _build_node_balance_problem() -> Problem:
    """A full nodeBalance-shaped Problem whose LHS is the Sum-reduced
    term that now carries a SumBlockMeta recipe."""
    p_idx = [0, 1]
    s_idx = ["s0", "s1"]
    d_idx = [10, 11]
    t_idx = [100, 101, 102]
    prob = Problem()
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
    P_unit = Param(("p",), pl.DataFrame({"p": p_idx, "value": [2.0, 3.0]}), name="P_unit")
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
    over_cstr = lhs.terms[0].dims  # (d, t, n)
    over_frame = lhs.terms[0].frame.select(list(over_cstr)).unique()
    prob.add_cstr(
        "nb",
        over=over_frame,
        sense="<=",
        lhs_terms={"lhs": lhs},
        rhs_terms={"rhs": 0.0},
    )
    return prob


def test_recipe_is_byte_identical_inert():
    """The captured recipe must NOT affect any output: the canonical
    matrix with the recipe present is byte-identical to one where every
    term's ``sum_block_meta`` is stripped to None before canonicalising.
    """
    prob_with = _build_node_balance_problem()
    snap_with = _matrix_arrays(prob_with._build_canonical_matrix())

    prob_without = _build_node_balance_problem()
    # Strip the recipe everywhere it could have been captured.
    for _name, proto, _over in prob_without._cstrs:
        for t in proto.expr.terms:
            t.sum_block_meta = None
    # Confirm we actually had a recipe to strip in the "with" run.
    had_recipe = any(
        t.sum_block_meta is not None for _n, proto, _o in prob_with._cstrs for t in proto.expr.terms
    )
    assert had_recipe, "test fixture must produce a block-eligible Sum recipe"
    snap_without = _matrix_arrays(prob_without._build_canonical_matrix())

    assert snap_with == snap_without
