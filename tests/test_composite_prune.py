"""Composite-Param pruning: a ``Var × P`` / ``Expr × P`` / ``Expr / P`` term
whose ``P`` is an *expanding* ``Param ⊗ Param`` composite (a cross join, or a
join wider than either operand) joins a rebuild of ``P`` with every leaf
semi-joined to the term's keys, instead of ``P``'s whole plan.

The rebuild keeps the operation tree's association and left/right roles, so the
pruned term must equal the term built with ``POLAR_HIGH_DISABLE_COMPOSITE_PRUNE=1``
bit for bit — support, schema and coefficient bits — including a Sum group of
three coefficients on one (row, col) whose float sum depends on their order.
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
import pytest

import polar_high as fp
from polar_high.engine import (
    _apply_where_frames,
    _apply_where_map_frames,
    _composite_prune_applies,
    _prune_composite_lazy,
)

HATCH = "POLAR_HIGH_DISABLE_COMPOSITE_PRUNE"
AXES = {"a": 7, "b": 5, "c": 6, "d": 4}


def _vocab(axis: str) -> list[str]:
    return [f"{axis}{i}" for i in range(AXES[axis])]


def _cast(df: pl.DataFrame, mode: str, rng: np.random.Generator) -> pl.DataFrame:
    """Key columns as Utf8 (``utf8``), full-vocabulary Enum (``enum``) or a
    random choice per frame (``mixed``)."""
    enum = mode == "enum" or (mode == "mixed" and rng.random() < 0.5)
    if not enum:
        return df
    return df.with_columns(*(pl.col(c).cast(pl.Enum(_vocab(c))) for c in df.columns if c in AXES))


def _rows(dims: tuple[str, ...], rng: np.random.Generator, density: float) -> pl.DataFrame:
    """A random sparse, shuffled key set over ``dims``."""
    combos = list(itertools.product(*(_vocab(d) for d in dims)))
    keep = [c for c in combos if rng.random() < density] or [combos[0]]
    order = rng.permutation(len(keep))
    keep = [keep[i] for i in order]
    return pl.DataFrame(
        {d: [r[i] for r in keep] for i, d in enumerate(dims)}, schema={d: pl.Utf8 for d in dims}
    )


def _leaf(dims: tuple[str, ...], rng: np.random.Generator, mode: str) -> fp.Param:
    f = _rows(dims, rng, 0.7)
    # values spread over many binades so the products / quotients are not exact
    vals = rng.uniform(0.5, 2.0, f.height) * 10.0 ** rng.integers(-6, 7, f.height)
    return fp.Param(dims, _cast(f.with_columns(value=pl.Series(vals)), mode, rng))


def _random_dims(rng: np.random.Generator) -> tuple[str, ...]:
    k = int(rng.integers(1, 3))
    return tuple(sorted(rng.choice(list(AXES), size=k, replace=False).tolist()))


def _random_tree(rng: np.random.Generator, mode: str, depth: int) -> fp.Param:
    """A random composite over mul / div / scalar multiply / scalar divide /
    negation.  Leaves: frame Params, ``Param + Param`` results, and Params
    built on a lazy plan that divides (as a caller re-wrapping a composite's
    ``.lazy`` would) — the pruning semi-joins on top of such a plan."""
    if depth == 0:
        r = rng.random()
        d = _random_dims(rng)
        if r < 0.15:
            return _leaf(d, rng, mode) + _leaf(d, rng, mode)
        if r < 0.25:
            inner = _leaf(d, rng, mode) / _leaf(d[:1], rng, mode)
            return fp.Param(inner.dims, inner.lazy.with_columns(value=pl.col("value") / 3.7))
        return _leaf(d, rng, mode)
    left = _random_tree(rng, mode, depth - 1)
    right = _random_tree(rng, mode, int(rng.integers(0, depth)))
    out = left * right if rng.random() < 0.75 else left / right
    r = rng.random()
    if r < 0.15:
        out = out * float(rng.uniform(0.1, 3.0))
    elif r < 0.22:
        out = out / float(rng.uniform(0.1, 3.0))
    elif r < 0.29:
        out = -out
    return out


def _var(
    prob: fp.Problem, name: str, dims: tuple[str, ...], rng: np.random.Generator, mode: str
) -> fp.Var:
    return prob.add_var(name, dims, _cast(_rows(dims, rng, 0.5), mode, rng))


def _seed(kind: str, prob: fp.Problem, rng: np.random.Generator, mode: str):
    """The left operand of the product: a Var, an Expr after a map-effect
    Where, an Expr after a deferred pure-filter Where, or a Sum result (no
    Var source left) after a deferred pure-filter Where."""
    if kind == "var":
        return _var(prob, "x", ("a", "b"), rng, mode)
    if kind == "map_where":
        x = _var(prob, "x", ("a",), rng, mode)
        mp = _cast(_rows(("a", "c"), rng, 0.4), mode, rng)
        return fp.Where(x, mp)
    if kind == "post_sum":
        x = _var(prob, "x", ("a", "c", "d"), rng, mode)
        flt = _cast(_rows(("a",), rng, 0.6), mode, rng)
        return fp.Where(fp.Sum(x, over=("d",)), flt)
    x = _var(prob, "x", ("a", "c"), rng, mode)
    flt = _cast(_rows(("a",), rng, 0.4), mode, rng)
    return fp.Where(x, flt)


def _collect_sorted(expr: fp.Expr) -> list[pl.DataFrame]:
    """Each term's rows as every ``t.lazy`` consumer sees them: deferred
    pure-filter Wheres, then deferred map-effect Wheres, applied."""
    out = []
    for t in expr.terms:
        lf = _apply_where_frames(t.lazy, t.dims, t.where_frames)
        lf, _ = _apply_where_map_frames(lf, t.dims, t.where_map_frames)
        df = lf.collect()
        out.append(df.sort([c for c in df.columns if c != "coef"] + ["coef"]))
    return out


def _assert_bitwise_equal(got: list[pl.DataFrame], ref: list[pl.DataFrame]) -> None:
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert g.schema == r.schema
        assert g.height == r.height
        for c in g.columns:
            if c == "coef":
                assert np.array_equal(
                    g[c].to_numpy().view(np.int64), r[c].to_numpy().view(np.int64)
                )
            else:
                assert g[c].equals(r[c])


def _build_both(make_term, monkeypatch):
    """Build the term with the pruning on, then with the escape hatch set."""
    monkeypatch.delenv(HATCH, raising=False)
    pruned = make_term()
    monkeypatch.setenv(HATCH, "1")
    ref = make_term()
    monkeypatch.delenv(HATCH, raising=False)
    return pruned, ref


@pytest.mark.parametrize("mode", ["utf8", "enum", "mixed"])
@pytest.mark.parametrize("seed_kind", ["var", "map_where", "filter_where", "post_sum"])
@pytest.mark.parametrize("op", ["mul", "div"])
def test_random_trees_bit_identical_to_escape_hatch(mode, seed_kind, op, monkeypatch):
    rng = np.random.default_rng(sum(map(ord, f"{mode}/{seed_kind}/{op}")))
    n_pruned = 0
    for trial in range(30):
        trng = np.random.default_rng(int(rng.integers(0, 2**31)) + trial)
        prob = fp.Problem()
        lhs = _seed(seed_kind, prob, trng, mode)
        if op == "div" and isinstance(lhs, fp.Var):
            lhs = lhs.to_expr()
        comp = _random_tree(trng, mode, depth=int(trng.integers(1, 4)))
        lhs_cols = set(lhs.dims) if isinstance(lhs, fp.Var) else set(lhs.terms[0].dims)
        n_pruned += _composite_prune_applies(comp, [d for d in comp.dims if d in lhs_cols])
        if comp._op is not None and comp._op[5]:
            assert not _composite_prune_applies(
                comp, list(comp.dims)
            )  # a tree that divides keeps its plan

        def make(lhs=lhs, comp=comp):
            return lhs * comp if op == "mul" else lhs / comp

        pruned, ref = _build_both(make, monkeypatch)
        _assert_bitwise_equal(_collect_sorted(pruned), _collect_sorted(ref))
        summed = _build_both(lambda: fp.Sum(make(), over=("a",)), monkeypatch)
        _assert_bitwise_equal(_collect_sorted(summed[0]), _collect_sorted(summed[1]))
    if op == "mul":
        assert n_pruned >= 5  # the random trees do exercise the pruned rebuild


def test_bound_pruned_composite_height():
    """2000 p × 2000 t cross-join composite consumed by a term over 2 p: the
    pruned rebuild has 2 × 2000 rows, not 4 M."""
    n = 2000
    rng = np.random.default_rng(3)
    p_leaf = fp.Param(
        ("p",), pl.DataFrame({"p": [f"p{i}" for i in range(n)], "value": rng.random(n)})
    )
    t_leaf = fp.Param(
        ("t",), pl.DataFrame({"t": [f"t{i}" for i in range(n)], "value": rng.random(n)})
    )
    comp = p_leaf * t_leaf
    prob = fp.Problem()
    x = prob.add_var("x", ("p",), pl.DataFrame({"p": ["p5", "p1234"]}))
    shared = ["p"]
    assert _composite_prune_applies(comp, shared)
    pruned = _prune_composite_lazy(comp._op, [x.frame.lazy()], shared).collect()
    assert pruned.height == 2 * n
    assert pruned.schema == comp.lazy.collect_schema()
    term = (x * comp).terms[0].lazy.collect()
    assert term.height == 2 * n


def test_non_expanding_product_keeps_plan(monkeypatch):
    """(p, t) × (p) is a pure filter: the term's plan is the plan built with
    the escape hatch set."""
    pt = fp.Param(
        ("p", "t"),
        pl.DataFrame({"p": ["p0", "p0", "p1"], "t": ["t0", "t1", "t0"], "value": [1.0, 2.0, 3.0]}),
    )
    pp = fp.Param(("p",), pl.DataFrame({"p": ["p0", "p1"], "value": [4.0, 5.0]}))
    comp = -(pt * pp) / 2.0
    assert comp._op is not None and not comp._op[4]
    prob = fp.Problem()
    x = prob.add_var("x", ("p", "t"), pl.DataFrame({"p": ["p0", "p1"], "t": ["t0", "t0"]}))
    pruned, ref = _build_both(lambda: fp.Where(x, pl.DataFrame({"p": ["p0"]})) * comp, monkeypatch)
    assert pruned.terms[0].lazy.explain() == ref.terms[0].lazy.explain()
    pruned, ref = _build_both(lambda: x * comp, monkeypatch)
    assert pruned.terms[0].lazy.explain() == ref.terms[0].lazy.explain()


def _three_entry_problem(rng_seed: int) -> fp.Problem:
    """``Σ_t x[p] · (a[p] · b[t])`` with b = (1e16, 1, −1e16) in shuffled
    order: each row's coefficient on x[p] is a three-entry group whose float
    sum (0 or 1) depends on the order the entries are summed in."""
    rng = np.random.default_rng(rng_seed)
    ps = [f"p{i}" for i in range(6)]
    ts = ["t0", "t1", "t2"]
    bvals = {"t0": 1e16, "t1": 1.0, "t2": -1e16}
    t_order = [ts[i] for i in rng.permutation(3)]
    a = fp.Param(
        ("p",),
        pl.DataFrame({"p": ps, "value": [1.0] * len(ps)}).sample(
            fraction=1.0, shuffle=True, seed=rng_seed
        ),
    )
    b = fp.Param(("t",), pl.DataFrame({"t": t_order, "value": [bvals[t] for t in t_order]}))
    prob = fp.Problem()
    x = prob.add_var("x", ("p",), pl.DataFrame({"p": ps[:4]}), lower=0.0, upper=1.0)
    y = prob.add_var("y", ("p",), pl.DataFrame({"p": ps[:4]}), lower=0.0, upper=1.0)
    prob.add_cstr(
        "c",
        over=pl.DataFrame({"p": ps[:4]}),
        sense="<=",
        lhs_terms={"x": fp.Sum(x * (a * b), over=("t",)), "y": y},
        rhs_terms={"one": 1.0},
    )
    prob.set_objective(fp.Sum(x) + fp.Sum(y), sense="max")
    return prob


@pytest.mark.parametrize("block_coo_off", [False, True])
@pytest.mark.parametrize("rng_seed", [0, 1, 2, 3])
def test_three_entry_group_bit_identical(rng_seed, block_coo_off, monkeypatch):
    if block_coo_off:
        monkeypatch.setenv("POLAR_HIGH_DISABLE_BLOCK_COO", "1")
    mats = []
    for hatch in (False, True):
        if hatch:
            monkeypatch.setenv(HATCH, "1")
        else:
            monkeypatch.delenv(HATCH, raising=False)
        cm = _three_entry_problem(rng_seed).canonicalise()
        mats.append(cm)
    got, ref = mats
    for f in ("col_ptr", "row_idx", "row_lb", "row_ub", "col_obj"):
        assert np.array_equal(getattr(got, f), getattr(ref, f)), f
    assert np.array_equal(np.asarray(got.val).view(np.int64), np.asarray(ref.val).view(np.int64))


def test_three_entry_group_term_bits(monkeypatch):
    """The reduced Sum term itself (the walk's backstop plan) is bit-equal."""
    for rng_seed in range(6):
        rng = np.random.default_rng(rng_seed)
        ts = ["t0", "t1", "t2"]
        bvals = {"t0": 1e16, "t1": 1.0, "t2": -1e16}
        order = [ts[i] for i in rng.permutation(3)]
        a = fp.Param(("p",), pl.DataFrame({"p": ["p0", "p1", "p2"], "value": [1.0, 3.0, 0.5]}))
        b = fp.Param(("t",), pl.DataFrame({"t": order, "value": [bvals[t] for t in order]}))
        prob = fp.Problem()
        x = prob.add_var("x", ("p",), pl.DataFrame({"p": ["p2", "p0"]}))
        comp = a * b
        pruned, ref = _build_both(lambda x=x, comp=comp: fp.Sum(x * comp, over=("t",)), monkeypatch)
        _assert_bitwise_equal(_collect_sorted(pruned), _collect_sorted(ref))
