"""Collects of polar-high's join chains keep the written join order.

polars 2.x reorders runs of inner joins by estimated cardinality
(``QueryOptFlags.join_order``).  On the Sum relabel emit
``result ⋈ map(p, s → n) ⋈ row_index(n, d, t)`` it joined ``row_index`` to
``result`` on ``(d, t)`` alone first, a many-to-many product that cost
FlexTool several GB per call.  :mod:`polar_high._collect` turns the flag off
where polars has it and is a plain collect on polars 1.x.
"""

from __future__ import annotations

import inspect
import itertools
import os
import re

import numpy as np
import polars as pl

import polar_high._collect as pc
import polar_high.engine as eng
from polar_high.engine import Param, Problem, Sum, Where

_HAS_JOIN_ORDER = "join_order" in inspect.signature(pl.QueryOptFlags).parameters


def _join_keys(plan: pl.LazyFrame, **kw) -> list[set[str]]:
    """The key set of every join in ``plan.explain()``, outermost first."""
    text = plan.explain(**kw)
    keys = []
    for line in text.splitlines():
        if "LEFT PLAN ON:" in line:
            keys.append(set(re.findall(r'col\("([^"]+)"\)', line)))
    return keys


def _relabel_shaped_query() -> pl.LazyFrame:
    """``row_index(n, t) ⋈ (var(f, t) ⋈ map(f → n))``, 40 flows on 6 nodes."""
    flows = pl.DataFrame({"f": [f"f{i}" for i in range(40)]})
    time_ = pl.DataFrame({"t": [f"t{i:04d}" for i in range(200)]})
    var = flows.join(time_, how="cross").with_row_index("col_id").with_columns(coef=pl.lit(1.0))
    fmap = flows.with_columns(n=pl.format("n{}", pl.int_range(pl.len()) % 6))
    rows = pl.DataFrame({"n": [f"n{i}" for i in range(6)]}).join(time_, how="cross")
    rows = rows.with_row_index("_rid")
    inner = var.lazy().join(fmap.lazy(), on="f", how="inner").select("t", "n", "col_id", "coef")
    return rows.lazy().join(inner, on=["t", "n"], how="inner").select("_rid", "col_id", "coef")


def test_flag_matches_installed_polars():
    """``join_order`` is switched off where polars has it, and only there."""
    if _HAS_JOIN_ORDER:
        assert pc.NO_JOIN_REORDER is not None
        assert pc.NO_JOIN_REORDER.join_order is False
        # Every other optimisation keeps its default.
        default = pl.QueryOptFlags()
        for name in ("predicate_pushdown", "projection_pushdown", "comm_subplan_elim"):
            assert getattr(pc.NO_JOIN_REORDER, name) == getattr(default, name)
    else:
        assert pc.NO_JOIN_REORDER is None


def test_helpers_pass_flags_only_when_supported(monkeypatch):
    seen: list[dict] = []
    orig = pl.LazyFrame.collect

    def _spy(self, *args, **kw):
        seen.append(dict(kw))
        return orig(self, *args, **kw)

    monkeypatch.setattr(pl.LazyFrame, "collect", _spy)
    lf = pl.LazyFrame({"a": [1, 2]})
    assert pc.collect_in_join_order(lf).height == 2
    assert pc.collect_streaming_in_join_order(lf).height == 2
    for kw in seen:
        if _HAS_JOIN_ORDER:
            assert kw.get("optimizations") is pc.NO_JOIN_REORDER
        else:
            assert "optimizations" not in kw


def test_written_join_order_kept_in_plan():
    """The plan the helper runs joins the map first, never on ``t`` alone."""
    q = _relabel_shaped_query()
    kw = {} if pc.NO_JOIN_REORDER is None else {"optimizations": pc.NO_JOIN_REORDER}
    assert _join_keys(q, **kw) == [{"t", "n"}, {"f"}]
    expected = q.collect().sort("_rid", "col_id")
    assert pc.collect_in_join_order(q).sort("_rid", "col_id").equals(expected)


def _node_balance_builder() -> Problem:
    """``Sum(Where(v * P_unit, map p → n) * P_step, over=("p", "s"))`` with
    20 flows on each of 3 nodes, so ``(d, t)`` repeats on both sides."""
    p_idx = list(range(60))
    d_idx, t_idx = [10, 11], list(range(100, 148))
    prob = Problem(dense_axes=("d", "t"))
    rows = list(itertools.product(p_idx, ["s0"], d_idx, t_idx))
    var_index = pl.DataFrame({k: [r[i] for r in rows] for i, k in enumerate(("p", "s", "d", "t"))})
    v = prob.add_var("v", ("p", "s", "d", "t"), var_index, lower=0.0, upper=1e6)
    p_unit = Param(("p",), pl.DataFrame({"p": p_idx, "value": [1.0 + i % 5 for i in p_idx]}))
    dt = list(itertools.product(d_idx, t_idx))
    p_step = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [r[0] for r in dt],
                "t": [r[1] for r in dt],
                "value": np.linspace(0.5, 1.5, len(dt)),
            }
        ),
    )
    fmap = pl.DataFrame({"p": p_idx, "s": ["s0"] * 60, "n": [f"n{p % 3}" for p in p_idx]})
    lhs = Sum(Where(v * p_unit, fmap) * p_step, over=("p", "s"))
    over_frame = lhs.terms[0].frame.select(list(lhs.terms[0].dims)).unique()
    prob.add_cstr("nb", over=over_frame, sense="<=", lhs_terms={"lhs": lhs}, rhs_terms={"rhs": 0.0})
    return prob


def test_sum_relabel_emit_keeps_written_order(monkeypatch):
    """The relabel emit's own plan, captured at its collect, keeps the map
    join innermost and joins ``row_index`` on ``(n, d, t)``."""
    plans: list[pl.LazyFrame] = []
    in_relabel = {"on": False}
    orig_collect = eng.collect_in_join_order
    orig_relabel = eng._build_sum_block_coo_relabel

    def _collect(plan):
        if in_relabel["on"]:
            plans.append(plan)
        return orig_collect(plan)

    def _relabel(*args, **kw):
        in_relabel["on"] = True
        try:
            return orig_relabel(*args, **kw)
        finally:
            in_relabel["on"] = False

    monkeypatch.setattr(eng, "collect_in_join_order", _collect)
    monkeypatch.setattr(eng, "_build_sum_block_coo_relabel", _relabel)
    monkeypatch.setitem(os.environ, "POLAR_HIGH_BLOCK_COO_MIN_DENSE", "1")
    _node_balance_builder()._build_canonical_matrix()
    assert plans, "the relabel emit did not run"
    kw = {} if pc.NO_JOIN_REORDER is None else {"optimizations": pc.NO_JOIN_REORDER}
    keys = _join_keys(plans[-1], **kw)
    assert keys and keys[0] == {"n", "d", "t"}, keys
    assert all(not k <= {"d", "t"} for k in keys), keys
