"""``TopKAbsReducer`` and ``locate_extremes``.

The oracle in each test is a plain numpy sort of the full array written in
the test, independent of the reducer's batching.
"""

from __future__ import annotations

import math
import os

import numpy as np
import polars as pl
import pytest

from polar_high.autoscale import TopKAbsReducer, locate_extremes
from polar_high.autoscale._coef_walk import CountWithinReducer
from polar_high.engine import Param, Problem


def _feed(red, rid, cid, coef, cuts) -> None:
    edges = [0, *cuts, coef.size]
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        red.update(rid[a:b], cid[a:b], coef[a:b])


@pytest.mark.parametrize("side", ["min", "max"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_topk_matches_numpy_oracle_for_any_batching(side: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 5000
    coef = rng.choice([-1.0, 1.0], n) * 10.0 ** rng.uniform(-12, 12, n)
    coef[rng.choice(n, 50, replace=False)] = 0.0
    coef[rng.choice(n, 20, replace=False)] = np.inf
    rid = rng.integers(0, 40, n)
    cid = rng.integers(0, 30, n)
    rf = 2.0 ** rng.integers(-4, 5, 40)
    cf = 2.0 ** rng.integers(-4, 5, 30)
    integer_cols = rng.random(30) < 0.3
    floor, infinite, k, factor = 1e-6, 1e9, 7, 100.0
    # Oracle on the whole array.
    post = np.abs(coef * np.abs(rf[rid]) * np.abs(cf[cid]))
    model = np.abs(coef)
    keep = np.isfinite(post) & (post != 0) & (post < infinite) & ~integer_cols[cid]
    if side == "min":
        keep &= model >= floor
    p = post[keep]
    best_first = np.sort(p) if side == "min" else np.sort(p)[::-1]
    want = best_first[:k]
    best = want[0]
    near = (p <= best * factor) if side == "min" else (p >= best / factor)
    for cuts in ([], [1], [17, 900, 901, 4000], list(range(250, n, 250))):
        red = TopKAbsReducer(
            (rf, 0, cf),
            k,
            side,
            floor,
            infinite=infinite,
            integer_cols=integer_cols,
            columns="continuous",
            factor=factor,
        )
        _feed(red, rid, cid, coef, cuts)
        entries, count = red.finalize()
        assert [e["post"] for e in entries] == list(want)
        assert count == int(near.sum())
        for e in entries:
            assert e["post"] == pytest.approx(
                e["model"] * abs(rf[e["rid"]]) * abs(cf[e["col_id"]]), rel=0
            )


def test_overflow_counts_with_a_second_pass() -> None:
    coef = np.arange(1.0, 1001.0)
    rid = np.arange(1000)
    cid = np.full(1000, -1)
    red = TopKAbsReducer((None, 0, None), 3, "max", factor=10.0, max_near=100)
    _feed(red, rid, cid, coef, [500])
    entries, count = red.finalize()
    assert [e["post"] for e in entries] == [1000.0, 999.0, 998.0]
    assert count is None and red.near_overflow
    counter = CountWithinReducer(red, red.best)
    counter.update(rid, cid, coef)
    # Values >= 1000 / 10 = 100: 100 .. 1000 → 901 entries.
    assert counter.finalize() == 901


def _enum_problem() -> Problem:
    """``cap[node, t] : x[node, t] <= limit[node, t]`` with Enum ``node``
    and Utf8 ``t``; limits 5, 0.5, 2e-15 (residue), 7, 50, 1e3."""
    node = pl.Enum(["n1", "n2"])
    over = pl.DataFrame(
        {"node": ["n1", "n1", "n1", "n2", "n2", "n2"], "t": ["t1", "t2", "t3"] * 2}
    ).with_columns(pl.col("node").cast(node))
    pb = Problem()
    ub = Param(("node", "t"), over.with_columns(value=pl.Series([1.0, 2, 3, 4, 5, 6])))
    x = pb.add_var("x", ("node", "t"), over, lower=0.0, upper=ub)
    limit = Param(
        ("node", "t"), over.with_columns(value=pl.Series([5.0, 0.5, 2e-15, 7.0, 50.0, 1e3]))
    )
    pb.add_cstr("cap", over=over, sense="<=", lhs_terms={"x": 0.25 * x}, rhs_terms={"l": limit})
    pb.set_objective(x, sense="min")
    return pb


def test_locate_rhs_min_skips_residue_and_decodes_dims() -> None:
    pb = _enum_problem()
    got = locate_extremes(pb, "cap", "rhs", "min", k=2)
    assert [(g["post"], g["dims"]) for g in got] == [
        (0.5, {"node": "n1", "t": "t2"}),
        (5.0, {"node": "n1", "t": "t1"}),
    ]
    # Within 10x of 0.5: 0.5 and 5.0 (7 is 14x).
    assert got[0]["count_within_factor"] == 2
    assert got[0]["component"] == "rhs" and got[0]["col_id"] == -1


def test_locate_rhs_max_and_row_factor() -> None:
    pb = _enum_problem()
    pb._layer2_row_factor = np.full(6, 0.5)
    got = locate_extremes(pb, "cap", "rhs", "max", k=1)
    assert got[0]["post"] == 500.0 and got[0]["model"] == 1e3
    assert got[0]["dims"] == {"node": "n2", "t": "t3"}


def test_locate_bound_and_matrix() -> None:
    pb = _enum_problem()
    got = locate_extremes(pb, "x", "bound", "max", k=1)
    assert (got[0]["post"], got[0]["dims"]) == (6.0, {"node": "n2", "t": "t3"})
    got = locate_extremes(pb, "x", "bound_rhs", "min", k=1)
    assert (got[0]["post"], got[0]["dims"]) == (1.0, {"node": "n1", "t": "t1"})
    got = locate_extremes(pb, "cap", "matrix", "min", k=1)
    assert got[0]["post"] == 0.25
    assert got[0]["column"]["family"] == "x"
    assert got[0]["count_within_factor"] == 6


def test_locate_two_million_rows_smoke(monkeypatch) -> None:
    """One 2M-row family: the walk stays bounded (batched) and is exact.
    Limits ``i + 1`` for row ``i``; row 1_234_567 holds 0.001 instead."""
    monkeypatch.setenv("POLAR_HIGH_RANGES_PROFILE", "1")
    n = 2_000_000
    over = pl.DataFrame({"i": np.arange(n, dtype=np.int64)})
    vals = np.arange(1.0, n + 1.0)
    vals[1_234_567] = 1e-3
    pb = Problem()
    x = pb.add_var("x", "i", over, lower=0.0, upper=math.inf)
    limit = Param(("i",), over.with_columns(value=pl.Series(vals)))
    pb.add_cstr("big", over=over, sense="<=", lhs_terms={"x": x}, rhs_terms={"l": limit})
    got = locate_extremes(pb, "big", "rhs", "min", k=3)
    assert [g["post"] for g in got] == [1e-3, 1.0, 2.0]
    assert got[0]["dims"] == {"i": 1_234_567}
    # Within 10x of 0.001: only itself (the next value is 1.0).
    assert got[0]["count_within_factor"] == 1
    assert os.environ["POLAR_HIGH_RANGES_PROFILE"] == "1"
