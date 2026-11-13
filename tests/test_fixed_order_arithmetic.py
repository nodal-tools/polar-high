"""Coefficients whose bits do not depend on how polars lays the data out.

* A Param divided by a scalar is a multiply by the reciprocal in every row: polars divides a column by a
  literal per chunk (true quotient for a one-row chunk, reciprocal multiply for a longer one), and the chunk
  layout of a collected join changes from run to run.
* A ``Sum`` whose groups hold several coefficients of one cell adds them in the order of the term's dims,
  whatever order the rows arrive in — a polars hash ``group_by`` adds them in the order the rows (and its
  per-chunk / per-thread partial sums) happen to come in.
"""

from __future__ import annotations

import io
import itertools
import sys

import numpy as np
import polars as pl
import pytest

import polar_high as fp
from polar_high._collect import collect_in_join_order
from polar_high.engine import _reciprocal


def _values(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.5, 2.0, n) * 10.0 ** rng.integers(-6, 7, n)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_param_div_scalar_same_bits_in_every_chunk(engine) -> None:
    rng = np.random.default_rng(7)
    s = 1.854907109208502
    lens = [1, 3, 1, 8, 1, 40]
    parts = [
        pl.DataFrame({"a": [f"a{len(lens)}_{i}_{j}" for j in range(n)], "value": _values(n, rng)})
        for i, n in enumerate(lens)
    ]
    frame = pl.concat(parts, rechunk=False)
    assert frame.n_chunks() == len(lens)
    v = frame["value"].to_numpy()
    # the two kernels polars mixes do differ on this data
    assert np.any(v / s != v * (1.0 / s))
    got = (fp.Param(("a",), frame) / s).lazy.collect(engine=engine)["value"].to_numpy()
    assert np.array_equal(got.view(np.int64), (v * (1.0 / s)).view(np.int64))


def test_reciprocal_ieee_edge_cases() -> None:
    assert _reciprocal(0.0) == np.inf and _reciprocal(-0.0) == -np.inf
    assert _reciprocal(np.inf) == 0.0 and _reciprocal(-np.inf) == 0.0
    assert np.isnan(_reciprocal(np.nan)) and _reciprocal(4.0) == 0.25


def _three_entry_sum(t_order: list[str], p_order: list[str]) -> pl.DataFrame:
    """``Σ_t x[p] · b[t]`` with b = (1e16, 1, −1e16): every coefficient is a three-entry group whose float sum
    is 0 or 1 depending on the order its entries are added in."""
    bvals = {"t0": 1e16, "t1": 1.0, "t2": -1e16}
    b = fp.Param(("t",), pl.DataFrame({"t": t_order, "value": [bvals[t] for t in t_order]}))
    prob = fp.Problem()
    x = prob.add_var("x", ("p",), pl.DataFrame({"p": p_order}))
    out = fp.Sum(x * b, over=("t",))
    df = collect_in_join_order(out.terms[0].lazy)
    return df.with_columns(p=pl.col("col_id").replace_strict(x.frame["col_id"], x.frame["p"])).sort(
        "p"
    )


def test_sum_adds_a_group_in_dim_order_whatever_the_row_order() -> None:
    ts = ["t0", "t1", "t2"]
    ps = ["p0", "p1", "p2", "p3"]
    # (1e16 + 1) + −1e16 = 0 in dim order t0, t1, t2
    expected = (1e16 + 1.0) + -1e16
    assert expected == 0.0
    rng = np.random.default_rng(0)
    for _ in range(6):
        t_order = [ts[i] for i in rng.permutation(3)]
        p_order = [ps[i] for i in rng.permutation(4)]
        df = _three_entry_sum(t_order, p_order)
        assert df.height == 4
        assert df["coef"].to_list() == [expected] * 4


def test_sum_relabel_keeps_single_row_groups() -> None:
    """A Sum over Var dims only (a relabel) is one coefficient per group: unchanged values."""
    prob = fp.Problem()
    x = prob.add_var(
        "x", ("p", "t"), pl.DataFrame({"p": ["p0", "p0", "p1"], "t": ["t0", "t1", "t0"]})
    )
    c = fp.Param(
        ("p", "t"),
        pl.DataFrame({"p": ["p0", "p0", "p1"], "t": ["t0", "t1", "t0"], "value": [0.1, 0.2, 0.3]}),
    )
    df = collect_in_join_order(fp.Sum(x * c, over=("t",)).terms[0].lazy).sort("col_id")
    assert df.columns == ["p", "col_id", "coef"]
    assert df["coef"].to_list() == [0.1, 0.2, 0.3]


def test_sum_skips_a_null_coefficient() -> None:
    """A null coefficient in a multi-row group is skipped, as polars' ``sum`` skips it; an all-null group
    sums to 0."""
    b = fp.Param(
        ("p", "t"),
        pl.DataFrame(
            {
                "p": ["p0", "p0", "p0", "p1", "p1"],
                "t": ["t0", "t1", "t2", "t0", "t1"],
                "value": [1.0, None, 2.5, None, None],
            },
            schema={"p": pl.String, "t": pl.String, "value": pl.Float64},
        ),
    )
    prob = fp.Problem()
    x = prob.add_var("x", ("p",), pl.DataFrame({"p": ["p0", "p1"]}))
    df = collect_in_join_order(fp.Sum(x * b, over=("t",)).terms[0].lazy)
    got = dict(zip(df["col_id"].replace_strict(x.frame["col_id"], x.frame["p"]), df["coef"]))
    assert got == {"p0": 3.5, "p1": 0.0}


# --------------------------------------------------------------------------------------------------------- #
# Block-COO combining reduce vs the reduced ``term.lazy`` (block-COO off): the same bits                    #
# --------------------------------------------------------------------------------------------------------- #

_P = ["a", "b"]
_S = ["s2", "s0", "s1"]
_T = ["t1", "t2", "t3", "t4"]
# Values that cancel: the bits of a group's sum depend on the order its entries are added in.
_VALS = [1e16, -1e16, 1.0, 3.0, -7.0, 1e-3]


def _enum_cast(frame: pl.DataFrame, enum: bool) -> pl.DataFrame:
    if not enum:
        return frame
    # Categories in reverse lexical order: an Enum sorts by its physical order, a String lexically.
    return frame.with_columns(
        pl.col("s").cast(pl.Enum(sorted(_S, reverse=True))),
        pl.col("t").cast(pl.Enum(sorted(_T, reverse=True))),
    )


def _shared_col_cstr(seed: int, enum: bool) -> fp.Problem:
    """``Sum(x[p,s,d,t] · c[t,s,p,d], over=(s, t))`` on a Var whose rows share one column per ``p``: each
    constraint cell adds 12 coefficients, drawn from ``_VALS``, Param rows shuffled and its dims in another
    order than the Var's."""
    rng = np.random.default_rng(seed)
    prob = fp.Problem(dense_axes=("d", "t"))
    reg = prob.add_var(
        "x",
        ("p", "s", "d", "t"),
        pl.DataFrame({"p": _P, "s": ["s0"] * 2, "d": ["d1"] * 2, "t": ["t1"] * 2}),
    )
    cid = dict(zip(reg.frame["p"].to_list(), reg.frame["col_id"].to_list()))
    rows = list(itertools.product(_P, _S, ["d1"], _T))
    frame = pl.DataFrame(rows, schema=["p", "s", "d", "t"], orient="row").with_columns(
        col_id=pl.Series([cid[r[0]] for r in rows], dtype=reg.frame.schema["col_id"])
    )
    x = fp.Var("x", ("p", "s", "d", "t"), _enum_cast(frame, enum).sort(["p", "s", "d", "t"]))
    perm = rng.permutation(len(rows))
    cf = pl.DataFrame([rows[i] for i in perm], schema=["p", "s", "d", "t"], orient="row")
    cf = cf.with_columns(value=pl.Series(rng.choice(_VALS, len(rows))))
    c = fp.Param(("t", "s", "p", "d"), _enum_cast(cf.select("t", "s", "p", "d", "value"), enum))
    prob.add_cstr(
        "cap",
        over=pl.DataFrame({"p": _P, "d": ["d1", "d1"]}),
        sense="<=",
        lhs_terms={"x": fp.Sum(x * c, over=("s", "t"))},
        rhs_terms={"c": 100.0},
    )
    return prob


def _reduceat(vals: list[float]) -> float:
    """``numpy.add.reduceat`` of one group, the reduction both paths use (not left to right: numpy fixes its
    own association)."""
    return float(np.add.reduceat(np.asarray(vals, dtype=np.float64), [0])[0])


def _cell_entries(seed: int, enum: bool, s_first: bool = True) -> list[list[float]]:
    """Each ``p``'s 12 coefficients of :func:`_shared_col_cstr`, in ``(s, t)`` (or ``(t, s)``) sort order."""
    rng = np.random.default_rng(seed)
    rows = list(itertools.product(_P, _S, ["d1"], _T))
    perm = rng.permutation(len(rows))
    vals = rng.choice(_VALS, len(rows))
    by_row = {rows[k]: v for k, v in zip(perm.tolist(), vals.tolist())}
    s_order = sorted(_S, reverse=enum)
    t_order = sorted(_T, reverse=enum)
    if s_first:
        return [[by_row[(p, s, "d1", t)] for s in s_order for t in t_order] for p in _P]
    return [[by_row[(p, s, "d1", t)] for t in t_order for s in s_order] for p in _P]


@pytest.mark.parametrize("enum", [False, True])
def test_block_coo_combining_reduce_same_bits_as_lazy_sum(monkeypatch, enum) -> None:
    monkeypatch.setenv("POLAR_HIGH_BLOCK_COO_MIN_DENSE", "1")
    monkeypatch.delenv("POLAR_HIGH_DISABLE_BLOCK_COO", raising=False)
    order_matters = False
    for seed in range(8):
        monkeypatch.setenv("POLAR_HIGH_BLOCK_COO_PROFILE", "1")
        buf = io.StringIO()
        monkeypatch.setattr(sys, "stderr", buf)
        on = np.asarray(_shared_col_cstr(seed, enum).canonicalise().val, dtype=np.float64)
        monkeypatch.setattr(sys, "stderr", sys.__stderr__)
        monkeypatch.delenv("POLAR_HIGH_BLOCK_COO_PROFILE")
        assert "kind=sum\tpath=combining" in buf.getvalue(), (
            "the block-COO combining reduce must fire"
        )
        monkeypatch.setenv("POLAR_HIGH_DISABLE_BLOCK_COO", "1")
        off = np.asarray(_shared_col_cstr(seed, enum).canonicalise().val, dtype=np.float64)
        monkeypatch.delenv("POLAR_HIGH_DISABLE_BLOCK_COO")
        assert on.shape == off.shape == (2,)
        assert np.array_equal(on.view(np.int64), off.view(np.int64)), (seed, on, off)
        # an independent reference: each cell's entries in the term's dim order (s, t) — an Enum by its
        # categories, a String lexically — not the Param's (t, s); and that order does change a sum somewhere
        ref = [_reduceat(cell) for cell in _cell_entries(seed, enum)]
        assert on.tolist() == ref
        if ref != [_reduceat(cell) for cell in _cell_entries(seed, enum, s_first=False)]:
            order_matters = True
    assert order_matters
