"""Coefficients whose bits do not depend on how polars lays the data out.

* A Param divided by a scalar is a multiply by the reciprocal in every row: polars divides a column by a
  literal per chunk (true quotient for a one-row chunk, reciprocal multiply for a longer one), and the chunk
  layout of a collected join changes from run to run.
* A ``Sum`` whose groups hold several coefficients of one cell adds them in the order of the term's dims,
  whatever order the rows arrive in — a polars hash ``group_by`` adds them in the order the rows (and its
  per-chunk / per-thread partial sums) happen to come in.
"""

from __future__ import annotations

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
