"""A Var whose frame rows share columns (one column held over a run of time steps, indexed on every step).

Built directly (not through ``Problem.add_var``) on top of a registered family: rows ``t1, t2`` carry the column of
``t1`` and rows ``t3, t4`` the column of ``t3``.  A term on it is summed per ``(row, col)`` like any duplicate
cell, so ``Sum`` over its time dim combines coefficients even though ``t`` is a Var dim.  Checked:

* the canonical matrix and the objective hold the summed coefficients, on the block-COO path (relabel would emit
  the per-row coefficients) and with block-COO off;
* the range readout (``_ranges_via_streaming``) reports the summed magnitudes, not the per-row ones;
* ``WarmProblem.update_param`` refuses a Param whose tracked cells repeat a ``(row, col)``.
"""

from __future__ import annotations

import polars as pl
import pytest

from polar_high.autoscale import ScalingConfig
from polar_high.autoscale._ranges import _ranges_via_streaming
from polar_high.engine import Param, Problem, Sum, Var, WarmProblem

STEPS = ["t1", "t2", "t3", "t4"]
FIRST = {"t1": "t1", "t2": "t1", "t3": "t3", "t4": "t3"}
# Per-step coefficients: the cells of column t1 sum to 1 + 3 = 4, those of column t3 to 5 + 7 = 12.
COEF = [1.0, 3.0, 5.0, 7.0]
COST = [2.0, 4.0, 6.0, 8.0]


@pytest.fixture(autouse=True)
def _small_dense(monkeypatch):
    monkeypatch.setenv("POLAR_HIGH_BLOCK_COO_MIN_DENSE", "1")
    monkeypatch.delenv("POLAR_HIGH_DISABLE_BLOCK_COO", raising=False)


def _shared_var(prob: Problem) -> tuple[Var, Var]:
    """The registered family ``x`` (columns at t1 and t3) and the per-step Var over its columns."""
    reg = prob.add_var(
        "x", ("p", "d", "t"), pl.DataFrame({"p": ["a", "a"], "d": ["d1", "d1"], "t": ["t1", "t3"]})
    )
    col = dict(zip(reg.frame["t"].to_list(), reg.frame["col_id"].to_list()))
    frame = pl.DataFrame(
        {
            "p": ["a"] * 4,
            "d": ["d1"] * 4,
            "t": STEPS,
            "col_id": pl.Series([col[FIRST[t]] for t in STEPS], dtype=reg.frame.schema["col_id"]),
        }
    )
    return reg, Var("x", ("p", "d", "t"), frame)


def _build() -> tuple[Problem, Var]:
    prob = Problem(dense_axes=("d", "t"))
    reg, x = _shared_var(prob)
    coef = Param(
        ("p", "d", "t"),
        pl.DataFrame({"p": ["a"] * 4, "d": ["d1"] * 4, "t": STEPS, "value": COEF}),
        name="p_coef",
    )
    cost = Param(
        ("p", "d", "t"), pl.DataFrame({"p": ["a"] * 4, "d": ["d1"] * 4, "t": STEPS, "value": COST})
    )
    over = pl.DataFrame({"p": ["a"], "d": ["d1"]})
    prob.add_cstr(
        "cap",
        over=over,
        sense="<=",
        lhs_terms={"x": Sum(x * coef, over=("t",))},
        rhs_terms={"c": 100.0},
    )
    prob.set_objective(Sum(x * cost), sense="min")
    return prob, reg


def _cells(prob: Problem) -> dict[int, float]:
    cm = prob.canonicalise()
    out: dict[int, float] = {}
    for j in range(cm.n_cols):
        for k in range(cm.col_ptr[j], cm.col_ptr[j + 1]):
            out[j] = out.get(j, 0.0) + float(cm.val[k])
            assert cm.row_idx[k] == 0
    return out


@pytest.mark.parametrize("block_coo", [True, False])
def test_matrix_and_objective_sum_shared_columns(monkeypatch, block_coo) -> None:
    if not block_coo:
        monkeypatch.setenv("POLAR_HIGH_DISABLE_BLOCK_COO", "1")
    prob, reg = _build()
    col = dict(zip(reg.frame["t"].to_list(), reg.frame["col_id"].to_list()))
    cm = prob.canonicalise()
    assert cm.nnz == 2
    assert _cells(prob) == {col["t1"]: 4.0, col["t3"]: 12.0}
    assert cm.col_obj[col["t1"]] == 6.0 and cm.col_obj[col["t3"]] == 14.0


def test_col_ids_unique() -> None:
    prob = Problem(dense_axes=("d", "t"))
    reg, x = _shared_var(prob)
    assert reg.col_ids_unique and not x.col_ids_unique


def test_range_readout_sees_summed_coefficients() -> None:
    """The matrix range is that of the summed cells (4, 12), not of the per-step terms (1 … 7); the cost range
    that of the summed costs (6, 14)."""
    prob, _ = _build()
    rep = _ranges_via_streaming(prob, ScalingConfig())
    assert rep.matrix == (4.0, 12.0)
    assert rep.cost == (6.0, 14.0)


class _CoeffSpy:
    """Wrap a HiGHS instance, recording every changeCoeff call."""

    def __init__(self, h):
        self._h = h
        self.calls: list[tuple[int, int, float]] = []

    def changeCoeff(self, r, c, v):  # noqa: N802 (mirror HiGHS API)
        self.calls.append((int(r), int(c), float(v)))
        return self._h.changeCoeff(r, c, v)

    def __getattr__(self, name):
        return getattr(self._h, name)


def _size(value: float, name: str | None = None) -> Param:
    return Param(("p", "d"), pl.DataFrame({"p": ["a"], "d": ["d1"], "value": [value]}), name=name)


@pytest.mark.parametrize("block_coo", [True, False])
def test_update_param_through_shared_columns(monkeypatch, block_coo) -> None:
    """``p_size`` (p, d) survives the Sum over t: each tracked cell is the summed coefficient, so an update
    rewrites the cells as a rebuild would (4·2 and 12·2), never one per-step term only."""
    if not block_coo:
        monkeypatch.setenv("POLAR_HIGH_DISABLE_BLOCK_COO", "1")
    prob = Problem(dense_axes=("d", "t"))
    reg, x = _shared_var(prob)
    col = dict(zip(reg.frame["t"].to_list(), reg.frame["col_id"].to_list()))
    coef = Param(
        ("p", "d", "t"), pl.DataFrame({"p": ["a"] * 4, "d": ["d1"] * 4, "t": STEPS, "value": COEF})
    )
    prob.add_cstr(
        "cap",
        over=pl.DataFrame({"p": ["a"], "d": ["d1"]}),
        sense="<=",
        lhs_terms={"x": Sum(x * coef * _size(1.5, "p_size"), over=("t",))},
        rhs_terms={"c": 100.0},
    )
    prob.set_objective(Sum(x * coef), sense="min")
    wp = WarmProblem(prob)
    wp.declare_mutable("p_size")
    wp.solve()
    spy = _CoeffSpy(wp._h)
    wp._h = spy
    wp.update_param("p_size", _size(2.0))
    assert sorted(spy.calls) == sorted([(0, col["t1"], 8.0), (0, col["t3"], 24.0)])


def test_update_param_refuses_repeated_cells() -> None:
    """``p_size`` in two terms of one row reaches each cell twice; a per-cell ``changeCoeff`` would keep only the
    last term, so the update is refused."""
    prob = Problem(dense_axes=("d", "t"))
    _reg, x = _shared_var(prob)
    size = _size(1.5, "p_size")
    prob.add_cstr(
        "cap",
        over=pl.DataFrame({"p": ["a"] * 2, "d": ["d1"] * 2, "t": ["t1", "t3"]}),
        sense="<=",
        lhs_terms={"x": x * size, "y": x * size * 2.0},
        rhs_terms={"c": 100.0},
    )
    prob.set_objective(Sum(x), sense="min")
    wp = WarmProblem(prob)
    wp.declare_mutable("p_size")
    wp.solve()
    with pytest.raises(ValueError, match="more than one tracked coefficient"):
        wp.update_param("p_size", _size(2.0))
