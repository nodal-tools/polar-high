"""Phase 3 tests for ``polar_high.solvers._lp_view.LpView``.

These verify:

* Shape sanity (n_cols, n_rows, name lengths, nnz)
* ``row_sense_rhs`` mapping for E / L / G / R rows
* ``split_ranged_rows`` correctness on a ranged row
* ``LpView.from_problem`` has zero side effects on the source Problem
"""

from __future__ import annotations

import numpy as np
import polars as pl

from polar_high import Problem
from polar_high.solvers._lp_view import LpView


def _toy_with_all_senses() -> Problem:
    """Build a tiny Problem with one row of each sense (E / L / G).

    The public ``add_cstr`` API supports ``==`` / ``<=`` / ``>=`` only,
    so we cover three of the four sense cases here.  The fourth case
    ("R", ranged) is exercised against a hand-built :class:`LpView` in
    :func:`_hand_built_ranged_view`.

    Variables (over a single-row index ``i``):
      x in [0, 5], y in [0, 5]
    Rows:
      e1 :  x + y == 3                       (E)
      l1 :  x + 2y <= 8                      (L)
      g1 :  2x + y >= 1                      (G)
    """
    pb = Problem()
    idx = pl.DataFrame({"i": [0]})
    x = pb.add_var("x", dims=("i",), index=idx, lower=0.0, upper=5.0)
    y = pb.add_var("y", dims=("i",), index=idx, lower=0.0, upper=5.0)

    pb.add_cstr("e1", over=idx, sense="==", lhs_terms={"x": x, "y": y}, rhs_terms={"c": 3.0})
    pb.add_cstr("l1", over=idx, sense="<=", lhs_terms={"x": x, "y2": 2.0 * y}, rhs_terms={"c": 8.0})
    pb.add_cstr("g1", over=idx, sense=">=", lhs_terms={"x2": 2.0 * x, "y": y}, rhs_terms={"c": 1.0})

    pb.set_objective(x + y, sense="min")
    return pb


def _hand_built_ranged_view() -> LpView:
    """Hand-built LpView with E, L, G, and R rows for sense + split tests.

    columns: x, y  (both [0, 5])
    rows:
      r0 (E): x + y == 3
      r1 (L): x + 2y <= 8
      r2 (G): 2x + y >= 1
      r3 (R): 1 <= x + y <= 4
    """
    col_obj = np.array([1.0, 1.0])
    col_lb = np.array([0.0, 0.0])
    col_ub = np.array([5.0, 5.0])

    row_lb = np.array([3.0, -np.inf, 1.0, 1.0])
    row_ub = np.array([3.0, 8.0, np.inf, 4.0])

    # CSC matrix
    # column 0 (x): touches rows 0,1,2,3 with coefs 1,1,2,1
    # column 1 (y): touches rows 0,1,2,3 with coefs 1,2,1,1
    a_start = np.array([0, 4, 8], dtype=np.int32)
    a_index = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
    a_value = np.array([1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0])

    return LpView(
        n_cols=2,
        n_rows=4,
        col_obj=col_obj,
        col_lb=col_lb,
        col_ub=col_ub,
        integrality=None,
        row_lb=row_lb,
        row_ub=row_ub,
        a_start=a_start,
        a_index=a_index,
        a_value=a_value,
        col_names=["x", "y"],
        row_names=["e1", "l1", "g1", "r1"],
        sense="min",
        obj_offset=0.0,
    )


def test_from_problem_shape_sanity() -> None:
    """``LpView.from_problem`` populates every field consistently."""
    pb = _toy_with_all_senses()
    view = LpView.from_problem(pb)

    assert view.n_cols == 2
    assert view.n_rows == 3  # E, L, G (no R via the public API)
    assert len(view.col_names) == view.n_cols
    assert len(view.row_names) == view.n_rows
    assert view.col_obj.shape == (view.n_cols,)
    assert view.col_lb.shape == (view.n_cols,)
    assert view.col_ub.shape == (view.n_cols,)
    assert view.row_lb.shape == (view.n_rows,)
    assert view.row_ub.shape == (view.n_rows,)
    assert view.a_start.shape == (view.n_cols + 1,)
    nnz = int(view.a_start[-1])
    assert view.a_index.shape == (nnz,)
    assert view.a_value.shape == (nnz,)
    # 3 rows × 2 cols, each non-zero → 6 nnz before dedup; coefs may
    # combine if duplicated, but our toy has no duplicates.
    assert nnz == 6
    assert view.integrality is None  # pure LP
    assert view.sense == "min"
    assert view.obj_offset == 0.0
    # Bounds use ±inf, not HiGHS' kHighsInf
    assert view.col_ub[0] == 5.0
    # row 1 is "<=" → row_lb is -inf
    assert np.isneginf(view.row_lb[1])
    # row 2 is ">=" → row_ub is +inf
    assert np.isposinf(view.row_ub[2])


def test_row_sense_rhs_all_four_cases() -> None:
    """``row_sense_rhs`` maps E / L / G / R correctly."""
    view = _hand_built_ranged_view()
    senses, rhs, ranges = view.row_sense_rhs()

    # E row (== 3)
    assert senses[0] == "E"
    assert rhs[0] == 3.0
    assert ranges[0] == 0.0

    # L row (<= 8)
    assert senses[1] == "L"
    assert rhs[1] == 8.0
    assert ranges[1] == 0.0

    # G row (>= 1)
    assert senses[2] == "G"
    assert rhs[2] == 1.0
    assert ranges[2] == 0.0

    # R row (1 <= ... <= 4): CPLEX convention rhs=row_ub, range = lb - ub
    assert senses[3] == "R"
    assert rhs[3] == 4.0
    assert ranges[3] == -3.0


def test_split_ranged_rows_expands_pair() -> None:
    """``split_ranged_rows`` replaces the one ranged row with a >=/<= pair.

    The new view has one extra row, the CSC entries for the original
    ranged row are duplicated in the new row, and the names are
    suffixed ``_lo`` / ``_hi``.
    """
    view = _hand_built_ranged_view()
    split = view.split_ranged_rows()

    # +1 row
    assert split.n_rows == view.n_rows + 1
    # original column count is preserved
    assert split.n_cols == view.n_cols

    # First 3 rows are unchanged (E, L, G)
    assert split.row_names[:3] == ["e1", "l1", "g1"]
    # The ranged row is split: "r1_lo" stays in the original slot 3 with
    # row_ub bumped to +inf; "r1_hi" is appended at the tail with
    # row_lb = -inf and row_ub = original row_ub.
    assert split.row_names[3] == "r1_lo"
    assert split.row_names[4] == "r1_hi"
    assert split.row_lb[3] == 1.0
    assert np.isposinf(split.row_ub[3])
    assert np.isneginf(split.row_lb[4])
    assert split.row_ub[4] == 4.0

    # Matrix entries: each column originally had a nonzero in the
    # ranged row → split view must have a matching nonzero in the new
    # "_hi" row, with the same coefficient value.
    n_cols = split.n_cols
    for c in range(n_cols):
        lo = int(split.a_start[c])
        hi = int(split.a_start[c + 1])
        rows_in_col = split.a_index[lo:hi]
        vals_in_col = split.a_value[lo:hi]
        # original ranged row id was 3 → now "r1_lo"; the duplicate
        # lives at row id 4 ("r1_hi") with the same value.
        idx_lo = np.where(rows_in_col == 3)[0]
        idx_hi = np.where(rows_in_col == 4)[0]
        assert idx_lo.size == 1
        assert idx_hi.size == 1
        assert vals_in_col[int(idx_lo[0])] == vals_in_col[int(idx_hi[0])]

    # Column data unchanged
    assert np.array_equal(split.col_obj, view.col_obj)
    assert np.array_equal(split.col_lb, view.col_lb)
    assert np.array_equal(split.col_ub, view.col_ub)


def test_from_problem_has_no_side_effects() -> None:
    """Extracting an :class:`LpView` twice from the same Problem yields
    identical arrays, and a subsequent ``p.solve()`` is unaffected."""
    pb = _toy_with_all_senses()
    v1 = LpView.from_problem(pb)
    v2 = LpView.from_problem(pb)

    assert v1.n_cols == v2.n_cols
    assert v1.n_rows == v2.n_rows
    assert np.array_equal(v1.col_obj, v2.col_obj)
    assert np.array_equal(v1.col_lb, v2.col_lb)
    assert np.array_equal(v1.col_ub, v2.col_ub)
    assert np.array_equal(v1.row_lb, v2.row_lb)
    assert np.array_equal(v1.row_ub, v2.row_ub)
    assert np.array_equal(v1.a_start, v2.a_start)
    assert np.array_equal(v1.a_index, v2.a_index)
    assert np.array_equal(v1.a_value, v2.a_value)
    assert v1.col_names == v2.col_names
    assert v1.row_names == v2.row_names
    assert v1.sense == v2.sense
    assert v1.obj_offset == v2.obj_offset
    assert v1.integrality is None and v2.integrality is None

    # And the Problem itself is still solvable to the analytic optimum:
    # minimize x + y s.t. x + y == 3, x + 2y <= 8, 2x + y >= 1.
    # The equality fixes x + y = 3 → objective = 3.
    sol = pb.solve()
    assert sol.optimal
    assert abs(sol.obj - 3.0) < 1e-9
