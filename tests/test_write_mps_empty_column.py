"""``Problem.write_mps`` keeps an EMPTY column in place.

An empty column has no matrix entry and no objective coefficient.  The
writer used to skip it in the COLUMNS section, so it appeared only in
BOUNDS (non-default bounds) or nowhere (default bounds).  HiGHS'
``readModel`` then moved it to the END of the column list or dropped it,
so every later column's position shifted and a caller that maps the
solution back by position read wrong values (an integer empty column
also lost its integrality).

The fix writes every column; an empty one gets an explicit ``cost 0``
entry.  These tests build ``a``, ``e``, ``b`` over ``i={0}`` in that
order, with ``e`` in no row and not in the objective, and check that the
MPS read back by HiGHS has every column, in order, with ``e``'s bounds
and integrality, and solves to the same per-variable values as the
in-process :meth:`Problem.solve`.
"""

from __future__ import annotations

import highspy
import polars as pl
import pytest

from polar_high import Problem

# (lower, upper, integer) of the empty column ``e``.
_E_VARIANTS = {
    "default_bounds": (0.0, float("inf"), False),
    "fixed_5": (5.0, 5.0, False),
    "integer_ub7": (0.0, 7.0, True),
}


def _build(e_lower: float, e_upper: float, e_integer: bool) -> Problem:
    pb = Problem()
    idx = pl.DataFrame({"i": [0]})
    a = pb.add_var("a", dims=("i",), index=idx)
    pb.add_var("e", dims=("i",), index=idx, lower=e_lower, upper=e_upper, integer=e_integer)
    b = pb.add_var("b", dims=("i",), index=idx)
    pb.add_cstr("c", over=idx, sense=">=", lhs_terms={"a": a, "b": b}, rhs_terms={"r": 3.0})
    pb.set_objective(a + 2.0 * b, sense="min")
    return pb


def _read_back(path: str) -> highspy.Highs:
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    status = h.readModel(path)
    assert status != highspy.HighsStatus.kError
    return h


@pytest.mark.parametrize("emit_names", [True, False], ids=["names", "generic"])
@pytest.mark.parametrize("variant", list(_E_VARIANTS))
def test_empty_column_kept_in_place(tmp_path, variant: str, emit_names: bool) -> None:
    e_lower, e_upper, e_integer = _E_VARIANTS[variant]

    direct = _build(e_lower, e_upper, e_integer).solve()
    assert direct.optimal
    assert len(direct.col_names) == 3
    assert direct.col_names[0].startswith("a") and direct.col_names[2].startswith("b")
    by_name = dict(zip(direct.col_names, direct.col_value.tolist()))
    # The in-process reference itself: a=3, b=0, e at its lower bound.
    assert by_name[direct.col_names[0]] == pytest.approx(3.0)
    assert by_name[direct.col_names[2]] == pytest.approx(0.0)
    assert by_name[direct.col_names[1]] == pytest.approx(e_lower)

    pb = _build(e_lower, e_upper, e_integer)
    mps = tmp_path / f"{variant}_{emit_names}.mps"
    pb.write_mps(str(mps), emit_names=emit_names)

    h = _read_back(str(mps))
    assert h.getNumCol() == 3

    lp = h.getLp()
    expected_names = list(direct.col_names) if emit_names else [f"C{j + 1:07d}" for j in range(3)]
    assert list(lp.col_names_) == expected_names

    # e (position 1) keeps its bounds and integrality.
    assert lp.col_lower_[1] == e_lower
    if e_upper == float("inf"):
        assert lp.col_upper_[1] >= highspy.kHighsInf
    else:
        assert lp.col_upper_[1] == e_upper
    integ = (
        list(lp.integrality_) if len(lp.integrality_) else [highspy.HighsVarType.kContinuous] * 3
    )
    want_e = highspy.HighsVarType.kInteger if e_integer else highspy.HighsVarType.kContinuous
    assert int(integ[1]) == int(want_e), integ
    assert int(integ[0]) == int(highspy.HighsVarType.kContinuous)
    assert int(integ[2]) == int(highspy.HighsVarType.kContinuous)

    h.run()
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    assert float(h.getObjectiveValue()) == pytest.approx(direct.obj)
    values = list(h.getSolution().col_value)
    assert len(values) == 3
    for j, name in enumerate(direct.col_names):
        assert values[j] == pytest.approx(by_name[name], abs=1e-9), (name, values, by_name)

    # The writer declares e with an explicit zero cost in COLUMNS.
    text = mps.read_text()
    e_name = direct.col_names[1] if emit_names else "C0000002"
    assert f"    {e_name}  cost  0\n" in text, text


def test_matrix_column_with_zero_cost_gets_no_cost_line(tmp_path) -> None:
    """Only empty columns get the explicit ``cost 0``: ``a`` and ``b`` here
    have matrix entries, ``a`` has zero cost and must get no cost line."""
    pb = Problem()
    idx = pl.DataFrame({"i": [0]})
    a = pb.add_var("a", dims=("i",), index=idx)
    pb.add_var("e", dims=("i",), index=idx)
    b = pb.add_var("b", dims=("i",), index=idx)
    pb.add_cstr("c", over=idx, sense=">=", lhs_terms={"a": a, "b": b}, rhs_terms={"r": 3.0})
    pb.set_objective(2.0 * b, sense="min")
    mps = tmp_path / "zero_cost.mps"
    pb.write_mps(str(mps), emit_names=False)
    cost_lines = [ln for ln in mps.read_text().splitlines() if "  cost  " in ln]
    assert cost_lines == ["    C0000002  cost  0", "    C0000003  cost  2"], cost_lines


@pytest.mark.parametrize("variant", list(_E_VARIANTS))
def test_fingerprint_generic_equals_named(tmp_path, variant: str) -> None:
    """``emit_names=False`` still stashes the real-name fingerprint."""
    e_lower, e_upper, e_integer = _E_VARIANTS[variant]
    pb_named = _build(e_lower, e_upper, e_integer)
    pb_named.write_mps(str(tmp_path / "named.mps"), emit_names=True)
    pb_generic = _build(e_lower, e_upper, e_integer)
    pb_generic.write_mps(str(tmp_path / "generic.mps"), emit_names=False)

    assert pb_generic._last_mps_fingerprint is not None
    assert pb_generic._last_mps_fingerprint == pb_named._last_mps_fingerprint
