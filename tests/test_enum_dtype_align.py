"""Tests for the generic Enum-dtype alignment helper at internal
join sites.

polars 1.40 refuses to join two ``pl.Enum`` columns with different
categorical vocabularies — even when one side's categories are a
strict subset of the other's.  polar_high auto-aligns by up-casting
the narrower side, and raises a clear error when neither vocab is a
subset.  These tests exercise the helper directly and end-to-end
through ``Problem.add_var`` + ``add_cstr`` + ``solve``.
"""

from __future__ import annotations

import polars as pl
import pytest

from polar_high import Param, Problem
from polar_high.engine import _align_enum_join_keys

# -- helper: direct unit tests ------------------------------------------


def test_same_enum_passthrough():
    """Same-Enum on both sides: no change to either frame; join works."""
    e = pl.Enum(["a", "b", "c"])
    left = pl.DataFrame({"k": pl.Series(["a", "b"], dtype=e), "v": [1, 2]})
    right = pl.DataFrame({"k": pl.Series(["a", "b"], dtype=e), "w": [10, 20]})
    l2, r2 = _align_enum_join_keys(left, right, ["k"])
    assert l2.schema["k"] == e
    assert r2.schema["k"] == e
    j = l2.join(r2, on="k", how="inner")
    assert j.height == 2


def test_narrower_left_upcast_to_wider_right():
    """Left's vocab ⊂ right's vocab → left's column is up-cast."""
    narrow = pl.Enum(["a", "b"])
    wide = pl.Enum(["a", "b", "c"])
    left = pl.DataFrame({"k": pl.Series(["a", "b"], dtype=narrow), "v": [1, 2]})
    right = pl.DataFrame({"k": pl.Series(["a", "b", "c"], dtype=wide), "w": [10, 20, 30]})
    l2, r2 = _align_enum_join_keys(left, right, ["k"])
    assert l2.schema["k"] == wide
    assert r2.schema["k"] == wide
    j = l2.join(r2, on="k", how="inner").sort("k")
    assert j["v"].to_list() == [1, 2]
    assert j["w"].to_list() == [10, 20]


def test_narrower_right_upcast_to_wider_left():
    """Right's vocab ⊂ left's vocab → right's column is up-cast."""
    narrow = pl.Enum(["a", "b"])
    wide = pl.Enum(["a", "b", "c"])
    left = pl.DataFrame({"k": pl.Series(["a", "b", "c"], dtype=wide), "v": [1, 2, 3]})
    right = pl.DataFrame({"k": pl.Series(["a", "b"], dtype=narrow), "w": [10, 20]})
    l2, r2 = _align_enum_join_keys(left, right, ["k"])
    assert l2.schema["k"] == wide
    assert r2.schema["k"] == wide
    j = l2.join(r2, on="k", how="inner").sort("k")
    assert j["v"].to_list() == [1, 2]
    assert j["w"].to_list() == [10, 20]


def test_enum_vs_utf8_casts_utf8_to_enum():
    """Enum on one side, Utf8 on the other → Utf8 is cast to Enum."""
    e = pl.Enum(["a", "b", "c"])
    left = pl.DataFrame({"k": pl.Series(["a", "b"], dtype=e), "v": [1, 2]})
    right = pl.DataFrame({"k": ["a", "b"], "w": [10, 20]})
    assert right.schema["k"] == pl.Utf8 or right.schema["k"] == pl.String
    l2, r2 = _align_enum_join_keys(left, right, ["k"])
    assert l2.schema["k"] == e
    assert r2.schema["k"] == e
    j = l2.join(r2, on="k", how="inner").sort("k")
    assert j["v"].to_list() == [1, 2]


def test_disjoint_enums_raise():
    """Two Enums with disjoint vocabs → ValueError with guidance."""
    e1 = pl.Enum(["a", "b", "c"])
    e2 = pl.Enum(["x", "y", "z"])
    left = pl.DataFrame({"k": pl.Series(["a", "b"], dtype=e1), "v": [1, 2]})
    right = pl.DataFrame({"k": pl.Series(["x", "y"], dtype=e2), "w": [10, 20]})
    with pytest.raises(ValueError) as excinfo:
        _align_enum_join_keys(left, right, ["k"])
    msg = str(excinfo.value)
    assert "no subset relation" in msg
    assert "'k'" in msg
    assert "pl.Utf8" in msg or "union Enum" in msg


def test_overlapping_neither_subset_raises():
    """Two Enums sharing some but not all categories → ValueError."""
    e1 = pl.Enum(["a", "b", "c"])
    e2 = pl.Enum(["b", "c", "d"])
    left = pl.DataFrame({"k": pl.Series(["a", "b"], dtype=e1), "v": [1, 2]})
    right = pl.DataFrame({"k": pl.Series(["b", "c"], dtype=e2), "w": [10, 20]})
    with pytest.raises(ValueError) as excinfo:
        _align_enum_join_keys(left, right, ["k"])
    assert "no subset relation" in str(excinfo.value)


def test_lazyframe_inputs_return_lazy():
    """LazyFrame in → LazyFrame out; alignment happens in the plan."""
    narrow = pl.Enum(["a", "b"])
    wide = pl.Enum(["a", "b", "c"])
    left = pl.DataFrame({"k": pl.Series(["a", "b"], dtype=narrow), "v": [1, 2]}).lazy()
    right = pl.DataFrame({"k": pl.Series(["a", "b", "c"], dtype=wide), "w": [10, 20, 30]}).lazy()
    l2, r2 = _align_enum_join_keys(left, right, ["k"])
    assert isinstance(l2, pl.LazyFrame)
    assert isinstance(r2, pl.LazyFrame)
    j = l2.join(r2, on="k", how="inner").collect().sort("k")
    assert j["v"].to_list() == [1, 2]


def test_empty_on_list_is_noop():
    """Empty ``on`` (cross-join-style) returns inputs untouched."""
    left = pl.DataFrame({"v": [1, 2]})
    right = pl.DataFrame({"w": [10, 20]})
    l2, r2 = _align_enum_join_keys(left, right, [])
    assert l2 is left and r2 is right


def test_missing_column_skipped():
    """Key absent on one side: skip — not the helper's job to validate."""
    e = pl.Enum(["a", "b"])
    left = pl.DataFrame({"k": pl.Series(["a", "b"], dtype=e), "v": [1, 2]})
    right = pl.DataFrame({"other": [10, 20]})
    # Should not raise; alignment for "k" is a no-op since right lacks it.
    l2, r2 = _align_enum_join_keys(left, right, ["k"])
    assert l2.schema["k"] == e


def test_non_enum_mismatch_left_unchanged():
    """Dtype mismatch that isn't Enum-vs-Enum / Enum-vs-Utf8: leave it.
    polars's normal coercion produces its own (clear) error if needed."""
    left = pl.DataFrame({"k": [1, 2], "v": [10, 20]})  # Int64
    right = pl.DataFrame({"k": [1.0, 2.0], "w": [100, 200]})  # Float64
    l2, r2 = _align_enum_join_keys(left, right, ["k"])
    # No silent re-casting outside the documented Enum / Utf8 paths.
    assert l2.schema["k"] == pl.Int64
    assert r2.schema["k"] == pl.Float64


# -- end-to-end: Problem.add_cstr with cross-Enum-vocab rhs Param -------


def test_problem_solve_with_mixed_enum_vocab_rhs():
    """tiny LP where the rhs Param's frame has a narrower Enum vocab
    than the row_index (constraint ``over``).  Must build without
    SchemaError and solve to the expected optimum."""
    # Wide vocab on the variable / constraint axis: three processes,
    # but the rhs cap Param only carries the two processes that have a
    # finite cap.
    wide = pl.Enum(["wind", "coal", "gas"])
    narrow = pl.Enum(["coal", "gas"])

    p = Problem()

    # Variable: v_production[unit] >= 0 for all three units.
    index = pl.DataFrame({"unit": pl.Series(["wind", "coal", "gas"], dtype=wide)})
    v = p.add_var("v_production", dims=("unit",), index=index, lower=0.0)

    # Cost: minimise so the LP has bounded production via the rhs cap.
    cost = Param(
        ("unit",),
        pl.DataFrame(
            {
                "unit": pl.Series(["wind", "coal", "gas"], dtype=wide),
                "value": [-1.0, -2.0, -3.0],  # negative → maximise production
            }
        ),
    )
    p.set_objective(cost * v, sense="min")

    # Capacity Param uses the NARROWER vocab — only coal and gas have
    # explicit caps.  This is where 4.8/4.9 used to require a cast.
    cap = Param(
        ("unit",),
        pl.DataFrame(
            {
                "unit": pl.Series(["coal", "gas"], dtype=narrow),
                "value": [5.0, 10.0],
            }
        ),
    )

    # Constraint indexed on the WIDER axis — left-join into cap will
    # be Enum-wide vs Enum-narrow on the join key.
    p.add_cstr(
        "cap_cstr",
        over=index,
        lhs_terms={"prod": v},
        sense="<=",
        rhs_terms={"cap": cap},
    )

    # Also bound wind from above so the LP is bounded (wind has no cap
    # row → left-join fills with 0.0 → wind production forced to 0).
    sol = p.solve()
    assert sol.optimal
    # Expected: coal=5, gas=10, wind=0; obj = 0*-1 + 5*-2 + 10*-3 = -40.
    assert abs(sol.obj - (-40.0)) < 1e-6
    vals = sol.value("v_production").sort("unit")
    assert vals.filter(pl.col("unit") == "coal")["value"][0] == pytest.approx(5.0)
    assert vals.filter(pl.col("unit") == "gas")["value"][0] == pytest.approx(10.0)
    assert vals.filter(pl.col("unit") == "wind")["value"][0] == pytest.approx(0.0)


def test_problem_solve_with_param_param_mixed_enum_vocab():
    """``Param * Param`` on mixed-vocab Enum dim: helper must reconcile
    inside __mul__ so the resulting Param frame stays well-typed and
    feeds an LP."""
    wide = pl.Enum(["wind", "coal", "gas"])
    narrow = pl.Enum(["coal", "gas"])

    p = Problem()
    index = pl.DataFrame({"unit": pl.Series(["wind", "coal", "gas"], dtype=wide)})
    v = p.add_var("v", dims=("unit",), index=index, lower=0.0, upper=1.0)

    a = Param(
        ("unit",),
        pl.DataFrame(
            {
                "unit": pl.Series(["wind", "coal", "gas"], dtype=wide),
                "value": [2.0, 3.0, 4.0],
            }
        ),
    )
    b = Param(
        ("unit",),
        pl.DataFrame(
            {
                "unit": pl.Series(["coal", "gas"], dtype=narrow),
                "value": [10.0, 20.0],
            }
        ),
    )
    # Without alignment this raises SchemaError at the Param*Param join.
    ab = a * b
    coll = ab.frame.sort("unit")
    # Only the (coal, gas) intersection survives the inner-join.
    assert coll["unit"].to_list() == ["coal", "gas"]
    assert coll["value"].to_list() == [pytest.approx(30.0), pytest.approx(80.0)]
    # Use the result as an objective coefficient to confirm the LP path.
    p.set_objective(ab * v, sense="max")
    p.add_cstr("cap", over=index, lhs_terms={"prod": v}, sense="<=", rhs_terms={"rhs": 1.0})
    sol = p.solve()
    assert sol.optimal
    # max over (coal=1, gas=1, wind=0 since wind missing from ab) → 30+80 = 110
    assert abs(sol.obj - 110.0) < 1e-6
