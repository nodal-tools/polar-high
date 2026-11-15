"""Scaling-check fields of :class:`RangeReport` (residue floor, integer
split, solver-infinite cut, extremes).

Every expected value below is worked out by hand from the model written in
the test (no value is copied from a run of the code):

* the *pool* is the finite non-zero bounds of the continuous columns plus
  the row bounds, minus values at or above ``infinite_bound`` (1e20);
* an entry is *residue* when its model-unit magnitude is below the floor
  ``10 × primal tolerance`` (1e-6 at HiGHS' default 1e-7);
* model units: a row bound is reported as ``rhs · row_factor``; a column
  bound is reported as written on the Var and its model value is
  ``bound · |col side vector|``.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from polar_high.autoscale import ScalingConfig, ranges_from_arrays
from polar_high.autoscale._ranges import _ranges_via_passmodel, _ranges_via_streaming
from polar_high.engine import Problem, Sum


def _cfg(**kw) -> ScalingConfig:
    return ScalingConfig(threshold_decades=9.0, **kw)


def _model() -> Problem:
    """Four rows, six columns.

    * ``x`` (2 cols) continuous, upper 1e3; ``y`` (2 cols) continuous, upper
      2e20 (no limit to HiGHS); ``z`` (2 cols) integer, upper 1e12.
    * ``c1[i]``: ``x + y + 1e-3 z <= 0.1``; ``c2[i]``: ``2 x >= 2e-15``.
    * objective ``x + 5 z``.
    """
    pb = Problem()
    idx = pl.DataFrame({"i": [0, 1]})
    x = pb.add_var("x", "i", idx, lower=0.0, upper=1e3)
    y = pb.add_var("y", "i", idx, lower=0.0, upper=2e20)
    z = pb.add_var("z", "i", idx, lower=0.0, upper=1e12, integer=True)
    pb.add_cstr(
        "c1",
        over=idx,
        sense="<=",
        lhs_terms={"x": x, "y": y, "z": 1e-3 * z},
        rhs_terms={"k": 0.1},
    )
    pb.add_cstr("c2", over=idx, sense=">=", lhs_terms={"x": 2.0 * x}, rhs_terms={"k": 2e-15})
    pb.set_objective(x + 5.0 * z, sense="min")
    return pb


def _new_fields(r) -> tuple:
    return (
        r.bound_rhs_meaningful,
        r.bound_rhs_residue,
        r.bound_rhs_beyond_infinite,
        r.integer_bound,
        r.matrix_continuous,
        r.integer_matrix,
        r.cost_continuous,
        r.integer_cost,
    )


def test_pools_without_side_vectors() -> None:
    pb = _model()
    s = _ranges_via_streaming(pb, _cfg())
    # Pool: x upper 1e3 (x2), c1 0.1 (x2), c2 2e-15 (x2, residue); y's 2e20 is
    # beyond 1e20; z is integer.
    assert s.bound_rhs_meaningful == (0.1, 1e3)
    assert s.bound_rhs_residue == (2, 2e-15)
    assert s.bound_rhs_beyond_infinite == 2
    assert s.integer_bound == (1e12, 1e12)
    # Matrix terms: x 1, y 1, x 2 continuous; z 1e-3 integer.
    assert s.matrix_continuous == (1.0, 2.0)
    assert s.integer_matrix == (1e-3, 1e-3)
    assert s.cost_continuous == (1.0, 1.0)
    assert s.integer_cost == (5.0, 5.0)
    assert s.residue_floor == pytest.approx(1e-6)
    assert s.infinite_bound == 1e20
    # The existing fields still include the integer bound and the 2e20.
    assert s.bound == (1e3, 2e20)

    p = _ranges_via_passmodel(pb, _cfg())
    assert _new_fields(p) == _new_fields(s)


def test_pools_with_side_vectors_streaming_equals_passmodel() -> None:
    """Row factors [2, 2, 4, 4] and a column side vector of 0.25 everywhere.

    * c1 rows report 0.1·2 = 0.2 (model 0.1); c2 rows 2e-15·4 = 8e-15 (model
      2e-15, residue).
    * x reports 1e3 (model 250); y 2e20 (beyond); z integer 1e12.
    * Matrix: c1 x, y: 1·2·0.25 = 0.5; z: 1e-3·0.5 = 5e-4; c2 x: 2·4·0.25 = 2.
    """
    pb = _model()
    pb._layer2_row_factor = np.array([2.0, 2.0, 4.0, 4.0])
    pb._layer2_col_factor = np.full(int(pb._next_col), 0.25)
    s = _ranges_via_streaming(pb, _cfg())
    assert s.bound_rhs_meaningful == (0.2, 1e3)
    assert s.bound_rhs_residue == (2, 2e-15)
    assert s.bound_rhs_beyond_infinite == 2
    assert s.integer_bound == (1e12, 1e12)
    assert s.matrix_continuous == (0.5, 2.0)
    assert s.integer_matrix == (5e-4, 5e-4)
    ext = s.extremes["bound_rhs"]
    assert ext["min"].family == "c1" and ext["min"].component == "rhs"
    assert ext["min"].post == 0.2 and ext["min"].model == pytest.approx(0.1)
    assert ext["max"].family == "x" and ext["max"].model == 250.0

    p = _ranges_via_passmodel(pb, _cfg())
    for got, want in zip(_new_fields(p)[:6], _new_fields(s)[:6], strict=True):
        assert got == want


def test_extremes_name_the_families() -> None:
    s = _ranges_via_streaming(_model(), _cfg())
    e = s.extremes
    assert (e["bound_rhs"]["min"].family, e["bound_rhs"]["min"].component) == ("c1", "rhs")
    assert (e["bound_rhs"]["max"].family, e["bound_rhs"]["max"].component) == ("x", "bound")
    assert e["residue"]["max"].family == "c2"
    assert e["residue"]["max"].model == 2e-15
    assert e["integer_bound"]["max"].family == "z"
    assert e["matrix"]["min"].family == "c1"
    assert e["matrix"]["max"].family == "c2"
    assert e["integer_matrix"]["min"].family == "c1"
    assert e["cost"]["min"].family == "x"
    assert e["integer_cost"]["max"].family == "z"


def test_residue_excluded_and_counted() -> None:
    """2e-15 among limits in [0.1, 1e3]: left out of the small end, counted."""
    pb = Problem()
    idx = pl.DataFrame({"i": [0, 1, 2]})
    x = pb.add_var("x", "i", idx, lower=0.0, upper=1e3)
    pb.add_cstr("lim", over=idx, sense="<=", lhs_terms={"x": x}, rhs_terms={"k": 0.1})
    pb.add_cstr("noise", over=None, sense="<=", lhs_terms={"x": Sum(x)}, rhs_terms={"k": 2e-15})
    pb.set_objective(Sum(x), sense="min")
    s = _ranges_via_streaming(pb, _cfg())
    assert s.rhs == (2e-15, 0.1)  # the existing field keeps it
    assert s.bound_rhs_meaningful == (0.1, 1e3)
    assert s.bound_rhs_residue == (1, 2e-15)


def test_floor_on_bounds_uses_the_inverse_col_factor() -> None:
    """Column side vector [0.0625, 16]: ``a`` reports 1e-5 (model 6.25e-7,
    below the 1e-6 floor: residue); ``b`` reports 1e-7 (model 1.6e-6:
    meaningful, and the smallest)."""
    pb = Problem()
    one = pl.DataFrame({"i": [0]})
    a = pb.add_var("a", "i", one, lower=0.0, upper=1e-5)
    b = pb.add_var("b", "i", one, lower=0.0, upper=1e-7)
    pb.add_cstr("c", over=one, sense="<=", lhs_terms={"a": a, "b": b}, rhs_terms={"k": 1.0})
    pb.set_objective(a + b, sense="min")
    pb._layer2_col_factor = np.array([0.0625, 16.0])
    pb._layer2_row_factor = np.array([1.0])
    s = _ranges_via_streaming(pb, _cfg())
    assert s.bound_rhs_meaningful == (1e-7, 1.0)
    assert s.bound_rhs_residue == (1, pytest.approx(6.25e-7))
    assert s.extremes["bound_rhs"]["min"].family == "b"
    assert s.extremes["residue"]["max"].family == "a"


def test_explicit_floor_and_tolerance() -> None:
    """``config.residue_floor`` wins; else 10 × the problem's tolerance."""
    pb = _model()
    s = _ranges_via_streaming(pb, _cfg(residue_floor=0.5))
    assert s.residue_floor == 0.5
    # 0.1 is now residue too: the smallest meaningful limit is x's 1e3.
    assert s.bound_rhs_meaningful == (1e3, 1e3)
    assert s.bound_rhs_residue == (4, 0.1)
    pb.set_solver_options({"primal_feasibility_tolerance": 1e-16})
    s = _ranges_via_streaming(pb, _cfg())
    assert s.residue_floor == pytest.approx(1e-15)
    assert s.bound_rhs_meaningful == (2e-15, 1e3)
    pb.set_solver_options({"infinite_bound": 1e25})
    s = _ranges_via_streaming(pb, _cfg())
    assert s.bound_rhs_beyond_infinite == 0
    assert s.bound_rhs_meaningful == (0.1, 2e20)


def test_ranges_from_arrays_new_fields() -> None:
    """The same model as :func:`_model` written out as arrays."""
    inf = math.inf
    col_lower = np.zeros(6)
    col_upper = np.array([1e3, 1e3, 2e20, 2e20, 1e12, 1e12])
    col_integer = np.array([False, False, False, False, True, True])
    # CSC by column: x0 (c1 r0: 1, c2 r2: 2), x1, y0, y1, z0, z1.
    values = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1e-3, 1e-3])
    col_index = np.array([0, 0, 1, 1, 2, 3, 4, 5])
    r = ranges_from_arrays(
        matrix_values=values,
        cost=np.array([1.0, 1.0, 0.0, 0.0, 5.0, 5.0]),
        col_lower=col_lower,
        col_upper=col_upper,
        row_lower=np.array([-inf, -inf, 2e-15, 2e-15]),
        row_upper=np.array([0.1, 0.1, inf, inf]),
        config=_cfg(),
        col_integer=col_integer,
        matrix_col_index=col_index,
    )
    assert r.bound_rhs_meaningful == (0.1, 1e3)
    assert r.bound_rhs_residue == (2, 2e-15)
    assert r.bound_rhs_beyond_infinite == 2
    assert r.integer_bound == (1e12, 1e12)
    assert r.matrix_continuous == (1.0, 2.0)
    assert r.integer_matrix == (1e-3, 1e-3)
    assert r.cost_continuous == (1.0, 1.0)
    assert r.integer_cost == (5.0, 5.0)
    assert r.extremes["bound_rhs"]["min"].family is None
