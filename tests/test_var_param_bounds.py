"""Per-element variable bounds: ``Problem.add_var(lower=Param, upper=Param)``.

Covers the bound-resolution semantics (subset-of-dims join, broadcast,
missing rows / null / NaN -> default, ±inf pass-through, error cases) and
that the per-column values reach EVERY consumer that materialises column
bounds: the streaming solve, the ``passModel`` solve, :class:`LpView`,
the in-house MPS writer, :class:`WarmProblem`, the autoscale range
detector (both pre-solve paths) and :meth:`Var.scale_bounds`.
"""

from __future__ import annotations

import math

import highspy
import numpy as np
import polars as pl
import pytest

import polar_high as fp
from polar_high.autoscale import ScalingConfig, detect_ranges
from polar_high.autoscale._ranges import _ranges_via_passmodel
from polar_high.solvers._lp_view import LpView

INF = float("inf")


def _p(dims, **cols) -> fp.Param:
    return fp.Param(tuple(dims), pl.DataFrame(cols))


def _vals(sol: fp.Solution, name: str, dims: tuple[str, ...]) -> dict:
    df = sol.value(name).sort(list(dims))
    keys = df.select(list(dims)).rows()
    return {k if len(k) > 1 else k[0]: v for k, v in zip(keys, df["value"].to_list())}


def _cap_problem(upper) -> tuple[fp.Problem, fp.Var]:
    """max sum x_i  s.t.  x_i <= 10  (as a constraint), ``upper`` as the
    variable bound.  Optimal x_i = min(upper_i, 10)."""
    pb = fp.Problem()
    idx = pl.DataFrame({"i": ["a", "b", "c", "d"]})
    x = pb.add_var("x", "i", idx, upper=upper)
    pb.add_cstr("cap", over=idx, sense="<=", lhs_terms={"x": x}, rhs_terms={"k": 10.0})
    pb.set_objective(-1.0 * x, sense="min")
    return pb, x


# ---------------------------------------------------------------------------
# Resolution semantics


def test_param_upper_tightens_subset_missing_rows_default():
    ub = _p(("i",), i=["a", "c"], value=[3.0, 5.0])
    pb, x = _cap_problem(ub)
    assert isinstance(x.upper, np.ndarray)
    # Elements b, d have no bound row -> default +inf.
    np.testing.assert_array_equal(x.upper, [3.0, INF, 5.0, INF])
    assert x.lower == 0.0 and not isinstance(x.lower, np.ndarray)
    sol = pb.solve()
    assert sol.optimal
    assert _vals(sol, "x", ("i",)) == pytest.approx({"a": 3.0, "b": 10.0, "c": 5.0, "d": 10.0})


def test_param_lower_missing_rows_default_zero():
    pb = fp.Problem()
    idx = pl.DataFrame({"i": [0, 1, 2]})
    x = pb.add_var("x", "i", idx, lower=_p(("i",), i=[1], value=[4.0]))
    pb.set_objective(1.0 * x, sense="min")
    np.testing.assert_array_equal(x.lower, [0.0, 4.0, 0.0])
    sol = pb.solve()
    assert sol.optimal
    assert _vals(sol, "x", ("i",)) == pytest.approx({0: 0.0, 1: 4.0, 2: 0.0})


def test_lower_dim_bound_broadcasts_over_var_dims():
    pb = fp.Problem()
    idx = pl.DataFrame({"g": ["u1", "u1", "u2", "u2"], "t": [0, 1, 0, 1]})
    # Bound keyed on g only -> applies to every t of that g.
    x = pb.add_var("x", ("g", "t"), idx, upper=_p(("g",), g=["u1", "u2"], value=[2.0, 7.0]))
    pb.set_objective(-1.0 * x, sense="min")
    np.testing.assert_array_equal(x.upper, [2.0, 2.0, 7.0, 7.0])
    sol = pb.solve()
    assert sol.optimal
    assert _vals(sol, "x", ("g", "t")) == pytest.approx(
        {("u1", 0): 2.0, ("u1", 1): 2.0, ("u2", 0): 7.0, ("u2", 1): 7.0}
    )


def test_bound_dims_in_different_order_than_var_dims():
    pb = fp.Problem()
    idx = pl.DataFrame({"g": ["a", "a", "b"], "t": [0, 1, 0]})
    ub = _p(("t", "g"), t=[1, 0], g=["a", "b"], value=[1.5, 2.5])
    x = pb.add_var("x", ("g", "t"), idx, upper=ub)
    np.testing.assert_array_equal(x.upper, [INF, 1.5, 2.5])


def test_null_and_nan_values_mean_default():
    pb = fp.Problem()
    idx = pl.DataFrame({"i": [0, 1, 2, 3]})
    ub = fp.Param(
        ("i",),
        pl.DataFrame({"i": [0, 1, 2], "value": [1.0, None, float("nan")]}),
    )
    lb = fp.Param(
        ("i",),
        pl.DataFrame({"i": [0, 1, 2], "value": [None, float("nan"), 0.5]}),
    )
    x = pb.add_var("x", "i", idx, lower=lb, upper=ub)
    np.testing.assert_array_equal(x.upper, [1.0, INF, INF, INF])
    np.testing.assert_array_equal(x.lower, [0.0, 0.0, 0.5, 0.0])


def test_lower_and_upper_params_together_with_infinities():
    """min sum x  s.t.  x_i >= -7 (constraint).  x_a free below
    (lower=-inf) -> -7; x_b in [2, 3] -> 2; x_c upper-only, default
    lower 0 -> 0."""
    pb = fp.Problem()
    idx = pl.DataFrame({"i": ["a", "b", "c"]})
    lb = _p(("i",), i=["a", "b"], value=[-INF, 2.0])
    ub = _p(("i",), i=["b", "c"], value=[3.0, INF])
    x = pb.add_var("x", "i", idx, lower=lb, upper=ub)
    pb.add_cstr("floor", over=idx, sense=">=", lhs_terms={"x": x}, rhs_terms={"k": -7.0})
    pb.set_objective(1.0 * x, sense="min")
    np.testing.assert_array_equal(x.lower, [-INF, 2.0, 0.0])
    np.testing.assert_array_equal(x.upper, [INF, 3.0, INF])
    sol = pb.solve()
    assert sol.optimal
    assert _vals(sol, "x", ("i",)) == pytest.approx({"a": -7.0, "b": 2.0, "c": 0.0})


def test_param_alignment_follows_index_row_order_not_param_order():
    pb = fp.Problem()
    idx = pl.DataFrame({"i": ["z", "a", "m"]})
    ub = _p(("i",), i=["a", "m", "z"], value=[1.0, 2.0, 3.0])
    x = pb.add_var("x", "i", idx, upper=ub)
    # frame rows (and col ids) follow the index order z, a, m.
    assert x.frame["i"].to_list() == ["z", "a", "m"]
    np.testing.assert_array_equal(x.upper, [3.0, 1.0, 2.0])


def test_enum_var_dim_joins_string_bound_dim():
    pb = fp.Problem()
    dt = pl.Enum(["n1", "n2", "n3"])
    idx = pl.DataFrame({"n": pl.Series(["n1", "n2", "n3"], dtype=dt)})
    x = pb.add_var("x", "n", idx, upper=_p(("n",), n=["n2"], value=[9.0]))
    np.testing.assert_array_equal(x.upper, [INF, 9.0, INF])


def test_zero_dim_param_collapses_to_scalar():
    pb = fp.Problem()
    idx = pl.DataFrame({"i": [0, 1]})
    x = pb.add_var("x", "i", idx, lower=fp.Param.scalar(-1.0), upper=fp.Param.scalar(2.0))
    assert x.lower == -1.0 and isinstance(x.lower, float)
    assert x.upper == 2.0 and isinstance(x.upper, float)
    assert not x.has_elementwise_bounds
    y = pb.add_var("y", "i", idx, upper=fp.Param.scalar(float("nan")))
    assert y.upper == INF


def test_empty_variable_with_param_bound():
    pb = fp.Problem()
    idx = pl.DataFrame({"i": pl.Series([], dtype=pl.Int64)})
    x = pb.add_var("x", "i", idx, upper=_p(("i",), i=[1], value=[2.0]))
    assert isinstance(x.upper, np.ndarray) and x.upper.size == 0


def test_bound_dims_not_in_var_raises():
    pb = fp.Problem()
    idx = pl.DataFrame({"i": [0, 1]})
    with pytest.raises(ValueError, match=r"dims \['j'\] that are not dims of the variable"):
        pb.add_var("x", "i", idx, upper=_p(("i", "j"), i=[0], j=[0], value=[1.0]))
    # The failed declaration must not have consumed column ids or names.
    assert "x" not in pb._vars


def test_duplicate_bound_rows_raise():
    pb = fp.Problem()
    idx = pl.DataFrame({"i": [0, 1]})
    with pytest.raises(ValueError, match="duplicate rows"):
        pb.add_var("x", "i", idx, lower=_p(("i",), i=[0, 0], value=[1.0, 2.0]))


def test_non_numeric_non_param_bound_raises():
    pb = fp.Problem()
    idx = pl.DataFrame({"i": [0, 1]})
    with pytest.raises(TypeError, match="must be a number or a Param"):
        pb.add_var("x", "i", idx, upper=pl.DataFrame({"i": [0], "value": [1.0]}))


def test_failed_declaration_keeps_col_ids():
    pb = fp.Problem()
    idx = pl.DataFrame({"i": [0, 1]})
    with pytest.raises(ValueError):
        pb.add_var("bad", "i", idx, upper=_p(("j",), j=[0], value=[1.0]))
    v = pb.add_var("ok", "i", idx)
    assert v.frame["col_id"].to_list() == [0, 1]


# ---------------------------------------------------------------------------
# Scalar path unchanged


def test_scalar_bounds_stored_verbatim():
    pb = fp.Problem()
    idx = pl.DataFrame({"i": [0, 1]})
    x = pb.add_var("x", "i", idx, lower=-3, upper=4.5)
    assert x.lower == -3 and type(x.lower) is int
    assert x.upper == 4.5 and type(x.upper) is float
    assert not x.has_elementwise_bounds
    assert x.col_lower() == -3.0 and isinstance(x.col_lower(), float)


def _uniform_pair(as_param: bool) -> fp.Problem:
    pb = fp.Problem()
    idx = pl.DataFrame({"i": [0, 1, 2]})
    if as_param:
        lo = _p(("i",), i=[0, 1, 2], value=[-1.0, -1.0, -1.0])
        hi = _p(("i",), i=[0, 1, 2], value=[6.0, 6.0, 6.0])
    else:
        lo, hi = -1.0, 6.0
    x = pb.add_var("x", "i", idx, lower=lo, upper=hi)
    y = pb.add_var("y", "i", idx)
    pb.add_cstr("c", over=idx, sense=">=", lhs_terms={"x": x, "y": y}, rhs_terms={"k": 4.0})
    pb.set_objective(2.0 * x + 3.0 * y, sense="min")
    return pb


def test_uniform_param_bounds_match_scalar_bounds(tmp_path):
    scal = _uniform_pair(False)
    arr = _uniform_pair(True)
    vs, va = LpView.from_problem(scal), LpView.from_problem(arr)
    np.testing.assert_array_equal(vs.col_lb, va.col_lb)
    np.testing.assert_array_equal(vs.col_ub, va.col_ub)
    ms, ma = scal.canonicalise(), arr.canonicalise()
    np.testing.assert_array_equal(ms.col_lb, ma.col_lb)
    np.testing.assert_array_equal(ms.col_ub, ma.col_ub)
    # MPS BOUNDS sections carry the same lines per column (scalar
    # families group per family, array families per column — the
    # column-major order is the same here, so the files are identical).
    scal.write_mps(tmp_path / "s.mps")
    arr.write_mps(tmp_path / "a.mps")
    assert (tmp_path / "s.mps").read_text() == (tmp_path / "a.mps").read_text()
    assert scal.solve().obj == pytest.approx(arr.solve().obj, abs=1e-12)


# ---------------------------------------------------------------------------
# Every bound consumer sees the per-column values


def _mixed_problem() -> tuple[fp.Problem, float, dict]:
    """Bounds of every MPS shape class in one family:
    a [0, inf) default, b [-inf, inf) free, c (-inf, 4], d [2, inf),
    e [1, 3], f [-5, 0]...  Objective pushes each element to a bound,
    linking constraint keeps the free ones finite."""
    pb = fp.Problem()
    keys = ["a", "b", "c", "d", "e", "f"]
    idx = pl.DataFrame({"i": keys})
    lb = _p(("i",), i=["b", "c", "d", "e", "f"], value=[-INF, -INF, 2.0, 1.0, -5.0])
    ub = _p(("i",), i=["b", "c", "e", "f"], value=[INF, 4.0, 3.0, 0.0])
    x = pb.add_var("x", "i", idx, lower=lb, upper=ub)
    # Keep b and c bounded: x_i >= -10 for all, x_i <= 20 for all.
    pb.add_cstr("lo", over=idx, sense=">=", lhs_terms={"x": x}, rhs_terms={"k": -10.0})
    pb.add_cstr("hi", over=idx, sense="<=", lhs_terms={"x": x}, rhs_terms={"k": 20.0})
    # Costs: minimise a,b,d,f ; maximise c,e.
    cost = _p(("i",), i=keys, value=[1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    pb.set_objective(cost * x, sense="min")
    expected = {"a": 0.0, "b": -10.0, "c": 4.0, "d": 2.0, "e": 3.0, "f": -5.0}
    obj = 0.0 + -10.0 - 4.0 + 2.0 - 3.0 - 5.0
    return pb, obj, expected


@pytest.mark.parametrize("streaming", [True, False])
def test_solve_paths(streaming):
    pb, obj, expected = _mixed_problem()
    sol = pb.solve(streaming=streaming)
    assert sol.optimal
    assert sol.obj == pytest.approx(obj)
    assert _vals(sol, "x", ("i",)) == pytest.approx(expected)


def test_lp_view_and_canonical_bounds():
    pb, _, _ = _mixed_problem()
    want_lb = [0.0, -INF, -INF, 2.0, 1.0, -5.0]
    want_ub = [INF, INF, 4.0, INF, 3.0, 0.0]
    view = LpView.from_problem(pb)
    np.testing.assert_array_equal(view.col_lb, want_lb)
    np.testing.assert_array_equal(view.col_ub, want_ub)
    m = pb.canonicalise()
    np.testing.assert_array_equal(m.col_lb, want_lb)
    np.testing.assert_array_equal(m.col_ub, want_ub)


def test_write_mps_per_column_bounds_round_trip(tmp_path):
    pb, obj, _ = _mixed_problem()
    path = tmp_path / "m.mps"
    pb.write_mps(path)
    text = path.read_text()
    bounds = text.split("BOUNDS\n", 1)[1].split("ENDATA", 1)[0]
    assert bounds.splitlines() == [
        " FR bnd  x[b]",
        " MI bnd  x[c]",
        " UP bnd  x[c]  4",
        " LO bnd  x[d]  2",
        " LO bnd  x[e]  1",
        " UP bnd  x[e]  3",
        " LO bnd  x[f]  -5",
        " UP bnd  x[f]  0",
    ]
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(str(path))
    h.run()
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    assert h.getInfo().objective_function_value == pytest.approx(obj)


def test_warm_problem_build_and_resolve():
    pb, obj, expected = _mixed_problem()
    wp = fp.WarmProblem(pb)
    sol = wp.solve()
    assert sol.optimal
    assert sol.obj == pytest.approx(obj)
    assert _vals(sol, "x", ("i",)) == pytest.approx(expected)
    ids = wp.col_id_of_var("x")
    lo, hi = wp.get_col_bounds(ids)
    np.testing.assert_array_equal(lo, [0.0, -INF, -INF, 2.0, 1.0, -5.0])
    np.testing.assert_array_equal(hi, [INF, INF, 4.0, INF, 3.0, 0.0])
    # Warm update on top of per-column bounds: pin e, re-solve.
    wp.fix_cols("x", [("e",)], np.array([1.5]))
    sol2 = wp.solve()
    assert sol2.optimal
    assert sol2.obj == pytest.approx(obj + 3.0 - 1.5)


def test_detect_ranges_sees_elementwise_bounds():
    pb, _, _ = _mixed_problem()
    cfg = ScalingConfig(threshold_decades=9.0, user_bound_scale=None, report_yaml_path=None)
    rep = detect_ranges(pb, cfg)  # streaming-aggregation path
    assert rep.bound == pytest.approx((1.0, 5.0))
    rep2 = _ranges_via_passmodel(pb, cfg)  # canonical-array path
    assert rep2.bound == pytest.approx((1.0, 5.0))


def test_scale_bounds_scalar_and_array():
    pb = fp.Problem()
    idx = pl.DataFrame({"i": [0, 1, 2]})
    s = pb.add_var("s", "i", idx, lower=-2.0, upper=INF)
    a = pb.add_var(
        "a",
        "i",
        idx,
        lower=_p(("i",), i=[0, 1], value=[-INF, 1.0]),
        upper=_p(("i",), i=[1, 2], value=[4.0, 8.0]),
    )
    original_upper = a.upper
    s.scale_bounds(0.5)
    a.scale_bounds(0.5)
    assert s.lower == -1.0 and s.upper == INF
    np.testing.assert_array_equal(a.lower, [-INF, 0.5, 0.0])
    np.testing.assert_array_equal(a.upper, [INF, 2.0, 4.0])
    # Scaling never mutates a caller-held array in place.
    np.testing.assert_array_equal(original_upper, [INF, 4.0, 8.0])
    # Per-column scaling is exactly the scalar arithmetic.
    for v_raw, v_scaled in zip([1.0, 4.0, 8.0], [a.lower[1], a.upper[1], a.upper[2]]):
        assert v_scaled == float(v_raw) * 0.5
    assert math.isinf(a.upper[0])


def test_solver_adapter_dispatch_via_lp_view():
    """``solvers.solve`` builds an :class:`LpView` and hands it to the
    adapter — the per-column bounds must reach the solver there too."""
    from polar_high.solvers import SolverStatus, solve

    pb, obj, _ = _mixed_problem()
    result = solve(pb, solver_name="highs")
    assert result.status == SolverStatus.OPTIMAL
    assert result.objective == pytest.approx(obj)
