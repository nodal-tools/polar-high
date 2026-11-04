"""WarmProblem ``update_*`` under Layer-2 side vectors.

The build multiplies every RHS by ``_layer2_row_factor[row]`` and every
objective coefficient by ``_layer2_col_factor[col]`` (which stores
``1 / cf_math``).  A warm update in the default ``frame="user"`` must do
the same, so that updating and re-solving gives the optimum of a fresh
build with the new values.  ``frame="scaled"`` writes values as given.
The small-value floor (``coef_zero_threshold``) is judged on the
user-space magnitude, as at build, for both the RHS path and the
``update_param`` matrix-cell path.

Side vectors are installed by hand with a distinct power-of-two factor
per row and per column (as in ``test_layer2_side_vector_emit.py``), so a
missed or mis-indexed multiply changes the optimum visibly.  Column
bounds are ``[0, inf)``, which the side vectors leave invariant, so the
scaled LP has the same optimal objective as the unscaled one.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polar_high as fp

N_T = 3
# Rows: balance[0..2] then cap[0..2].  Columns: x[0..2] then y[0..2].
ROW_FACTOR = np.array([2.0, 0.25, 8.0, 0.5, 4.0, 0.125])
COL_FACTOR = np.array([0.5, 4.0, 0.125, 2.0, 0.25, 8.0])


def _p(values, t=None) -> fp.Param:
    t = np.arange(N_T, dtype=np.int64) if t is None else np.asarray(t, dtype=np.int64)
    return fp.Param(("t",), pl.DataFrame({"t": t, "value": np.asarray(values, dtype=np.float64)}))


def _build(
    demand,
    cap,
    cost_x,
    cost_y,
    *,
    side_vectors: bool = True,
    coef_name: str | None = None,
    coef: float = 2.0,
) -> fp.Problem:
    """``coef·x[t] + y[t] == demand[t]``, ``x[t] <= cap[t]``,
    ``min Σ cost_x·x + cost_y·y``; unique optimum ``x = min(cap, demand/coef)``."""
    p = fp.Problem()
    t_idx = pl.DataFrame({"t": np.arange(N_T, dtype=np.int64)})
    x = p.add_var("x", "t", t_idx, lower=0.0)
    y = p.add_var("y", "t", t_idx, lower=0.0)
    coef_p = fp.Param(
        ("t",),
        pl.DataFrame({"t": np.arange(N_T, dtype=np.int64), "value": np.full(N_T, coef)}),
        name=coef_name,
    )
    p.add_cstr(
        "balance",
        over=t_idx,
        sense="==",
        lhs_terms={"x": coef_p * x, "y": y.to_expr()},
        rhs_terms={"demand": _p(demand)},
    )
    p.add_cstr(
        "cap",
        over=t_idx,
        sense="<=",
        lhs_terms={"x": x.to_expr()},
        rhs_terms={"cap": _p(cap)},
    )
    p.set_objective(_p(cost_x) * x + _p(cost_y) * y, sense="min")
    if side_vectors:
        p._layer2_row_factor = ROW_FACTOR.copy()
        p._layer2_col_factor = COL_FACTOR.copy()
    return p


BASE = dict(
    demand=[10.0, 20.0, 30.0],
    cap=[4.0, 6.0, 8.0],
    cost_x=[1.0, 2.0, 3.0],
    cost_y=[100.0, 90.0, 80.0],
)
NEW = dict(
    demand=[14.0, 4.0, 36.0],
    cap=[5.0, 1.0, 20.0],
    cost_x=[2.0, 1.0, 5.0],
    cost_y=[70.0, 60.0, 50.0],
)


def _fresh(**kw) -> fp.Solution:
    return fp.WarmProblem(_build(**kw)).solve()


def _assert_same_optimum(warm: fp.Solution, fresh: fp.Solution) -> None:
    assert warm.optimal and fresh.optimal
    assert warm.obj == pytest.approx(fresh.obj, rel=1e-12)
    np.testing.assert_allclose(warm.col_value, fresh.col_value, rtol=1e-12, atol=1e-12)


class _Spy:
    """Wrap a HiGHS handle, recording row-bound and column-cost writes."""

    def __init__(self, h):
        self._h = h
        self.bounds: list[tuple[np.ndarray, np.ndarray]] = []
        self.costs: list[np.ndarray] = []
        self.coeffs: list[float] = []
        self.cells: dict[tuple[int, int], float] = {}

    def changeRowsBounds(self, n, idx, lb, ub):  # noqa: N802 (HiGHS API)
        self.bounds.append((np.array(lb, dtype=np.float64), np.array(ub, dtype=np.float64)))
        return self._h.changeRowsBounds(n, idx, lb, ub)

    def changeColsCost(self, n, idx, cost):  # noqa: N802
        self.costs.append(np.array(cost, dtype=np.float64))
        return self._h.changeColsCost(n, idx, cost)

    def changeCoeff(self, r, c, v):  # noqa: N802
        self.coeffs.append(float(v))
        self.cells[(int(r), int(c))] = float(v)
        return self._h.changeCoeff(r, c, v)

    def __getattr__(self, name):
        return getattr(self._h, name)


def test_side_vectors_leave_the_optimum_unchanged():
    """Sanity: the fixture's side vectors are a pure change of variables."""
    scaled = _fresh(**BASE)
    plain = fp.WarmProblem(_build(**BASE, side_vectors=False)).solve()
    assert scaled.obj == pytest.approx(plain.obj, rel=1e-12)


def test_user_frame_updates_equal_a_fresh_build():
    wp = fp.WarmProblem(_build(**BASE))
    wp.solve()
    wp.update_rhs("balance", _p(NEW["demand"]))
    wp.update_rhs("cap", np.asarray(NEW["cap"]))
    wp.update_obj_coef("x", _p(NEW["cost_x"]))
    wp.update_obj_coef_array("y", [(t,) for t in range(N_T)], np.asarray(NEW["cost_y"]))
    _assert_same_optimum(wp.solve(), _fresh(**NEW))


def test_user_frame_param_join_by_label():
    """A Param in another row order (label join, not the in-place hot path)
    and a Param that covers only some labels (missing -> 0)."""
    wp = fp.WarmProblem(_build(**BASE))
    wp.solve()
    order = [2, 0, 1]
    wp.update_rhs("balance", _p(np.asarray(NEW["demand"])[order], t=order))
    wp.update_obj_coef("x", _p(np.asarray(NEW["cost_x"])[order], t=order))
    _assert_same_optimum(
        wp.solve(), _fresh(**{**BASE, "demand": NEW["demand"], "cost_x": NEW["cost_x"]})
    )

    wp.update_rhs("cap", _p([5.0, 7.0], t=[2, 0]))
    expect = {**BASE, "demand": NEW["demand"], "cost_x": NEW["cost_x"], "cap": [7.0, 0.0, 5.0]}
    _assert_same_optimum(wp.solve(), _fresh(**expect))


def test_scaled_frame_writes_values_verbatim():
    wp = fp.WarmProblem(_build(**BASE))
    wp.solve()
    spy = _Spy(wp._h)
    wp._h = spy
    vals = np.array([3.0, 5.0, 7.0])
    wp.update_rhs("cap", vals, frame="scaled")
    wp.update_obj_coef("x", _p(vals), frame="scaled")
    wp.update_obj_coef_array("y", [(t,) for t in range(N_T)], vals, frame="scaled")
    lb, ub = spy.bounds[-1]
    np.testing.assert_array_equal(ub, vals)
    assert np.all(np.isneginf(lb))
    np.testing.assert_array_equal(spy.costs[0], vals)
    np.testing.assert_array_equal(spy.costs[1], vals)


def test_user_frame_applies_the_side_vectors():
    wp = fp.WarmProblem(_build(**BASE))
    wp.solve()
    spy = _Spy(wp._h)
    wp._h = spy
    vals = np.array([3.0, 5.0, 7.0])
    wp.update_rhs("cap", vals)
    wp.update_obj_coef("x", _p(vals))
    wp.update_obj_coef_array("y", [(t,) for t in range(N_T)], vals)
    np.testing.assert_array_equal(spy.bounds[-1][1], vals * ROW_FACTOR[3:6])
    np.testing.assert_array_equal(spy.costs[0], vals * COL_FACTOR[0:3])
    np.testing.assert_array_equal(spy.costs[1], vals * COL_FACTOR[3:6])


def test_inf_sentinels_survive_a_non_unit_row_factor():
    wp = fp.WarmProblem(_build(**BASE))
    wp.solve()
    spy = _Spy(wp._h)
    wp._h = spy
    wp.update_rhs("cap", np.array([np.inf, 2.0, np.inf]))
    lb, ub = spy.bounds[-1]
    assert np.all(np.isneginf(lb))
    assert ub[0] == np.inf and ub[2] == np.inf
    assert ub[1] == 2.0 * ROW_FACTOR[4]
    # Unbounded cap rows: the optimum is a fresh build with an infinite cap there.
    _assert_same_optimum(wp.solve(), _fresh(**{**BASE, "cap": [np.inf, 2.0, np.inf]}))


@pytest.mark.parametrize("frame", ["user", "scaled"])
def test_no_side_vectors_push_is_unchanged(frame):
    """Without side vectors both frames push exactly the resolved values."""
    wp = fp.WarmProblem(_build(**BASE, side_vectors=False))
    wp.solve()
    spy = _Spy(wp._h)
    wp._h = spy
    d = np.array([1.5, 2.5, 3.5])
    wp.update_rhs("balance", _p(d), frame=frame)
    wp.update_obj_coef("x", _p(d), frame=frame)
    wp.update_obj_coef_array("y", [(t,) for t in range(N_T)], d, frame=frame)
    lb, ub = spy.bounds[-1]
    np.testing.assert_array_equal(lb, d)
    np.testing.assert_array_equal(ub, d)
    np.testing.assert_array_equal(spy.costs[0], d)
    np.testing.assert_array_equal(spy.costs[1], d)


def test_bad_frame_is_rejected():
    wp = fp.WarmProblem(_build(**BASE))
    wp.solve()
    with pytest.raises(ValueError, match="frame"):
        wp.update_rhs("cap", 1.0, frame="physical")
    with pytest.raises(ValueError, match="frame"):
        wp.update_obj_coef("x", 1.0, frame="physical")
    with pytest.raises(ValueError, match="frame"):
        wp.update_obj_coef_array("x", [(0,)], np.array([1.0]), frame="physical")


def _single_row_problem(rf: float, cf: float, *, coef_name: str | None = None) -> fp.Problem:
    """``coef·v[t] == 10`` over 3 rows; every row has factor ``rf``, every column ``cf``."""
    p = fp.Problem()
    t_idx = pl.DataFrame({"t": np.arange(N_T, dtype=np.int64)})
    v = p.add_var("v", "t", t_idx, lower=0.0, upper=1.0e6)
    coef_p = fp.Param(
        ("t",),
        pl.DataFrame({"t": np.arange(N_T, dtype=np.int64), "value": np.ones(N_T)}),
        name=coef_name,
    )
    p.add_cstr(
        "balance",
        over=t_idx,
        sense="==",
        lhs_terms={"v": coef_p * v},
        rhs_terms={"demand": _p(np.full(N_T, 10.0))},
    )
    p.set_objective(_p(np.ones(N_T)) * v, sense="min")
    p._layer2_row_factor = np.full(N_T, rf)
    p._layer2_col_factor = np.full(N_T, cf)
    p.coef_zero_threshold = 1e-4
    return p


@pytest.mark.parametrize("frame", ["user", "scaled"])
def test_rhs_floor_is_judged_in_user_space(frame):
    """rf = 2**-14: a user RHS of 1 (scaled 6.1e-5 < 1e-4) is kept; a user RHS
    of 1e-8 is floored.  In the scaled frame the bound in HiGHS is the same
    scaled value, so the decision is the same."""
    rf = 2.0**-14
    wp = fp.WarmProblem(_single_row_problem(rf, 1.0))
    wp.solve()
    spy = _Spy(wp._h)
    wp._h = spy
    scale = rf if frame == "scaled" else 1.0
    wp.update_rhs("balance", np.array([1.0, 1e-8, 1.0]) * scale, frame=frame)
    lb, ub = spy.bounds[-1]
    np.testing.assert_array_equal(lb, [rf, 0.0, rf])
    np.testing.assert_array_equal(ub, [rf, 0.0, rf])


def test_update_param_floor_is_judged_in_user_space():
    """rf·cf = 2**-14: a tracked user coefficient of 1 (scaled 6.1e-5) is kept,
    a user coefficient of 1e-8 is floored.  Checked per cell (row t, column
    t): update_param writes the cells in the order the build's join produced
    them, which polars does not guarantee."""
    rf, cf = 2.0**-7, 2.0**-7
    wp = fp.WarmProblem(_single_row_problem(rf, cf, coef_name="p_coef"))
    wp.declare_mutable("p_coef")
    wp.solve()
    spy = _Spy(wp._h)
    wp._h = spy
    wp.update_param("p_coef", _p([1.0, 1e-8, 1.0]))
    assert spy.cells == {(0, 0): rf * cf, (1, 1): 0.0, (2, 2): rf * cf}
