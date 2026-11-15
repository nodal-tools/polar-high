"""``Problem.solve(pre_run_check=...)`` and the floored bound minima it gets.

Expected values are hand-derived from the LP written in each test.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polar_high.autoscale import ScalingConfig
from polar_high.autoscale._ranges import _ranges_via_streaming
from polar_high.engine import Problem, Sum, WarmProblem


def _lp() -> Problem:
    """``x[i] <= ub`` with ub 1e3 / 2e20 (no limit) per column; ``y`` integer
    (left out: continuous columns only); rows ``x0 + x1 >= 0.5``,
    ``x0 <= 1e5`` and ``x1 <= 3e-15`` (residue at the default floor 1e-6)."""
    pb = Problem()
    i = pl.DataFrame({"i": [0, 1]})
    one = pl.DataFrame({"k": [0]})
    x = pb.add_var("x", "i", i, lower=0.0, upper=1e3)
    y = pb.add_var("y", "k", one, lower=0.0, upper=1e12, integer=True)
    z = pb.add_var("z", "k", one, lower=0.0, upper=2e20)
    pb.add_cstr(
        "dem", over=None, sense=">=", lhs_terms={"x": Sum(x), "y": Sum(y)}, rhs_terms={"r": 0.5}
    )
    pb.add_cstr(
        "cap0", over=None, sense="<=", lhs_terms={"x": Sum(x), "z": Sum(z)}, rhs_terms={"r": 1e5}
    )
    pb.add_cstr("tiny", over=None, sense="<=", lhs_terms={"z": Sum(z)}, rhs_terms={"r": 3e-15})
    pb.set_objective(Sum(x) + Sum(y) + Sum(z), sense="min")
    return pb


def test_hook_receives_floored_minima_once_before_run() -> None:
    pb = _lp()
    calls = []

    def hook(ranges, h):
        # Not solved yet: HiGHS has no model status beyond "not set".
        calls.append((dict(ranges), str(h.getModelStatus())))
        return None

    sol = pb.solve(pre_run_check=hook, keep_solver=True)
    assert len(calls) == 1
    ranges, status = calls[0]
    assert status == "HighsModelStatus.kNotset"
    # Continuous columns: x upper 1e3 (x2), z upper 2e20 (beyond 1e20).
    assert ranges["col_bound_meaningful"] == (1e3, 1e3)
    # Rows: 0.5, 1e5, 3e-15 (residue).
    assert ranges["row_bound_meaningful"] == (0.5, 1e5)
    assert ranges["bound_rhs_residue"] == (1, 3e-15)
    assert ranges["bound_rhs_beyond_infinite"] == 1
    assert ranges["residue_floor"] == pytest.approx(1e-6)
    assert ranges["infinite_bound"] == 1e20
    assert sol.optimal
    # The existing keys are unchanged.
    assert ranges["row_bound"] == sol.streamed_lp_ranges["row_bound"] == (3e-15, 1e5)


def test_floored_minima_agree_with_the_range_walk() -> None:
    """Same Problem, with side vectors: the streamed pool equals the range
    walk's pool (b_lo*, b_hi)."""
    pb = _lp()
    pb._layer2_row_factor = np.array([4.0, 0.5, 2.0])
    pb._layer2_col_factor = np.full(int(pb._next_col), 0.25)
    walk = _ranges_via_streaming(pb, ScalingConfig())
    seen = {}

    def hook(ranges, h):
        seen.update(ranges)

    pb.solve(pre_run_check=hook)
    lo = min(seen["col_bound_meaningful"][0], seen["row_bound_meaningful"][0])
    hi = max(seen["col_bound_meaningful"][1], seen["row_bound_meaningful"][1])
    assert (lo, hi) == walk.bound_rhs_meaningful
    assert seen["bound_rhs_residue"] == walk.bound_rhs_residue
    assert seen["bound_rhs_beyond_infinite"] == walk.bound_rhs_beyond_infinite
    # Hand check: rows 0.5·4 = 2, 1e5·0.5 = 5e4, 3e-15·2 (residue, model
    # 3e-15); x reports 1e3 (model 250).
    assert (lo, hi) == (2.0, 5e4)


def test_hook_can_set_user_bound_scale() -> None:
    pb = _lp()

    def set_on_handle(ranges, h):
        h.setOptionValue("user_bound_scale", -3)

    sol = pb.solve(pre_run_check=set_on_handle, keep_solver=True)
    assert sol.solve_diagnostics().user_bound_scale == -3

    sol = _lp().solve(pre_run_check=lambda r, h: {"user_bound_scale": 2}, keep_solver=True)
    assert sol.solve_diagnostics().user_bound_scale == 2


def test_hook_can_raise_and_no_solve_runs() -> None:
    class Stop(Exception):
        pass

    def hook(ranges, h):
        raise Stop("too wide")

    pb = _lp()
    with pytest.raises(Stop):
        pb.solve(pre_run_check=hook)


def test_hook_with_save_memory_sees_the_fresh_handle() -> None:
    pb = _lp()
    sol = pb.solve(
        save_memory=True,
        keep_solver=True,
        pre_run_check=lambda r, h: {"user_bound_scale": -1},
    )
    assert sol.solve_diagnostics().user_bound_scale == -1


def test_hook_rejected_on_non_streaming_path() -> None:
    with pytest.raises(ValueError, match="streaming"):
        _lp().solve(streaming=False, pre_run_check=lambda r, h: None)


def test_warm_problem_solve_takes_no_hook() -> None:
    import inspect

    assert "pre_run_check" not in inspect.signature(WarmProblem.solve).parameters
