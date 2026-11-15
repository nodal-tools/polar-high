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


def _scaled_lp() -> Problem:
    """:func:`_lp` with hand-set side vectors: rows × (4, 0.5, 2), columns ×
    0.25 (bounds left as written: they are the reported ones)."""
    pb = _lp()
    pb._layer2_row_factor = np.array([4.0, 0.5, 2.0])
    pb._layer2_col_factor = np.full(int(pb._next_col), 0.25)
    return pb


def test_integer_pools_reach_the_hook() -> None:
    """With the side vectors of :func:`_scaled_lp` (term = coef · row factor
    · col factor, cost = coef · col factor): x in dem 4·0.25 = 1, x in cap0
    0.5·0.25 = 0.125, z in cap0 0.125, z in tiny 2·0.25 = 0.5 → continuous
    matrix (0.125, 1); y (integer) in dem 4·0.25 = 1 → (1, 1); every cost
    0.25; y's bound 1e12 (its lower 0 is no limit)."""
    seen = {}
    _scaled_lp().solve(pre_run_check=lambda r, h: seen.update(r))
    assert seen["matrix_continuous"] == (0.125, 1.0)
    assert seen["integer_matrix"] == (1.0, 1.0)
    assert seen["cost_continuous"] == (0.25, 0.25)
    assert seen["integer_cost"] == (0.25, 0.25)
    assert seen["integer_bound"] == (1e12, 1e12)


def test_pure_lp_has_no_integer_pools() -> None:
    pb = Problem()
    i = pl.DataFrame({"i": [0, 1]})
    x = pb.add_var("x", "i", i, lower=0.0, upper=4.0)
    pb.add_cstr("c", over=None, sense=">=", lhs_terms={"x": Sum(x)}, rhs_terms={"r": 1.0})
    pb.set_objective(Sum(x), sense="min")
    seen = {}
    pb.solve(pre_run_check=lambda r, h: seen.update(r))
    assert seen["integer_matrix"] is None and seen["integer_cost"] is None
    assert seen["integer_bound"] is None
    assert seen["matrix_continuous"] == (1.0, 1.0)


def test_warm_problem_first_build_gets_the_streaming_ranges() -> None:
    """The WarmProblem build reads the same ranges from its canonical
    matrix as the streaming build: every key equal."""
    streamed = {}
    _scaled_lp().solve(pre_run_check=lambda r, h: streamed.update(r))
    warm = {}
    WarmProblem(_scaled_lp()).solve(pre_run_check=lambda r, h: warm.update(r))
    assert warm == streamed
    # Hand check of the pool (as in the walk test above).
    assert warm["row_bound_meaningful"] == (2.0, 5e4)
    assert warm["col_bound_meaningful"] == (1e3, 1e3)
    assert warm["bound_rhs_residue"] == (1, 3e-15)


def test_warm_problem_hook_once_on_the_first_build() -> None:
    wp = WarmProblem(_lp())
    calls = []

    def hook(ranges, h):
        calls.append(str(h.getModelStatus()))
        return {"user_bound_scale": -3}

    sol = wp.solve(pre_run_check=hook)
    assert calls == ["HighsModelStatus.kNotset"]
    assert sol.optimal
    assert sol.solve_diagnostics().user_bound_scale == -3
    # A re-solve of the built instance does not call it.
    wp.solve(pre_run_check=hook)
    assert len(calls) == 1


def test_warm_problem_hook_can_raise_before_run() -> None:
    class Stop(Exception):
        pass

    def hook(ranges, h):
        raise Stop("too wide")

    wp = WarmProblem(_lp())
    with pytest.raises(Stop):
        wp.solve(pre_run_check=hook)
    import highspy

    assert wp._h.getModelStatus() == highspy.HighsModelStatus.kNotset


def test_names_decision_follows_the_hook(tmp_path) -> None:
    """An option the hook sets that writes a file with names gets them
    under ``highs_names="auto"`` (streaming and warm)."""

    def hook(ranges, h):
        return {"write_basis_file": str(tmp_path / "b.bas")}

    sol = _lp().solve(highs_names="auto", keep_solver=True, pre_run_check=hook)
    assert sol.highs_names_passed
    sol = WarmProblem(_lp(), highs_names="auto").solve(pre_run_check=hook)
    assert sol.highs_names_passed
    # Without the hook nothing needs names.
    assert not _lp().solve(highs_names="auto", keep_solver=True).highs_names_passed


def test_build_only_runs_the_hook_and_returns_its_options(tmp_path) -> None:
    seen = {}

    def hook(ranges, h):
        seen.update(ranges)
        h.setOptionValue("user_objective_scale", 2)
        return {"user_bound_scale": -4}

    mps = tmp_path / "m.mps"
    out = _lp().build_only(str(mps), pre_run_check=hook)
    assert out == {"user_objective_scale": 2, "user_bound_scale": -4}
    assert mps.exists()
    assert seen["row_bound_meaningful"] == (0.5, 1e5)
    # Without a hook: nothing to carry.
    assert _lp().build_only(str(tmp_path / "n.mps")) == {}


def test_build_only_hook_raise_writes_no_file(tmp_path) -> None:
    def hook(ranges, h):
        raise RuntimeError("too wide")

    mps = tmp_path / "m.mps"
    with pytest.raises(RuntimeError, match="too wide"):
        _lp().build_only(str(mps), pre_run_check=hook)
    assert not mps.exists()


def test_residue_floor_uses_the_primal_tolerance_not_kkt() -> None:
    """kkt_tolerance 1e-5 does not move the floor: 10 × the primal 1e-7."""
    pb = _lp()
    pb.set_solver_options({"kkt_tolerance": 1e-5})
    seen = {}
    pb.solve(pre_run_check=lambda r, h: seen.update(r))
    assert seen["residue_floor"] == pytest.approx(1e-6)
