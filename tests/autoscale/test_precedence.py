"""Precedence-check tests.

When the caller has already set ``user_bound_scale`` or
``user_objective_scale`` on the ``Problem`` — via
:meth:`set_solver_options`, :meth:`set_solver_option`, or by loading a
``highs.opt`` file into ``_solver_options`` — the autoscaler must
respect that value and skip its own recommendation *for that axis only*.
The other axis still auto-recommends.
"""

from __future__ import annotations

import math

import polars as pl

from polar_high.autoscale import (
    RangeReport,
    ScalingConfig,
    apply_scaling,
    get_explicit_option,
    has_explicit_option,
    recommend_scaling,
)
from polar_high.engine import Problem


def _cfg() -> ScalingConfig:
    return ScalingConfig(threshold_decades=9.0)


def _ranges(
    *,
    cost=(math.nan, math.nan),
    bound=(math.nan, math.nan),
    rhs=(math.nan, math.nan),
) -> RangeReport:
    return RangeReport(
        matrix=(1e-2, 1e2),
        cost=cost,
        bound=bound,
        rhs=rhs,
        cross_group_max_ratio=math.nan,
        trigger=True,
    )


def _make_problem() -> Problem:
    """Tiny LP that the autoscaler can attach options to."""
    pb = Problem()
    idx = pl.DataFrame({"i": [0]})
    pb.add_var("x", "i", idx, lower=0.0, upper=1.0)
    return pb


def test_get_and_has_explicit_option_via_method() -> None:
    pb = _make_problem()
    assert get_explicit_option(pb, "user_bound_scale") is None
    assert has_explicit_option(pb, "user_bound_scale") is False

    pb.set_solver_option("user_bound_scale", -3)
    assert get_explicit_option(pb, "user_bound_scale") == -3
    assert has_explicit_option(pb, "user_bound_scale") is True


def test_get_explicit_option_via_dict_fallback() -> None:
    """Duck-typed problem with ``_solver_options`` dict still works."""

    class FakeProblem:
        def __init__(self):
            self._solver_options = {"user_bound_scale": -5}

    fp = FakeProblem()
    assert get_explicit_option(fp, "user_bound_scale") == -5
    assert has_explicit_option(fp, "user_bound_scale") is True


def test_precedence_bound_skips_recommendation() -> None:
    """Caller-set ``user_bound_scale=-5`` → recommendation skipped for bound axis."""
    pb = _make_problem()
    pb.set_solver_option("user_bound_scale", -5)

    # A range report where the auto-recommendation would normally pick
    # something non-zero (large bound overshoot).
    r = _ranges(cost=(1.0, 1e7), bound=(1e1, 1e8), rhs=(1e1, 1e8))
    plan = recommend_scaling(r, _cfg(), problem=pb)

    assert plan.bound_skipped_external is True
    assert plan.user_bound_scale == -5  # preserved from caller
    # Objective axis still auto-runs (centres the 8-decade cost band).
    assert plan.objective_skipped_external is False
    assert plan.user_objective_scale == -12
    assert "external user_bound_scale=-5" in plan.reasoning


def test_precedence_objective_skips_recommendation() -> None:
    """Caller-set ``user_objective_scale=7`` → skipped for objective axis."""
    pb = _make_problem()
    pb.set_solver_option("user_objective_scale", 7)

    r = _ranges(cost=(1.0, 1e7), bound=(1e1, 1e8), rhs=(1e1, 1e8))
    plan = recommend_scaling(r, _cfg(), problem=pb)

    assert plan.objective_skipped_external is True
    assert plan.user_objective_scale == 7
    assert plan.bound_skipped_external is False
    # Bound axis still auto-runs.
    assert plan.user_bound_scale == -7
    assert "external user_objective_scale=7" in plan.reasoning


def test_precedence_both_axes_skipped() -> None:
    """Both axes caller-set → both skipped, no auto recommendations."""
    pb = _make_problem()
    pb.set_solver_options({"user_bound_scale": -3, "user_objective_scale": 2})

    r = _ranges(cost=(1.0, 1e7), bound=(1e1, 1e8), rhs=(1e1, 1e8))
    plan = recommend_scaling(r, _cfg(), problem=pb)

    assert plan.objective_skipped_external is True
    assert plan.bound_skipped_external is True
    assert plan.user_objective_scale == 2
    assert plan.user_bound_scale == -3


def test_apply_scaling_does_not_overwrite_external_options() -> None:
    """``apply_scaling`` must leave caller-set option values intact."""
    pb = _make_problem()
    pb.set_solver_option("user_bound_scale", -5)

    r = _ranges(cost=(1.0, 1e7), bound=(1e1, 1e8), rhs=(1e1, 1e8))
    plan = recommend_scaling(r, _cfg(), problem=pb)
    apply_scaling(pb, plan)

    # Caller's -5 must still be there; autoscaler must NOT overwrite.
    assert pb.get_solver_option("user_bound_scale") == -5
    # Objective auto picked -12 (centres the cost band), applied normally.
    assert pb.get_solver_option("user_objective_scale") == -12
    # Simplex strategy always set.
    assert pb.get_solver_option("simplex_scale_strategy") == 2


def test_recommend_without_problem_argument_uses_config_only() -> None:
    """When ``problem=None`` is passed, precedence-check path is bypassed."""
    r = _ranges(cost=(1.0, 1e7), bound=(1e1, 1e8), rhs=(1e1, 1e8))
    plan = recommend_scaling(r, _cfg())
    assert plan.objective_skipped_external is False
    assert plan.bound_skipped_external is False
    assert plan.user_objective_scale == -12
    assert plan.user_bound_scale == -7
