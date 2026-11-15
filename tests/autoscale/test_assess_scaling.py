"""Layer 3 tolerance window and :func:`assess_scaling`.

Expected values are worked out by hand in each docstring (``log2`` /
``2**N`` arithmetic on the numbers written in the test), not copied from a
run of the code.  Window at tolerance ``tol``: hard edges ``[10·tol,
1e18·tol]``, preferred small end ``100·tol``.
"""

from __future__ import annotations

import math

import pytest

from polar_high.autoscale import (
    Layer3Plan,
    RangeReport,
    ScalingConfig,
    assess_scaling,
    bound_scale_window,
    recommend_scaling,
)

NAN = (math.nan, math.nan)


def _r(lo: float, hi: float, **kw) -> RangeReport:
    """A report whose bound / RHS pool is ``[lo, hi]``."""
    kw.setdefault("bound_rhs_meaningful", (lo, hi))
    kw.setdefault("bound_rhs_residue", (0, 0.0))
    kw.setdefault("bound_rhs_beyond_infinite", 0)
    return RangeReport(
        matrix=kw.pop("matrix", (1e-2, 1e2)),
        cost=kw.pop("cost", (1.0, 1e2)),
        bound=kw.pop("bound", (lo, hi)),
        rhs=kw.pop("rhs", (lo, hi)),
        cross_group_max_ratio=math.nan,
        trigger=True,
        **kw,
    )


def _plan(n_bnd: int, n_obj: int = 0, source: str = "manual") -> Layer3Plan:
    return Layer3Plan(
        user_objective_scale=n_obj,
        user_bound_scale=n_bnd,
        simplex_scale_strategy=2,
        reasoning="test",
        bound_source=source,
    )


def _assess(ranges, plan, tol=1e-7, **kw):
    return assess_scaling(ranges, plan, primal_tol=tol, dual_tol=tol, **kw)


def test_incident_shape_is_unscalable_error() -> None:
    """0.5 … 1e18 at tol 1e-7: N_min = ceil(log2(1e-6/0.5)) = -18,
    N_max = floor(log2(1e11/1e18)) = -24 → no exponent.  The tier clamp is
    floor(log2(1e6/1e18)) = -40 → -30 (HiGHS' range), tag unscalable.
    Fitting the largest value (N = -24) allows 1e-7·2**24 = 1.6777216 in
    model units, 3.3554432 × the smallest limit."""
    r = _r(0.5, 1e18)
    plan = recommend_scaling(r, ScalingConfig())
    assert plan.user_bound_scale == -30
    assert plan.bound_tag == "unscalable"
    assert not plan.bound_window.fits
    v = _assess(r, plan)
    assert v.level == "error"
    f = v.finding("bound_rhs")
    assert f.kind == "spread"
    assert (f.numbers["n_min"], f.numbers["n_max"]) == (-18.0, -24.0)
    assert f.numbers["decades"] == pytest.approx(18.30103, abs=1e-5)
    assert f.numbers["n_fit_large"] == -24.0
    assert f.numbers["fit_large_violation_model"] == pytest.approx(1.6777216)
    assert f.numbers["fit_large_violation_ratio"] == pytest.approx(3.3554432)


def test_wide_but_scalable_warns_inside_window() -> None:
    """1.6e-5 … 1.3e9: D* = log10(8.125e13) = 13.91 > 13 → warn.  Tier:
    floor(log2(1e6/1.3e9)) = -11, smallest-side; window: preferred
    ceil(log2(1e-5/1.6e-5)) = 0, large floor(log2(1e11/1.3e9)) = 6 → [0, 6],
    so N = 0 (raised)."""
    r = _r(1.6e-5, 1.3e9)
    plan = recommend_scaling(r, ScalingConfig())
    assert plan.user_bound_scale == 0
    assert plan.bound_tag == "smallest-side+raised"
    v = _assess(r, plan)
    assert v.level == "warn"
    f = v.finding("bound_rhs")
    assert (f.level, f.kind) == ("warn", "spread")
    assert f.numbers["decades"] == pytest.approx(13.90982, abs=1e-5)
    assert (f.numbers["s_lo"], f.numbers["s_hi"]) == (1.6e-5, 1.3e9)


def test_residue_left_out_is_ok_and_reported() -> None:
    """Raw RHS minimum 2e-15 is residue; the pool is [0.1, 1e3] → N = 0, ok;
    the residue is an info finding."""
    r = _r(0.1, 1e3, rhs=(2e-15, 1e3), bound_rhs_residue=(5, 2e-15), residue_floor=1e-6)
    plan = recommend_scaling(r, ScalingConfig())
    assert plan.user_bound_scale == 0
    v = _assess(r, plan)
    assert v.level == "ok"
    info = [f for f in v.findings if f.group == "info"]
    assert [(f.kind, f.numbers["count"]) for f in info] == [("residue", 5.0)]


def test_uniform_huge_hits_the_scale_option_limit() -> None:
    """5e19 everywhere at tol 1e-8: the large end needs
    N <= floor(log2(1e10/5e19)) = -33, outside [-30, 30], while the spread
    alone would fit (hard N_min = -88) → scale_option_limit.  The plan keeps
    clamp(floor(log2(1e6/5e19)) = -46) = -30: 5e19·2**-30 = 4.66e10 > 1e10."""
    r = _r(5e19, 5e19)
    plan = recommend_scaling(r, ScalingConfig(), primal_feasibility_tolerance=1e-8)
    assert plan.user_bound_scale == -30
    assert plan.bound_tag == "unscalable"
    f = _assess(r, plan, tol=1e-8).finding("bound_rhs")
    assert (f.level, f.kind) == ("error", "scale_option_limit")
    assert f.numbers["s_hi"] == pytest.approx(4.6566128730773926e10)


def test_uniform_huge_fits_at_default_tolerance() -> None:
    """5e19 at tol 1e-7: large end N <= floor(log2(1e11/5e19)) = -29, the
    window is [-30, -29]; the tier's -30 is inside → ok."""
    r = _r(5e19, 5e19)
    plan = recommend_scaling(r, ScalingConfig())
    assert plan.user_bound_scale == -30
    assert _assess(r, plan).level == "ok"


def test_manual_exponent_outside_window_is_error() -> None:
    """Manual N = -10 with b_lo* = 1e-4: 1e-4·2**-10 = 9.765625e-8 < 1e-6."""
    r = _r(1e-4, 1e3)
    plan = recommend_scaling(r, ScalingConfig(user_bound_scale=-10))
    assert plan.user_bound_scale == -10 and plan.bound_source == "manual"
    f = _assess(r, plan).finding("bound_rhs")
    assert (f.level, f.kind) == ("error", "manual_out_of_window")
    assert f.numbers["s_lo"] == 9.765625e-08
    assert f.numbers["n_used"] == -10.0


def test_window_moves_with_tolerance_and_keeps_17_decades() -> None:
    for tol, lo, hi in ((1e-8, 1e-7, 1e10), (1e-6, 1e-5, 1e12)):
        w = bound_scale_window(1.0, 1.0, tol)
        assert (w.lo_edge, w.hi_edge) == (pytest.approx(lo), pytest.approx(hi))
        assert math.log10(w.hi_edge / w.lo_edge) == pytest.approx(17.0)
        assert _assess(_r(1.0, 1.0), _plan(0), tol=tol).window == (
            pytest.approx(lo),
            pytest.approx(hi),
        )


def test_kkt_tolerance_replaces_the_primal_tolerance() -> None:
    """kkt_tolerance = 1e-6 (≠ its 1e-7 default) overrides
    primal_feasibility_tolerance = 1e-9, as HiGHS does."""

    class _Pb:
        _solver_options = {"kkt_tolerance": 1e-6, "primal_feasibility_tolerance": 1e-9}

    plan = recommend_scaling(_r(1.0, 10.0), ScalingConfig(), problem=_Pb())
    assert plan.primal_feasibility_tolerance == 1e-6


def test_integer_matrix_scaled_below_drop_value_warns() -> None:
    """Integer-column terms × 2**-20: 1e-3·2**-20 = 9.5367e-10 <= 1e-9."""
    r = _r(1.0, 1e3, matrix_continuous=(1e-2, 1e2), integer_matrix=(1e-3, 1.0))
    f = _assess(r, _plan(-20)).finding("matrix")
    assert (f.level, f.kind) == ("warn", "dropped")
    assert f.numbers["s_lo"] == 9.5367431640625e-10


def test_integer_matrix_scaled_above_refuse_value_errors() -> None:
    """Integer-column terms × 2**20: 1e10·2**20 = 1.048576e16 >= 1e15."""
    r = _r(1.0, 1e3, matrix_continuous=(1e-2, 1e2), integer_matrix=(1.0, 1e10))
    f = _assess(r, _plan(20)).finding("matrix")
    assert (f.level, f.kind) == ("error", "refused")
    assert f.numbers["s_hi"] == 1.048576e16


def test_integer_cost_scaled_by_bound_exponent() -> None:
    """N_obj = 0, N_bnd = -10: continuous costs [1, 10]; integer costs
    1·2**-10 = 9.765625e-4; spread log10(10/9.765625e-4) = 4.0103 → ok."""
    r = _r(1.0, 1e3, cost=(1.0, 10.0), cost_continuous=(1.0, 10.0), integer_cost=(1.0, 1.0))
    f = _assess(r, _plan(-10)).finding("cost")
    assert f.level == "ok"
    assert f.numbers["s_lo"] == 0.0009765625
    assert f.numbers["decades"] == pytest.approx(4.0103, abs=1e-4)


def test_small_cost_warns() -> None:
    """Costs [1e-7, 1] at N_obj = 0, dual tol 1e-7: 1e-7 < 10·1e-7."""
    r = _r(1.0, 1e3, cost=(1e-7, 1.0), cost_continuous=(1e-7, 1.0))
    f = _assess(r, _plan(0)).finding("cost")
    assert (f.level, f.kind) == ("warn", "small_cost")


def test_commercial_assesses_at_zero() -> None:
    """apply_user_scale=False ignores the plan's -20: the pool [1.0, 1e3]
    stays as it is."""
    v = _assess(_r(1.0, 1e3), _plan(-20), apply_user_scale=False)
    assert (v.bound_scale, v.objective_scale, v.bound_source) == (0, 0, "unscaled")
    f = v.finding("bound_rhs")
    assert (f.numbers["s_lo"], f.level) == (1.0, "ok")


def test_caused_by_prescale_error_and_warn() -> None:
    """Error: post [0.5, 1e18] has no exponent; pre [0.5, 1e9] has
    (N_min -18 <= N_max floor(log2(1e11/1e9)) = 6).  Warn: post
    [2.1e-5, 1.4e9] is 13.82 decades, pre [2.7e-3, 1.7e8] 10.80."""
    post = _r(0.5, 1e18)
    f = _assess(post, _plan(-20, source="auto"), ranges_pre=_r(0.5, 1e9)).finding("bound_rhs")
    assert f.level == "error" and f.caused_by_prescale
    f = _assess(post, _plan(-20, source="auto"), ranges_pre=_r(0.5, 1e18)).finding("bound_rhs")
    assert f.level == "error" and not f.caused_by_prescale
    post = _r(2.1e-5, 1.4e9)
    f = _assess(post, _plan(-1, source="auto"), ranges_pre=_r(2.7e-3, 1.7e8)).finding("bound_rhs")
    assert f.level == "warn" and f.caused_by_prescale
    assert f.numbers["decades_pre"] == pytest.approx(10.7991, abs=1e-4)


def test_h2_shaped_pool_depends_on_tolerance() -> None:
    """Pool 2.1e-5 … 1.4e9 (tier: floor(log2(1e6/1.4e9)) = -11, smallest
    side).  tol 1e-8: preferred ceil(log2(1e-6/2.1e-5)) = -4, large
    floor(log2(1e10/1.4e9)) = 2 → -4.  tol 1e-7: ceil(log2(1e-5/2.1e-5)) =
    -1, large floor(log2(1e11/1.4e9)) = 6 → -1."""
    r = _r(2.1e-5, 1.4e9)
    p8 = recommend_scaling(r, ScalingConfig(), primal_feasibility_tolerance=1e-8)
    p7 = recommend_scaling(r, ScalingConfig(), primal_feasibility_tolerance=1e-7)
    assert (p8.user_bound_scale, p7.user_bound_scale) == (-4, -1)


def test_falls_back_to_hard_edge_when_preferred_does_not_fit() -> None:
    """Pool 1e-5 … 1e10 at tol 1e-7: preferred N >= ceil(log2(1e-5/1e-5)) =
    0 but the large end needs N <= floor(log2(1e11/1e10)) = 3 → preferred
    fits ([0, 3]).  Pool 1e-6 … 1e10: preferred ceil(log2(10)) = 4 > 3; hard
    ceil(log2(1e-6/1e-6)) = 0 → window [0, 3] at factor 10."""
    w = bound_scale_window(1e-5, 1e10, 1e-7)
    assert (w.n_min, w.n_max, w.small_end_factor) == (0, 3, 100.0)
    w = bound_scale_window(1e-6, 1e10, 1e-7)
    assert (w.n_min, w.n_max, w.small_end_factor) == (0, 3, 10.0)


def test_streamed_report_without_pool_uses_raw_union() -> None:
    """A report without the meaningful readout (e.g. from streamed ranges)
    falls back to the raw bound ∪ RHS pool."""
    r = RangeReport(
        matrix=(1.0, 1.0),
        cost=NAN,
        bound=(1e-3, 1e3),
        rhs=(1e-2, 1e12),
        cross_group_max_ratio=math.nan,
        trigger=True,
    )
    plan = recommend_scaling(r, ScalingConfig())
    assert plan.user_bound_scale == -6  # same pool as the smallest-side test
