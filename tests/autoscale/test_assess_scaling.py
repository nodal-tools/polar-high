"""Layer 3 tolerance window and :func:`assess_scaling`.

Expected values are worked out by hand in each docstring (``log2`` /
``2**N`` arithmetic on the numbers written in the test), not copied from a
run of the code.  Window at tolerance ``tol``: hard edges ``[10·tol,
1e18·tol]``; the preferred small end ``100·tol`` only in the smallest-side
tier.
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
    """0.5 … 1e18 at tol 1e-7: N_min = ceil(log2(1e-6/0.5)) =
    ceil(1 + log2(1e-6)) = ceil(-18.93) = -18 (0.5·2**-18 = 2**-19 =
    1.9073486328125e-6 >= 1e-6, 0.5·2**-19 = 9.5e-7 < 1e-6), N_max =
    floor(log2(1e11/1e18)) = floor(-23.25) = -24 → no exponent.  The plan
    keeps the smallest limit enforceable: N = -18 (the large end
    1e18·2**-18 = 3.814697265625e12 stays below the solver infinity 1e20,
    so the cap does not bind), tag unscalable+smallest-side.  Fitting the
    largest value (N = -24) allows 1e-7·2**24 = 1.6777216 in model units,
    3.3554432 × the smallest limit."""
    r = _r(0.5, 1e18)
    plan = recommend_scaling(r, ScalingConfig())
    assert plan.user_bound_scale == -18
    assert plan.bound_tag == "unscalable+smallest-side"
    assert not plan.bound_window.fits
    v = _assess(r, plan)
    assert v.level == "error"
    f = v.finding("bound_rhs")
    assert f.kind == "spread"
    assert (f.numbers["s_lo"], f.numbers["s_hi"]) == (1.9073486328125e-06, 3.814697265625e12)
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
    alone would fit (hard N_min = ceil(log2(1e-7/5e19)) = ceil(log2(2e-27))
    = ceil(1 - 27·3.3219) = ceil(-88.69) = -88) → scale_option_limit.  The
    plan takes the smallest-side -88 (below the infinity cap) clamped to
    HiGHS' range: -30, and 5e19·2**-30 = 4.66e10 > 1e10."""
    r = _r(5e19, 5e19)
    plan = recommend_scaling(r, ScalingConfig(), primal_feasibility_tolerance=1e-8)
    assert plan.user_bound_scale == -30
    assert plan.bound_tag == "unscalable+smallest-side"
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


def test_layer3_window_uses_the_primal_tolerance_not_kkt() -> None:
    """kkt_tolerance = 1e-6 (≠ its 1e-7 default) does not replace
    primal_feasibility_tolerance = 1e-9 for the window (the simplex enforces
    the latter on the bounds); the diagnostics' resolver keeps HiGHS' kkt
    rule."""
    from polar_high.autoscale import resolve_feasibility_tolerances

    class _Pb:
        _solver_options = {"kkt_tolerance": 1e-6, "primal_feasibility_tolerance": 1e-9}

    plan = recommend_scaling(_r(1.0, 10.0), ScalingConfig(), problem=_Pb())
    assert plan.primal_feasibility_tolerance == 1e-9
    assert plan.bound_window.lo_edge == pytest.approx(1e-8)
    assert resolve_feasibility_tolerances(_Pb()) == (1e-6, 1e-6)
    assert resolve_feasibility_tolerances(_Pb(), kkt_override=False) == (1e-9, 1e-7)


def test_integer_matrix_scaled_below_drop_value_warns() -> None:
    """Integer-column terms × 2**-20: 1e-3·2**-20 = 9.5367e-10 <= 1e-9."""
    r = _r(1.0, 1e3, matrix_continuous=(1e-2, 1e2), integer_matrix=(1e-3, 1.0))
    f = _assess(r, _plan(-20)).finding("matrix")
    assert (f.level, f.kind) == ("warn", "dropped")
    assert f.numbers["s_lo"] == 9.5367431640625e-10


def test_integer_matrix_scaled_above_refuse_value_warns() -> None:
    """Integer-column terms × 2**20: 1e10·2**20 = 1.048576e16 >= 1e15 —
    HiGHS will refuse the model and say so itself; the finding warns."""
    r = _r(1.0, 1e3, matrix_continuous=(1e-2, 1e2), integer_matrix=(1.0, 1e10))
    v = _assess(r, _plan(20))
    f = v.finding("matrix")
    assert (f.level, f.kind) == ("warn", "refused")
    assert v.level == "warn"
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


def test_tier_reads_the_meaningful_pool() -> None:
    """Raw bounds reach 1e30 (no limit to the solver) but the meaningful
    pool is (1e-2, 1e6): max 1e6 is not above the comfort ceiling 1e6 →
    in-zone, N = 0; the hard window [ceil(log2(1e-6/1e-2)),
    floor(log2(1e11/1e6))] = [ceil(-13.29), floor(16.61)] = [-13, 16] keeps
    it."""
    r = _r(1e-2, 1e6, bound=(1e-2, 1e30), rhs=(1e-2, 1e6))
    plan = recommend_scaling(r, ScalingConfig())
    assert (plan.user_bound_scale, plan.bound_tag) == (0, "in-zone")
    assert (plan.bound_window.n_min, plan.bound_window.n_max) == (-13, 16)


def test_tier_falls_back_to_the_raw_range() -> None:
    """The same raw ranges without the readout: pool (1e-2, 1e30).  Tier
    smallest-side (clamp floor(log2(1e6/1e30)) = -80 crushes the small end,
    1e30 >= 1e9).  Hard N_min = -13 > N_max = floor(log2(1e11/1e30)) =
    floor(-63.1) = -64 → unscalable; the infinity cap needs 1e30·2**N <
    1e20, i.e. 2**N < 1e-10: 2**-33 = 1.16e-10 is not, 2**-34 = 5.8e-11 is
    → N = min(-13, -34) = -34, clamped to -30."""
    r = RangeReport(
        matrix=(1.0, 1.0),
        cost=NAN,
        bound=(1e-2, 1e30),
        rhs=(1e-2, 1e6),
        cross_group_max_ratio=math.nan,
        trigger=True,
    )
    plan = recommend_scaling(r, ScalingConfig())
    assert (plan.user_bound_scale, plan.bound_tag) == (-30, "unscalable+smallest-side")


def test_unscalable_exponent_capped_below_the_solver_infinity() -> None:
    """Pool 1e-12 … 1e15 at tol 1e-7 (27 decades).  Hard N_min =
    ceil(log2(1e-6/1e-12)) = ceil(19.93) = 20; N_max = floor(log2(1e11/1e15))
    = floor(-13.29) = -14 → spread.  The small end wants N = 20, but
    1e15·2**N < 1e20 needs 2**N < 1e5: 2**16 = 65536 fits, 2**17 = 131072
    does not → N = 16, the large end at 1e15·65536 = 6.5536e19."""
    r = _r(1e-12, 1e15)
    plan = recommend_scaling(r, ScalingConfig())
    assert (plan.user_bound_scale, plan.bound_tag) == (16, "unscalable+smallest-side")
    f = _assess(r, plan).finding("bound_rhs")
    assert (f.level, f.kind) == ("error", "spread")
    assert f.numbers["s_hi"] == 6.5536e19


def test_infinity_cap_follows_the_reported_infinite_bound() -> None:
    """Same pool with infinite_bound 1e18: 1e15·2**N < 1e18 needs 2**N <
    1e3: 2**9 = 512 fits, 2**10 = 1024 does not → N = 9."""
    r = _r(1e-12, 1e15, infinite_bound=1e18)
    assert recommend_scaling(r, ScalingConfig()).user_bound_scale == 9


def test_in_zone_and_refuse_tiers_keep_the_hard_edge() -> None:
    """The 100·tol preference does not apply outside the smallest-side tier.

    * [1e-6, 1e3] (in-zone): hard N_min = ceil(log2(1e-6/1e-6)) = 0 → N = 0
      (the preference would have raised it to ceil(log2(10)) = 4).
    * [1e-6, 2e8] (refuse: clamp floor(log2(1e6/2e8)) = floor(-7.64) = -8
      puts 1e-6·2**-8 = 3.9e-9 below 1e-4, and 2e8 < 1e9): N = 0, inside
      [0, floor(log2(1e11/2e8)) = floor(8.97) = 8].
    * [1e-7, 1e3] (in-zone, below the hard edge): raised to
      ceil(log2(1e-6/1e-7)) = ceil(3.32) = 4 (the preference would give
      ceil(log2(100)) = 7)."""
    p = recommend_scaling(_r(1e-6, 1e3), ScalingConfig())
    assert (p.user_bound_scale, p.bound_tag, p.bound_window.small_end_factor) == (
        0,
        "in-zone",
        10.0,
    )
    p = recommend_scaling(_r(1e-6, 2e8), ScalingConfig())
    assert (p.user_bound_scale, p.bound_tag) == (0, "refuse")
    assert (p.bound_window.n_min, p.bound_window.n_max) == (0, 8)
    p = recommend_scaling(_r(1e-7, 1e3), ScalingConfig())
    assert (p.user_bound_scale, p.bound_tag) == (4, "in-zone+raised")


def test_only_residue_means_nothing_to_scale() -> None:
    """The readout ran and every limit was residue: no pool → N = 0."""
    r = _r(
        1.0,
        1.0,
        bound=(1e-12, 1e-9),
        rhs=(1e-12, 1e-9),
        bound_rhs_meaningful=None,
        bound_rhs_residue=(4, 1e-9),
    )
    plan = recommend_scaling(r, ScalingConfig())
    assert (plan.user_bound_scale, plan.bound_tag, plan.bound_window) == (0, "in-zone", None)
