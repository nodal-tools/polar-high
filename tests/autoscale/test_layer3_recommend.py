"""Layer 3 (HiGHS-native top-up) recommendation unit tests.

Constructs synthetic :class:`RangeReport` inputs covering the
documented branches of :func:`recommend_scaling`:

* **In-zone** — every magnitude already inside HiGHS' comfort zone.
  ``N_obj = N_bnd = 0`` (no-op).
* **Cost large-end only** — only the objective overshoots HiGHS'
  ``|c| <= 1e+4`` ceiling; bounds stay clean.  ``N_obj`` is the
  power-of-two exponent that pulls the worst cost into the comfort
  zone, ``N_bnd == 0``.
* **Bound large-end only** — symmetric: bounds overshoot, cost stays
  clean.  ``N_bnd != 0``, ``N_obj == 0``.
* **Severe overshoot (D's escape)** — ``max(|b|) >= 1e+9`` and the
  naive clamp would crush the small end.  Geometric-centering branch
  fires and the chosen exponent lies between the two clamping
  alternatives.
* **Refuse-to-scale (Rivendell-shaped)** — moderate overshoot where
  the naive recommendation would crush the small end below ``1e-4``.
  Must refuse and emit ``N_bnd == 0``.
* **Manual override** — ``config.user_bound_scale`` set; the integer
  is used verbatim and the reasoning string surfaces "manual override".
  Objective auto-recommendation still runs.
* **Clamp** — an input that would yield ``N=-50`` is clamped to
  ``-30`` (HiGHS' option-range floor).
"""

from __future__ import annotations

import math

from polar_high.autoscale import (
    RangeReport,
    ScalingConfig,
    recommend_scaling,
)
from polar_high.autoscale._layer3 import _HIGHS_LARGE_COST, _HIGHS_SMALL_COST


def _cfg(
    user_bound_scale: int | None = None,
    user_objective_scale: int | None = None,
) -> ScalingConfig:
    return ScalingConfig(
        threshold_decades=9.0,
        user_bound_scale=user_bound_scale,
        user_objective_scale=user_objective_scale,
        report_yaml_path=None,
    )


def _ranges(
    *,
    cost=(math.nan, math.nan),
    bound=(math.nan, math.nan),
    rhs=(math.nan, math.nan),
    matrix=(1e-2, 1e2),
) -> RangeReport:
    """Build a :class:`RangeReport` from per-group magnitude pairs."""
    return RangeReport(
        matrix=matrix,
        cost=cost,
        bound=bound,
        rhs=rhs,
        cross_group_max_ratio=math.nan,
        trigger=True,
    )


def test_in_zone_no_scaling() -> None:
    """All magnitudes inside HiGHS' comfort zone → both N values 0."""
    r = _ranges(cost=(1.0, 1e3), bound=(1e-2, 1e3), rhs=(1e-2, 1e3))
    plan = recommend_scaling(r, _cfg())
    assert plan.user_objective_scale == 0
    assert plan.user_bound_scale == 0
    assert plan.simplex_scale_strategy == 2
    assert "in-zone" in plan.reasoning


def test_cost_overshoot_only_centres() -> None:
    """``max(|c|) = 1e+7`` (min in zone) → centre the band over the zone.

    geo_range = sqrt(1.0 * 1e+7) = 3162; geo_zone = 1.0;
    N_obj = round(log2(1 / 3162)) = -12.  The band lands ``[2.4e-4,
    2.4e+3]`` — both ends inside ``[1e-4, 1e+4]``, straddling 1.0.
    """
    r = _ranges(cost=(1.0, 1e7), bound=(1e-2, 1e3), rhs=(1e-2, 1e3))
    plan = recommend_scaling(r, _cfg())
    assert plan.user_objective_scale == -12
    assert plan.user_bound_scale == 0
    factor = 2.0**plan.user_objective_scale
    assert _HIGHS_SMALL_COST <= 1.0 * factor
    assert 1e7 * factor <= _HIGHS_LARGE_COST
    assert "center" in plan.reasoning


def test_bound_overshoot_only_moderate_clamp_large() -> None:
    """Bound overshoot of ~3 decades, min comfortably above 1e-4 → clamp.

    max_b = 1e+8, min_b = 1e+1.  Clamp-large brings max to ~1e6.
    dl = floor(log2(1e+6 / 1e+8)) = floor(-6.64) = -7.
    scaled_min = 1e+1 * 2**-7 ≈ 0.078 → above 1e-4 → no refuse.
    """
    r = _ranges(cost=(1.0, 1e2), bound=(1e1, 1e8), rhs=(1e1, 1e8))
    plan = recommend_scaling(r, _cfg())
    assert plan.user_objective_scale == 0
    assert plan.user_bound_scale == -7
    assert "auto" in plan.reasoning


def test_severe_overshoot_geometric_escape() -> None:
    """``max(|b|) >= 1e+9`` AND naive clamp crushes min → geo-centering.

    min_b = 1e-3, max_b = 1e+12: naive dl = floor(log2(1e-6)) = -20,
    scaled_min = 1e-3 * 2**-20 ≈ 9.5e-10 (well below 1e-4) → escape.
    Severe trigger: 1e+12 >= 1e+9 ✓ → escape via geometric centering.

    Geo-centering exponent: log2(sqrt(1e-4 * 1e+6) / sqrt(1e-3 * 1e+12))
    = log2(sqrt(1e+2) / sqrt(1e+9)) = log2(10 / ~31623) ≈ -11.6 → -12.
    """
    r = _ranges(cost=(1.0, 1e2), bound=(1e-3, 1e12), rhs=(1e-3, 1e12))
    plan = recommend_scaling(r, _cfg())
    assert plan.user_bound_scale == -12
    assert "escape" in plan.reasoning


def test_refuse_to_scale_rivendell_shape() -> None:
    """Moderate overshoot where the naive clamp would crush min → refuse.

    Mirrors the Rivendell B0/S17 shape: tight col bounds at 1.0, RHS up
    to ~2e+8.  Naive dl ≈ floor(log2(1e+6 / 2e+8)) = floor(-7.64) = -8.
    With min ~1.84e-3, scaled_min = 1.84e-3 * 2**-8 ≈ 7.2e-6 → below
    1e-4.  Severe trigger: 2e+8 < 1e+9 → refuse.  N_bnd = 0.
    """
    r = _ranges(cost=(1.0, 1e2), bound=(1.0, 1.0), rhs=(1.84e-3, 2.02e8))
    plan = recommend_scaling(r, _cfg())
    assert plan.user_bound_scale == 0
    assert "refuse" in plan.reasoning


def test_manual_override_disables_auto_for_bounds_only() -> None:
    """``config.user_bound_scale`` set → that integer wins for bounds.

    Objective auto-recommendation still runs.  Reasoning string surfaces
    "manual override" for auditability.
    """
    r = _ranges(cost=(1.0, 1e7), bound=(1e-2, 1e3), rhs=(1e-2, 1e3))
    plan = recommend_scaling(r, _cfg(user_bound_scale=-5))
    assert plan.user_bound_scale == -5
    assert plan.user_objective_scale == -12
    assert "manual override" in plan.reasoning
    assert "user_bound_scale=-5" in plan.reasoning


def test_manual_override_objective() -> None:
    """``config.user_objective_scale`` set → that integer wins for cost.

    Bound auto-recommendation still runs.
    """
    r = _ranges(cost=(1.0, 1e7), bound=(1e1, 1e8), rhs=(1e1, 1e8))
    plan = recommend_scaling(r, _cfg(user_objective_scale=3))
    assert plan.user_objective_scale == 3
    assert plan.user_bound_scale == -7
    assert "manual override user_objective_scale=3" in plan.reasoning


def test_clamp_to_30() -> None:
    """An input that would yield N=-50 is clamped to N=-30."""
    r = _ranges(cost=(1.0, 1e20), bound=(1.0, 1.0), rhs=(1.0, 1.0))
    plan = recommend_scaling(r, _cfg())
    assert plan.user_objective_scale == -30


def test_empty_cost_yields_zero() -> None:
    """An LP with no finite-non-zero costs → N_obj=0."""
    r = _ranges(cost=(math.nan, math.nan), bound=(1.0, 1.0), rhs=(1.0, 1.0))
    plan = recommend_scaling(r, _cfg())
    assert plan.user_objective_scale == 0
    assert plan.user_bound_scale == 0


def test_cost_undershoot_only_centres() -> None:
    """``min(|c|) < 1e-4`` with a clean max → centre the band over the zone.

    geo_range = sqrt(1e-6 * 1e-3) = 3.16e-5; N_obj = round(log2(1 /
    3.16e-5)) = 15.  The band lands ``[3.3e-2, 3.3e+1]`` — both ends in
    zone, straddling 1.0.
    """
    r = _ranges(cost=(1e-6, 1e-3), bound=(1e-2, 1e3), rhs=(1e-2, 1e3))
    plan = recommend_scaling(r, _cfg())
    assert plan.user_objective_scale == 15
    assert plan.user_bound_scale == 0
    assert "center" in plan.reasoning
    factor = 2.0**plan.user_objective_scale
    assert _HIGHS_SMALL_COST <= 1e-6 * factor
    assert 1e-3 * factor <= _HIGHS_LARGE_COST


def test_cost_spread_equals_zone_lands_on_edges() -> None:
    """Spread exactly the zone width (8 decades) → ends land on the edges.

    cost = (1e-6, 1e+2): geo_range = sqrt(1e-6 * 1e+2) = 1e-2;
    N_obj = round(log2(1 / 1e-2)) = 7.  Band → ``[1.28e-4, 1.28e+4]``:
    min just clears the 1e-4 floor, max just past the conservative 1e+4
    ceiling (still ~2 decades below HiGHS' 1e+6 warning).
    """
    r = _ranges(cost=(1e-6, 1e2), bound=(1.0, 1.0), rhs=(1.0, 1.0))
    plan = recommend_scaling(r, _cfg())
    assert plan.user_objective_scale == 7
    assert "center" in plan.reasoning
    factor = 2.0**plan.user_objective_scale
    assert 1e-6 * factor >= _HIGHS_SMALL_COST  # small end cleared the floor


def test_cost_wide_spread_centres_symmetrically() -> None:
    """Spread > zone width → centre so the overshoot is symmetric.

    cost = (1e-9, 5e+3) (~12.7 decades): geo_range = sqrt(1e-9 * 5e+3) =
    2.24e-3; N_obj = round(log2(1 / 2.24e-3)) = 9.  Band → ``[5.1e-7,
    2.6e+6]``: neither end is slammed to a boundary — both overshoot by
    ~2.3 decades in log-space (small end below 1e-4, large end above
    1e+4 by nearly the same amount).
    """
    r = _ranges(cost=(1e-9, 5e3), bound=(1.0, 1.0), rhs=(1.0, 1.0))
    plan = recommend_scaling(r, _cfg())
    assert plan.user_objective_scale == 9
    assert "center" in plan.reasoning
    factor = 2.0**plan.user_objective_scale
    lo, hi = 1e-9 * factor, 5e3 * factor
    out_lo = math.log10(_HIGHS_SMALL_COST / lo)  # decades below the floor
    out_hi = math.log10(hi / _HIGHS_LARGE_COST)  # decades above the ceiling
    assert out_lo > 0 and out_hi > 0  # genuinely wider than the zone
    assert abs(out_lo - out_hi) < 0.5  # symmetric to within a rounding step


def test_both_ends_bind_centres() -> None:
    """Both ends out of zone (always > zone width) → centre, not clamp.

    cost = (1e-6, 1e+7): min undershoots 1e-4 AND max overshoots 1e+4.
    geo_range = sqrt(1e-6 * 1e+7) = 3.16; N_obj = round(log2(1 / 3.16))
    = -2.  Symmetric overshoot around 1.0 rather than pulling the top to
    the ceiling and crushing the bottom further.
    """
    r = _ranges(cost=(1e-6, 1e7), bound=(1.0, 1.0), rhs=(1.0, 1.0))
    plan = recommend_scaling(r, _cfg())
    assert plan.user_objective_scale == -2
    assert "center" in plan.reasoning


def test_cost_undershoot_legacy_objective_scale_band() -> None:
    """Canonical case: a small operational-cost band under a built-in
    objective rescale.

    A model that multiplies its whole objective by a small constant
    (e.g. a legacy ``1e-6`` cost scale) leaves a cheap-but-real
    operational cost near ~2e-5 while a slack/penalty term tops out
    below 1 — tripping HiGHS' "excessively small costs" warning even
    though the spread is in-zone.  Layer 3 centres it: geo_range =
    sqrt(2e-5 * 0.64) = 3.58e-3, N_obj = round(log2(1 / 3.58e-3)) = 8,
    landing the band in ``[~5e-3, ~1.6e2]`` straddling 1.0 (warning
    cleared, both ends in zone).
    """
    r = _ranges(cost=(2e-5, 0.64), bound=(1.0, 3e2), rhs=(1e-4, 2.5e2))
    plan = recommend_scaling(r, _cfg())
    assert plan.user_objective_scale == 8
    factor = 2.0**plan.user_objective_scale
    assert _HIGHS_SMALL_COST <= 2e-5 * factor <= _HIGHS_LARGE_COST
    assert 0.64 * factor <= _HIGHS_LARGE_COST
    assert "center" in plan.reasoning


def test_cost_in_band_not_scaled() -> None:
    """A band already inside ``[1e-4, 1e+4]`` is left alone — no scaling."""
    r = _ranges(cost=(1e-4, 1e2), bound=(1.0, 1.0), rhs=(1.0, 1.0))
    plan = recommend_scaling(r, _cfg())
    assert plan.user_objective_scale == 0
    assert "in-zone" in plan.reasoning
