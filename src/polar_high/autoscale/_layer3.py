"""Layer 3 (HiGHS-native global scaling) — recommendation + apply.

Layer 3 sets three HiGHS options that let the solver apply its own
power-of-two-exact, internally unscaled-on-output global magnitude
shifts before the simplex starts:

* ``user_objective_scale`` — exponent ``N_obj`` such that HiGHS sees
  cost coefficients multiplied by ``2 ** N_obj``.  When a cost would
  trip a HiGHS warning (smallest ``|c| < 1e-4`` or worst ``|c| > 1e+4``)
  the band is **geometrically centred** over the comfort zone ``[1e-4,
  1e+4]`` — straddling ``1.0`` when it fits, overshooting symmetrically
  on both ends when the spread is wider than the zone (the warnings are
  soft, so centring beats slamming one end to a boundary).  HiGHS
  applies this power-of-two-exact scale internally and unscales the
  objective and duals on output, so it never perturbs the reported
  solution — only the magnitudes the simplex pivots on.
* ``user_bound_scale`` — exponent ``N_bnd`` such that HiGHS sees
  variable bounds AND row bounds (RHS) multiplied by ``2 ** N_bnd``.
  Pulls the worst-bound magnitude into HiGHS' comfort zone
  ``[1e-4, 1e+6]``.
* ``simplex_scale_strategy`` — HiGHS' matrix equilibration knob; pinned
  to ``2`` (ADVANCED equilibration) so the constraint-matrix spread
  that neither ``user_*_scale`` touches gets standard Curtis-Reid
  equilibration treatment.

``highspy.Highs.getObjectiveBoundScaling`` is not exposed in pinned
HiGHS builds, so the recommendation is reproduced in Python from the
post-detection :class:`RangeReport` (the same arithmetic HiGHS uses
internally to decide what to print in its "Consider scaling …"
warning).

The two-sided "geometric-centering escape" branch fires when the
worst-bound magnitude is severely above HiGHS' ceiling
(``>= _HIGHS_LARGE_BOUND * _SEVERE_LARGE_OVERSHOOT`` ≈ 1e+9) AND the
naive recommendation would drive the small end below the floor.  In
that regime the cost of refusing to scale (parallel-HiGHS pivots
overflowing single-precision intermediates) outweighs the cost of
crushing the small end below ``_HIGHS_SMALL_BOUND`` (presolve
false-infeasibility risk), so we replace the refusal with geometric
centering of the bound range over HiGHS' comfort zone.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from ._config import (
    USER_SCALE_CLAMP_HI,
    USER_SCALE_CLAMP_LO,
    ScalingConfig,
)
from ._precedence import get_explicit_option, has_explicit_option
from ._ranges import RangeReport

_logger = logging.getLogger(__name__)


# HiGHS' comfort zones (mirrored from polar-high's port of
# ``HighsSolve.cpp::suggestScaling``):
#
# * Bounds + RHS: ``[1e-4, 1e+6]``.
# * Cost: HiGHS warns "Problem has some excessively small costs" when the
#   smallest non-zero ``|c|`` falls below ``1e-4`` and "Problem has some
#   excessively large costs" when the worst ``|c|`` exceeds ``1e+6``
#   (both verified empirically against HiGHS 1.14: ``min|c|=1e-5`` warns
#   while ``1e-4`` does not; ``max|c|=2e+6`` warns while ``1e+6`` does
#   not).  ``_HIGHS_LARGE_COST`` is polar-high's **conservative working
#   ceiling** for the cost axis — 2 decades below HiGHS' actual ``1e+6``
#   warning, mirroring the down-scale target the bound axis uses — so the
#   autoscaler keeps the scaled costs well clear of the warning band.
_HIGHS_LARGE_BOUND = 1e6
_HIGHS_SMALL_BOUND = 1e-4
_HIGHS_LARGE_COST = 1e4
_HIGHS_SMALL_COST = 1e-4

# Severe-overshoot trigger for the two-sided escape branch.  When
# ``max_b >= _HIGHS_LARGE_BOUND * _SEVERE_LARGE_OVERSHOOT`` (~1e+9) the
# post-detect LP is so skewed that one-sided clamping crushes the small
# end below the HiGHS warning threshold.  Geometric-centering distributes
# the (unavoidable) violation symmetrically.  Threshold tuned empirically
# against operational LPs: models with ``rhs_max ≥ ~1e+10`` need the
# escape under parallel simplex, while ``rhs_max ≤ ~2e+8`` LPs solve
# cleanly at N=0 with the serial dual simplex.
_SEVERE_LARGE_OVERSHOOT = 1e3

# HiGHS ``simplex_scale_strategy`` value: 2 = ADVANCED equilibration
# (Curtis–Reid).
_SIMPLEX_SCALE_STRATEGY_DEFAULT = 2


@dataclass(frozen=True)
class Layer3Plan:
    """HiGHS-native scaling decisions for one solve.

    Attributes
    ----------
    user_objective_scale:
        Power-of-two exponent applied via the HiGHS option
        ``user_objective_scale`` (``cost coefficients × 2**N``).  ``0``
        means "no scaling needed" — within HiGHS' comfort zone already.
    user_bound_scale:
        Power-of-two exponent applied via ``user_bound_scale``
        (variable bounds AND row bounds × ``2**N``).  ``0`` is a no-op.
    simplex_scale_strategy:
        HiGHS' equilibration strategy (0..5).  Default ``2`` (ADVANCED).
    reasoning:
        One-line free-text string explaining where the values came
        from (auto / manual override / escape).  Surfaces in caller
        audit logs.
    objective_skipped_external:
        ``True`` when the objective recommendation was skipped because
        a caller-set ``user_objective_scale`` already exists on the
        problem (precedence-respect).
    bound_skipped_external:
        ``True`` when the bound recommendation was skipped because a
        caller-set ``user_bound_scale`` already exists on the problem.
    """

    user_objective_scale: int
    user_bound_scale: int
    simplex_scale_strategy: int
    reasoning: str
    objective_skipped_external: bool = False
    bound_skipped_external: bool = False


def _clamp(n: int) -> int:
    if n < USER_SCALE_CLAMP_LO:
        return USER_SCALE_CLAMP_LO
    if n > USER_SCALE_CLAMP_HI:
        return USER_SCALE_CLAMP_HI
    return n


def _safe_float(x: float, fallback: float) -> float:
    if x is None:
        return fallback
    if isinstance(x, float) and math.isnan(x):
        return fallback
    return float(x)


def _recommend_objective_scale(cost_max: float, cost_min: float) -> tuple[int, str]:
    """Power-of-two exponent that geometrically **centres** the cost band
    over the comfort zone ``[_HIGHS_SMALL_COST, _HIGHS_LARGE_COST]``
    (``[1e-4, 1e+4]``).

    The objective carries a single scale exponent, and HiGHS' small/large
    cost warnings are soft (conditioning degrades smoothly, not at a hard
    cliff).  So rather than clamping whichever end is out of zone to a
    boundary — which can shove the other end further out — we place the
    band's geometric centre ``sqrt(cost_min * cost_max)`` at the zone's
    geometric centre ``sqrt(1e-4 * 1e+4) = 1.0``:

    * **Spread narrower than the zone** (the usual case): both ends land
      inside ``[1e-4, 1e+4]``, the band straddling ``1.0`` — the
      best-conditioned position, with margin above the dual-feasibility
      tolerance for the cheapest costs.
    * **Spread exactly the zone width** (~8 decades): the two ends land
      on the two boundaries.
    * **Spread wider than the zone**: the (unavoidable) violation falls
      **symmetrically** on both ends in log-space — neither end is
      slammed to a boundary while the other overshoots far.

    Scaling is only triggered when a cost would actually trip a HiGHS
    warning (``min(|c|) < 1e-4`` or ``max(|c|) > 1e+4``); a band already
    inside the zone is left untouched (``N = 0``) so well-scaled models
    are never perturbed.  Returns ``(exponent, tag)`` with ``tag`` ∈
    {``in-zone``, ``center``} for the operator audit log.
    """
    if not math.isfinite(cost_max) or cost_max <= 0.0:
        return 0, "in-zone"
    has_min = math.isfinite(cost_min) and cost_min > 0.0
    small = has_min and cost_min < _HIGHS_SMALL_COST
    large = cost_max > _HIGHS_LARGE_COST
    if not small and not large:
        return 0, "in-zone"
    # Degenerate single-magnitude vector (min absent) → centre that value.
    lo = cost_min if has_min else cost_max
    geo_range = math.sqrt(lo * cost_max)
    geo_zone = math.sqrt(_HIGHS_SMALL_COST * _HIGHS_LARGE_COST)
    if not (math.isfinite(geo_range) and geo_range > 0.0):
        return 0, "in-zone"
    if not (math.isfinite(geo_zone) and geo_zone > 0.0):
        return 0, "in-zone"
    return int(round(math.log2(geo_zone / geo_range))), "center"


def _recommend_bound_scale(
    bound_max: float,
    bound_min: float,
    rhs_max: float,
    rhs_min: float,
) -> tuple[int, str]:
    """Power-of-two exponent that pulls ``max(|b|)`` into the HiGHS
    comfort zone, with D's geometric-centering escape on severe
    overshoots.

    Inputs are the four magnitudes (``±inf`` / ``nan`` → treated as 0
    for the max comparison, ``+inf`` for the min).  Returns
    ``(n, reasoning_tag)`` where ``reasoning_tag`` is one of
    ``"in-zone"``, ``"clamp-large"``, ``"clamp-small"``, ``"escape"``,
    or ``"refuse"`` for the operator log.
    """
    max_b = max(_safe_float(bound_max, 0.0), _safe_float(rhs_max, 0.0))
    min_b = min(_safe_float(bound_min, math.inf), _safe_float(rhs_min, math.inf))

    if not math.isfinite(max_b) or max_b <= 0.0:
        return 0, "in-zone"

    if max_b > _HIGHS_LARGE_BOUND:
        ratio = _HIGHS_LARGE_BOUND / max_b
        tag = "clamp-large"
    elif max_b < _HIGHS_SMALL_BOUND:
        ratio = _HIGHS_SMALL_BOUND / max_b
        tag = "clamp-small"
    else:
        return 0, "in-zone"

    # Outer-rounded log2: floor when ratio<1 (scaling down), ceil when
    # ratio>1 (scaling up).  Same conservative rule HiGHS uses.
    if ratio < 1.0:
        dl = math.floor(math.log2(ratio))
    else:
        dl = math.ceil(math.log2(ratio))

    # Min-floor guard: when scaling down would drive the small end
    # below ``_HIGHS_SMALL_BOUND``, decide between refusing (the
    # refuse-safe behaviour for moderate overshoots) and the
    # geometric-centering escape (the refuse-unsafe behaviour for
    # severe overshoots).
    if dl < 0 and math.isfinite(min_b) and min_b > 0.0:
        scaled_min = min_b * (2.0**dl)
        if scaled_min < _HIGHS_SMALL_BOUND:
            if max_b >= _HIGHS_LARGE_BOUND * _SEVERE_LARGE_OVERSHOOT:
                # Severe overshoot: geometric centering over the comfort
                # zone, distributing the unavoidable violation across
                # both ends.
                geo_range = math.sqrt(min_b * max_b)
                geo_band = math.sqrt(_HIGHS_SMALL_BOUND * _HIGHS_LARGE_BOUND)
                if (
                    math.isfinite(geo_range)
                    and geo_range > 0.0
                    and math.isfinite(geo_band)
                    and geo_band > 0.0
                ):
                    dl = int(round(math.log2(geo_band / geo_range)))
                    tag = "escape"
                else:
                    return 0, "refuse"
            else:
                # Moderate overshoot — LPs with tightly-clustered col
                # bounds land here.  Refusing keeps N=0 and lets HiGHS'
                # own default scaling handle the row spread.
                return 0, "refuse"

    return int(dl), tag


def recommend_scaling(
    ranges: RangeReport,
    config: ScalingConfig,
    *,
    problem: Any = None,
) -> Layer3Plan:
    """Compute the three HiGHS-native scaling values for one solve.

    Pulls ``max(|c|)`` from ``ranges.cost`` and ``max(|b|)``, ``min(|b|)``
    from the union of ``ranges.bound`` and ``ranges.rhs``.

    Precedence rules (per axis, independent):

    1. **Precedence-respect** — when ``problem`` is given and already
       has ``user_bound_scale`` or ``user_objective_scale`` set
       explicitly (via :meth:`Problem.set_solver_options` or a
       loaded ``highs.opt``), the corresponding recommendation is
       skipped and the existing value is preserved on the plan.  The
       reasoning string surfaces "external user_*_scale=N".
    2. **Manual override** — when the corresponding
       ``config.user_objective_scale`` / ``config.user_bound_scale``
       is set, the plan uses that integer verbatim.  Reasoning
       string surfaces "manual override ...".
    3. **Severe-overshoot escape** — geometric-centering branch fires
       when the naive recommendation would crush the small end and
       ``max_b >= 1e+9``.  D's branch from polar-high's
       ``_recommend_user_bound_scale``.
    4. **Default auto** — power-of-two exponents that pull the worst
       end into the HiGHS comfort zone.

    The resulting exponents are clamped to ``[-30, 30]`` (HiGHS' option
    bounds).
    """
    cost_lo, cost_hi = ranges.cost
    bound_lo, bound_hi = ranges.bound
    rhs_lo, rhs_hi = ranges.rhs

    # --- Objective axis ---------------------------------------------------
    obj_skipped_external = False
    reasoning_obj_tag = "in-zone"
    if problem is not None and has_explicit_option(problem, "user_objective_scale"):
        ext = get_explicit_option(problem, "user_objective_scale")
        try:
            n_obj = _clamp(int(ext))
        except (TypeError, ValueError):
            n_obj_raw, obj_tag = _recommend_objective_scale(
                _safe_float(cost_hi, 0.0), _safe_float(cost_lo, math.inf)
            )
            n_obj = _clamp(n_obj_raw)
            reasoning_obj_tag = obj_tag
        else:
            obj_skipped_external = True
            reasoning_obj_tag = "external"
            _logger.info(
                "respecting external user_objective_scale=%d",
                n_obj,
            )
    elif config.user_objective_scale is not None:
        n_obj = _clamp(int(config.user_objective_scale))
        reasoning_obj_tag = "manual"
    else:
        n_obj_raw, obj_tag = _recommend_objective_scale(
            _safe_float(cost_hi, 0.0), _safe_float(cost_lo, math.inf)
        )
        n_obj = _clamp(n_obj_raw)
        reasoning_obj_tag = obj_tag

    # --- Bound axis -------------------------------------------------------
    bnd_skipped_external = False
    if problem is not None and has_explicit_option(problem, "user_bound_scale"):
        ext = get_explicit_option(problem, "user_bound_scale")
        try:
            n_bnd = _clamp(int(ext))
        except (TypeError, ValueError):
            n_bnd_raw, bnd_tag = _recommend_bound_scale(
                bound_max=_safe_float(bound_hi, 0.0),
                bound_min=_safe_float(bound_lo, math.inf),
                rhs_max=_safe_float(rhs_hi, 0.0),
                rhs_min=_safe_float(rhs_lo, math.inf),
            )
            n_bnd = _clamp(n_bnd_raw)
            reasoning_bnd_tag = bnd_tag
        else:
            bnd_skipped_external = True
            reasoning_bnd_tag = "external"
            _logger.info(
                "respecting external user_bound_scale=%d",
                n_bnd,
            )
    elif config.user_bound_scale is not None:
        n_bnd = _clamp(int(config.user_bound_scale))
        reasoning_bnd_tag = "manual"
    else:
        n_bnd_raw, bnd_tag = _recommend_bound_scale(
            bound_max=_safe_float(bound_hi, 0.0),
            bound_min=_safe_float(bound_lo, math.inf),
            rhs_max=_safe_float(rhs_hi, 0.0),
            rhs_min=_safe_float(rhs_lo, math.inf),
        )
        n_bnd = _clamp(n_bnd_raw)
        reasoning_bnd_tag = bnd_tag

    # --- Reasoning string -------------------------------------------------
    parts: list[str] = []
    if obj_skipped_external:
        parts.append(f"external user_objective_scale={n_obj}")
    elif config.user_objective_scale is not None:
        parts.append(f"manual override user_objective_scale={config.user_objective_scale}")
    if bnd_skipped_external:
        parts.append(f"external user_bound_scale={n_bnd}")
    elif config.user_bound_scale is not None:
        parts.append(f"manual override user_bound_scale={config.user_bound_scale}")

    if not parts:
        if n_obj == 0 and n_bnd == 0:
            reasoning = f"in-zone (obj={reasoning_obj_tag}, bound={reasoning_bnd_tag})"
        else:
            reasoning = (
                f"auto (N_obj={n_obj}, obj_tag={reasoning_obj_tag}, "
                f"N_bnd={n_bnd}, bound_tag={reasoning_bnd_tag})"
            )
    else:
        # Mixed: at least one axis is external/manual.  Always surface
        # the auto value too, so the audit shows what the autoscaler
        # would have picked for the non-overridden axis.
        parts.append(
            f"auto (N_obj={n_obj}, obj_tag={reasoning_obj_tag}, "
            f"N_bnd={n_bnd}, bound_tag={reasoning_bnd_tag})"
        )
        reasoning = "; ".join(parts)

    return Layer3Plan(
        user_objective_scale=n_obj,
        user_bound_scale=n_bnd,
        simplex_scale_strategy=_SIMPLEX_SCALE_STRATEGY_DEFAULT,
        reasoning=reasoning,
        objective_skipped_external=obj_skipped_external,
        bound_skipped_external=bnd_skipped_external,
    )


def apply_scaling(problem: Any, plan: Layer3Plan) -> None:
    """Set the three HiGHS options on ``problem``.

    Uses polar-high's :meth:`Problem.set_solver_options` so the values
    survive both warm-LP and cold-LP code paths.  Existing options are
    preserved — we *merge* with whatever the caller already set on the
    Problem.

    Skipped-external axes do NOT overwrite the caller's value (that's
    the whole point of the precedence check).  We only write the axes
    the autoscaler is authoritative for.
    """
    existing = dict(getattr(problem, "_solver_options", None) or {})
    if not plan.objective_skipped_external and plan.user_objective_scale != 0:
        existing["user_objective_scale"] = int(plan.user_objective_scale)
    if not plan.bound_skipped_external and plan.user_bound_scale != 0:
        existing["user_bound_scale"] = int(plan.user_bound_scale)
    # Always set the simplex strategy — Layer 3 owns this value (callers
    # who want a different strategy should set it explicitly post-apply).
    existing["simplex_scale_strategy"] = int(plan.simplex_scale_strategy)
    problem.set_solver_options(existing)


__all__ = [
    "Layer3Plan",
    "apply_scaling",
    "recommend_scaling",
]
