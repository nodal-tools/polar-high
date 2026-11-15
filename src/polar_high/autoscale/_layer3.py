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

The "smallest-side" branch fires when the worst-bound magnitude is
severely above HiGHS' ceiling (``>= _HIGHS_LARGE_BOUND *
_SEVERE_LARGE_OVERSHOOT`` ≈ 1e+9) AND pulling it into the comfort zone
would drive the small end below the floor.  In that regime refusing to
scale (parallel-HiGHS pivots overflowing) is worse than scaling, so the
recommendation keeps the large-end clamp and leaves it to the tolerance
window below to lift the small end.

**Tolerance window.**  HiGHS checks ``primal_feasibility_tolerance`` (``tol``)
as an absolute amount on the user-scaled bounds, so the smallest meaningful
limit must stay well above ``tol`` after scaling, and the largest must stay
small enough for its rows to be checked at all.  Every automatic
``user_bound_scale`` is clipped into the window of exponents that keeps the
pool (:attr:`RangeReport.bound_rhs_meaningful`) inside ``[10·tol,
1e18·tol]`` (:func:`bound_scale_window`).  Only in the "smallest-side" tier,
where the range forces a choice anyway, the window prefers the narrower
``[100·tol, 1e18·tol]`` (the smallest limit enforced to 1 %) whenever an
exponent fits it, and the large end takes whatever is left; the other tiers
keep ``10·tol``, so a model that needed no scaling still gets none.
``10·tol`` follows from the tolerance (the smallest limit enforced to 10 %);
``1e18·tol`` is calibrated (the largest scaled value of a solve that worked
was 2.3e10 at ``tol = 1e-7``), so it is a keyword.  ``tol`` is the primal
feasibility tolerance the simplex enforces (``kkt_tolerance`` does not
replace it here).  When no exponent fits, the plan is tagged
``unscalable+smallest-side``: the exponent keeps the smallest meaningful
limit at ``10·tol`` or more (capped so the largest stays below the solver
infinity, and clamped to HiGHS' ``[-30, 30]``), and :func:`assess_scaling`
reports an error.  :func:`assess_scaling` judges any plan (automatic, manual
or external) against the hard window.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._config import (
    USER_SCALE_CLAMP_HI,
    USER_SCALE_CLAMP_LO,
    ScalingConfig,
)
from ._precedence import (
    HIGHS_DEFAULT_FEASIBILITY_TOLERANCE,
    HIGHS_DEFAULT_INFINITE_BOUND,
    get_explicit_option,
    has_explicit_option,
    resolve_feasibility_tolerances,
    resolve_infinite_bound,
)
from ._ranges import Extreme, RangeReport

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

# Tolerance window for the scaled bound / RHS pool: ``[10·tol, 1e18·tol]``
# (17 decades at any tolerance).  The lower edge enforces the smallest limit
# to 10 %; the upper edge is calibrated (see the module docstring).
# ``_PREFERRED_LO_FACTOR`` is the small-end target of the "smallest-side"
# tier whenever an exponent fits it (1 %).
_WINDOW_LO_FACTOR = 10.0
_WINDOW_HI_FACTOR = 1e18
_PREFERRED_LO_FACTOR = 100.0

# HiGHS ``small_matrix_value`` / ``large_matrix_value`` defaults: a matrix
# entry at or below the first is dropped, one at or above the second makes
# HiGHS refuse the model.
HIGHS_SMALL_MATRIX_VALUE = 1e-9
HIGHS_LARGE_MATRIX_VALUE = 1e15


@dataclass(frozen=True)
class ScaleWindow:
    """The ``user_bound_scale`` exponents that keep a pool ``[b_lo, b_hi]``
    inside the tolerance window, intersected with HiGHS' ``[-30, 30]``.

    ``n_min`` / ``n_max`` bound the window actually used: ``n_min`` comes
    from the preferred small-end target (``small_end_factor`` =
    ``preferred_lo_factor``) when that still leaves room for the large end,
    else from the hard one (``small_end_factor`` = ``lo_factor``).  ``fits``
    is ``n_min <= n_max``.  When it does not fit, ``reason`` says why:
    ``"spread"`` (no power of two fits the pool even without HiGHS' option
    range) or ``"scale_option_limit"`` (one would, but outside ``[-30, 30]``).
    ``n_min_hard`` / ``n_max_unclamped`` are the hard-edge exponents before the
    ``[-30, 30]`` cut.
    """

    n_min: int
    n_max: int
    small_end_factor: float
    lo_edge: float
    hi_edge: float
    n_min_hard: int
    n_max_unclamped: int
    reason: str = ""

    @property
    def fits(self) -> bool:
        return self.n_min <= self.n_max


def _ceil_exponent(target: float, value: float) -> int:
    """Smallest integer ``n`` with ``value · 2**n >= target`` (exact)."""
    n = math.ceil(math.log2(target / value))
    while math.ldexp(value, n) < target:
        n += 1
    while math.ldexp(value, n - 1) >= target:
        n -= 1
    return n


def _floor_exponent(target: float, value: float) -> int:
    """Largest integer ``n`` with ``value · 2**n <= target`` (exact)."""
    n = math.floor(math.log2(target / value))
    while math.ldexp(value, n) > target:
        n -= 1
    while math.ldexp(value, n + 1) <= target:
        n += 1
    return n


def bound_scale_window(
    b_lo: float,
    b_hi: float,
    primal_tol: float,
    *,
    lo_factor: float = _WINDOW_LO_FACTOR,
    hi_factor: float = _WINDOW_HI_FACTOR,
    preferred_lo_factor: float = _PREFERRED_LO_FACTOR,
) -> ScaleWindow:
    """The window of ``user_bound_scale`` exponents for a pool whose
    smallest meaningful magnitude is ``b_lo`` and largest ``b_hi`` (see
    :class:`ScaleWindow`).  The scaled pool is ``[b_lo·2**N, b_hi·2**N]``;
    the window keeps it inside ``[lo_factor·tol, hi_factor·tol]``, with the
    small end at ``preferred_lo_factor·tol`` or more when the large end still
    fits."""
    tol = float(primal_tol)
    lo_edge = lo_factor * tol
    hi_edge = hi_factor * tol
    n_max_raw = _floor_exponent(hi_edge, b_hi)
    n_min_hard = _ceil_exponent(lo_edge, b_lo)
    n_min_pref = _ceil_exponent(preferred_lo_factor * tol, b_lo)
    n_max = min(n_max_raw, USER_SCALE_CLAMP_HI)
    if max(n_min_pref, USER_SCALE_CLAMP_LO) <= n_max:
        n_min, factor = max(n_min_pref, USER_SCALE_CLAMP_LO), preferred_lo_factor
    else:
        n_min, factor = max(n_min_hard, USER_SCALE_CLAMP_LO), lo_factor
    reason = ""
    if n_min > n_max:
        reason = "spread" if n_min_hard > n_max_raw else "scale_option_limit"
    return ScaleWindow(
        n_min=n_min,
        n_max=n_max,
        small_end_factor=factor,
        lo_edge=lo_edge,
        hi_edge=hi_edge,
        n_min_hard=n_min_hard,
        n_max_unclamped=n_max_raw,
        reason=reason,
    )


def _pool(ranges: RangeReport) -> tuple[float, float] | None:
    """``(b_lo*, b_hi)`` of the bound / RHS pool: the meaningful readout when
    the report has it, else the raw bound ∪ RHS minimum and maximum."""
    if ranges.bound_rhs_meaningful is not None:
        lo, hi = ranges.bound_rhs_meaningful
        return float(lo), float(hi)
    if ranges.bound_rhs_residue is not None:
        # The readout ran and every entry was residue: no meaningful limit.
        return None
    los = [v for v in (ranges.bound[0], ranges.rhs[0]) if _finite_pos(v)]
    his = [v for v in (ranges.bound[1], ranges.rhs[1]) if _finite_pos(v)]
    if not los or not his:
        return None
    return float(min(los)), float(max(his))


def _finite_pos(x: Any) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x) and x > 0.0


def _below_infinite_exponent(b_hi: float, infinite: float) -> int:
    """Largest integer ``n`` with ``b_hi · 2**n < infinite`` (exact).

    An infinite ``infinite`` (``infinite_bound = inf``: the solver treats
    no finite value as "no limit") caps nothing, so the answer is the
    largest exponent the scale option accepts, ``USER_SCALE_CLAMP_HI``."""
    if math.isinf(infinite):
        return USER_SCALE_CLAMP_HI
    n = _floor_exponent(infinite, b_hi)
    while math.ldexp(b_hi, n) >= infinite:
        n -= 1
    return n


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
    #: Where ``user_bound_scale`` came from: ``auto``, ``manual`` or
    #: ``external``.
    bound_source: str = "auto"
    #: The bound tier tag (``in-zone``, ``clamp-large``, ``clamp-small``,
    #: ``smallest-side``, ``refuse``; ``+raised`` / ``+lowered`` when the
    #: tolerance window moved it; ``unscalable+smallest-side`` when no
    #: exponent fits and the smallest meaningful limit was kept enforceable).
    bound_tag: str = ""
    #: The tolerance window the automatic exponent was clipped into
    #: (``None`` for a manual / external exponent or an empty pool).
    bound_window: ScaleWindow | None = None
    #: The primal feasibility tolerance the window used.
    primal_feasibility_tolerance: float | None = None


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
    comfort zone; on a severe overshoot it keeps that exponent even when the
    small end leaves the zone ("smallest-side": the tolerance window then
    lifts the small end).

    Inputs are the four magnitudes (``±inf`` / ``nan`` → treated as 0
    for the max comparison, ``+inf`` for the min).  Returns
    ``(n, reasoning_tag)`` where ``reasoning_tag`` is one of
    ``"in-zone"``, ``"clamp-large"``, ``"clamp-small"``,
    ``"smallest-side"``, or ``"refuse"`` for the operator log.  The result
    is the policy exponent before the tolerance window
    (:func:`bound_scale_window`) clips it.
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
    # refuse-safe behaviour for moderate overshoots) and scaling anyway
    # (severe overshoots, where refusing is worse).
    if dl < 0 and math.isfinite(min_b) and min_b > 0.0:
        scaled_min = min_b * (2.0**dl)
        if scaled_min < _HIGHS_SMALL_BOUND:
            if max_b >= _HIGHS_LARGE_BOUND * _SEVERE_LARGE_OVERSHOOT:
                # Severe overshoot: keep the large-end clamp; the tolerance
                # window lifts the exponent until the smallest meaningful
                # limit is enforceable and the large end takes the rest.
                tag = "smallest-side"
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
    primal_feasibility_tolerance: float | None = None,
) -> Layer3Plan:
    """Compute the three HiGHS-native scaling values for one solve.

    Pulls ``max(|c|)`` from ``ranges.cost`` and ``max(|b|)``, ``min(|b|)``
    from the bound / RHS pool: ``ranges.bound_rhs_meaningful`` when present,
    else the union of ``ranges.bound`` and ``ranges.rhs``.

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
    3. **Default auto** — power-of-two exponents that pull the worst
       end of the bound / RHS pool (:attr:`RangeReport.bound_rhs_meaningful`
       when the report has it, else the raw ``bound`` ∪ ``rhs`` range) into
       the HiGHS comfort zone; on a severe overshoot (``max_b >= 1e+9``) the
       large-end clamp is kept even when the small end leaves the zone
       ("smallest-side").
    4. **Tolerance window** — an automatic bound exponent is then clipped
       into :func:`bound_scale_window` of the same pool, at
       ``primal_feasibility_tolerance`` (``None``: the
       ``primal_feasibility_tolerance`` set on ``problem``, else ``1e-7``;
       ``kkt_tolerance`` does not replace it).  The 100·tol small-end
       preference applies in the smallest-side tier only; the other tiers
       use the hard ``10·tol`` edge.  No exponent fits → the exponent is the
       smallest one that keeps the smallest meaningful limit at ``10·tol``
       or more, capped so the largest stays below ``infinite_bound`` and
       clamped to ``[-30, 30]``; tag ``unscalable+smallest-side``.  Manual
       and external exponents are not clipped (:func:`assess_scaling` judges
       them).

    The resulting exponents are clamped to ``[-30, 30]`` (HiGHS' option
    bounds).
    """
    if primal_feasibility_tolerance is not None:
        tol_p = float(primal_feasibility_tolerance)
    elif problem is not None:
        tol_p, _tol_d = resolve_feasibility_tolerances(problem, kkt_override=False)
    else:
        tol_p = HIGHS_DEFAULT_FEASIBILITY_TOLERANCE
    if ranges.infinite_bound is not None:
        infinite = float(ranges.infinite_bound)
    elif problem is not None:
        infinite = resolve_infinite_bound(problem)
    else:
        infinite = HIGHS_DEFAULT_INFINITE_BOUND
    cost_lo, cost_hi = ranges.cost

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
    def _auto_bound() -> tuple[int, str, ScaleWindow | None]:
        # The tier reads the same pool the window does: the meaningful
        # readout (no values at or above the solver infinity, no residue, no
        # integer-column bounds) when the report has it, else the raw
        # bound ∪ RHS range.  No pool (no limit at all, or only residue) →
        # nothing to scale.
        pool = _pool(ranges)
        if pool is None:
            return 0, "in-zone", None
        b_lo, b_hi = pool
        n_raw, tag = _recommend_bound_scale(
            bound_max=b_hi, bound_min=b_lo, rhs_max=0.0, rhs_min=math.inf
        )
        n = _clamp(n_raw)
        # The 100·tol preference only where the range forces the choice.
        pref = _PREFERRED_LO_FACTOR if tag == "smallest-side" else _WINDOW_LO_FACTOR
        window = bound_scale_window(b_lo, b_hi, tol_p, preferred_lo_factor=pref)
        if not window.fits:
            # Keep the smallest meaningful limit enforceable (>= 10·tol) and
            # let the large end take the excess, but never up to the solver
            # infinity (where a limit becomes no limit).
            n_un = min(window.n_min_hard, _below_infinite_exponent(b_hi, infinite))
            return _clamp(n_un), "unscalable+smallest-side", window
        if n < window.n_min:
            return window.n_min, f"{tag}+raised", window
        if n > window.n_max:
            return window.n_max, f"{tag}+lowered", window
        return n, tag, window

    bnd_skipped_external = False
    bnd_source = "auto"
    bnd_window: ScaleWindow | None = None
    if problem is not None and has_explicit_option(problem, "user_bound_scale"):
        ext = get_explicit_option(problem, "user_bound_scale")
        try:
            n_bnd = _clamp(int(ext))
        except (TypeError, ValueError):
            n_bnd, reasoning_bnd_tag, bnd_window = _auto_bound()
        else:
            bnd_skipped_external = True
            bnd_source = "external"
            reasoning_bnd_tag = "external"
            _logger.info(
                "respecting external user_bound_scale=%d",
                n_bnd,
            )
    elif config.user_bound_scale is not None:
        n_bnd = _clamp(int(config.user_bound_scale))
        bnd_source = "manual"
        reasoning_bnd_tag = "manual"
    else:
        n_bnd, reasoning_bnd_tag, bnd_window = _auto_bound()

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
        bound_source=bnd_source,
        bound_tag=reasoning_bnd_tag,
        bound_window=bnd_window,
        primal_feasibility_tolerance=tol_p,
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


_LEVEL_ORDER = {"ok": 0, "info": 0, "warn": 1, "error": 2}


@dataclass(frozen=True)
class Finding:
    """One result of :func:`assess_scaling`.

    ``group`` is ``bound_rhs``, ``cost``, ``matrix`` or ``info``; ``level``
    ``ok`` / ``warn`` / ``error`` (``info`` for the ``info`` group); ``kind``
    names what was found (``+``-joined when several):

    * ``bound_rhs``: ``spread`` (warn: more than ``warn_decades``; error: no
      exponent fits), ``scale_option_limit`` (an exponent would fit, but
      outside HiGHS' ``[-30, 30]``), ``manual_out_of_window`` /
      ``external_out_of_window`` / ``unscaled_out_of_window`` (the exponent
      used, set by the caller or ``0`` when not applied, leaves the window
      while one would fit), ``ok``;
    * ``cost``: ``spread``, ``small_cost`` (smallest scaled cost below
      ``10 · dual tolerance``), ``ok``;
    * ``matrix``: ``dropped`` (warn: a scaled term at or below
      ``small_matrix``, which HiGHS drops), ``refused`` (warn: one at or
      above ``large_matrix`` — HiGHS will refuse the model and say so
      itself), ``ok``;
    * ``info``: ``residue``, ``beyond_infinite``, ``integer_bounds``.

    ``numbers`` carries the measured values (see :func:`assess_scaling`);
    ``extremes`` maps ``min`` / ``max`` to the :class:`Extreme` holding that
    end (when the report has extremes).  ``caused_by_prescale`` is ``True``
    when ``ranges_pre`` was given and the finding is worse than it would be
    on ``ranges_pre`` (the scaling applied between the two readouts caused
    it).
    """

    group: str
    level: str
    kind: str
    numbers: Mapping[str, float] = field(default_factory=dict)
    extremes: Mapping[str, Extreme] = field(default_factory=dict)
    caused_by_prescale: bool = False


@dataclass(frozen=True)
class ScalingVerdict:
    """What :func:`assess_scaling` concluded: the worst ``level`` over the
    findings (``ok`` / ``warn`` / ``error``), the findings, and the inputs
    they were judged at."""

    level: str
    findings: tuple[Finding, ...]
    primal_tol: float
    dual_tol: float
    window: tuple[float, float]
    bound_scale: int
    objective_scale: int
    bound_source: str

    def finding(self, group: str) -> Finding | None:
        """The first finding of ``group`` (``None`` when there is none)."""
        for f in self.findings:
            if f.group == group:
                return f
        return None


def _ext(ranges: RangeReport, group: str) -> dict[str, Extreme]:
    e = ranges.extremes or {}
    return dict(e.get(group, {}))


def _log10_ratio(hi: float, lo: float) -> float:
    return math.log10(hi / lo) if lo > 0.0 and hi > 0.0 else 0.0


def _assess_bound_rhs(
    ranges: RangeReport,
    ranges_pre: RangeReport | None,
    n: int,
    source: str,
    tol: float,
    warn_decades: float,
    lo_factor: float,
    hi_factor: float,
) -> Finding | None:
    pool = _pool(ranges)
    if pool is None:
        return None
    b_lo, b_hi = pool
    # The verdict judges against the hard window (no small-end preference).
    window = bound_scale_window(
        b_lo, b_hi, tol, lo_factor=lo_factor, hi_factor=hi_factor, preferred_lo_factor=lo_factor
    )
    s_lo = math.ldexp(b_lo, n)
    s_hi = math.ldexp(b_hi, n)
    decades = _log10_ratio(b_hi, b_lo)
    extremes = _ext(ranges, "bound_rhs")
    lo_end = extremes.get("min")
    # Model units per reported unit at the smallest limit (1 / Layer 2 factor).
    ratio = 1.0
    if lo_end is not None and lo_end.post > 0.0 and math.isfinite(lo_end.model):
        ratio = lo_end.model / lo_end.post
    model_lo = b_lo * ratio
    numbers: dict[str, float] = {
        "decades": decades,
        "b_lo": b_lo,
        "b_hi": b_hi,
        "s_lo": s_lo,
        "s_hi": s_hi,
        "r_lo": tol / s_lo,
        "violation_model": math.ldexp(tol, -n) * ratio,
        "window_lo": window.lo_edge,
        "window_hi": window.hi_edge,
        "n_min": float(window.n_min),
        "n_max": float(window.n_max),
        "n_used": float(n),
        "fits": 1.0 if window.fits else 0.0,
    }
    # What fitting the largest value costs the smallest limit: the exponent
    # that puts the large end on the upper edge (within HiGHS' range).
    n_fit_large = max(min(window.n_max_unclamped, USER_SCALE_CLAMP_HI), USER_SCALE_CLAMP_LO)
    numbers["n_fit_large"] = float(n_fit_large)
    numbers["fit_large_violation_model"] = math.ldexp(tol, -n_fit_large) * ratio
    numbers["fit_large_violation_ratio"] = numbers["fit_large_violation_model"] / model_lo
    pre_fits = None
    pre_decades = None
    if ranges_pre is not None:
        pre_pool = _pool(ranges_pre)
        if pre_pool is not None:
            pre_decades = _log10_ratio(pre_pool[1], pre_pool[0])
            numbers["decades_pre"] = pre_decades
            pre_fits = bound_scale_window(
                pre_pool[0],
                pre_pool[1],
                tol,
                lo_factor=lo_factor,
                hi_factor=hi_factor,
                preferred_lo_factor=lo_factor,
            ).fits
    out_of_window = s_lo < window.lo_edge or s_hi > window.hi_edge
    caused = False
    if out_of_window:
        level = "error"
        if not window.fits:
            kind = window.reason
            caused = bool(pre_fits)
        else:
            kind = {"manual": "manual_out_of_window", "external": "external_out_of_window"}.get(
                source, "unscaled_out_of_window" if source == "unscaled" else "out_of_window"
            )
        if decades > warn_decades and kind != "spread":
            kind = f"spread+{kind}"
    elif decades > warn_decades:
        level, kind = "warn", "spread"
        caused = pre_decades is not None and pre_decades <= warn_decades
    else:
        level, kind = "ok", "ok"
    return Finding("bound_rhs", level, kind, numbers, extremes, caused)


def _assess_cost(
    ranges: RangeReport, n_obj: int, n_bnd: int, tol_d: float, warn_decades: float
) -> Finding | None:
    cont = ranges.cost_continuous
    if cont is None and ranges.integer_cost is None:
        cont = ranges.cost if _finite_pos(ranges.cost[0]) else None
    los: list[float] = []
    his: list[float] = []
    if cont is not None:
        los.append(math.ldexp(cont[0], n_obj))
        his.append(math.ldexp(cont[1], n_obj))
    if ranges.integer_cost is not None:
        los.append(math.ldexp(ranges.integer_cost[0], n_obj + n_bnd))
        his.append(math.ldexp(ranges.integer_cost[1], n_obj + n_bnd))
    if not los:
        return None
    c_lo, c_hi = min(los), max(his)
    decades = _log10_ratio(c_hi, c_lo)
    kinds = []
    if decades > warn_decades:
        kinds.append("spread")
    if c_lo < 10.0 * tol_d:
        kinds.append("small_cost")
    extremes = _ext(ranges, "cost")
    ints = _ext(ranges, "integer_cost")
    if ranges.integer_cost is not None:
        if cont is None or math.ldexp(ranges.integer_cost[0], n_obj + n_bnd) == c_lo:
            if "min" in ints:
                extremes["min"] = ints["min"]
        if cont is None or math.ldexp(ranges.integer_cost[1], n_obj + n_bnd) == c_hi:
            if "max" in ints:
                extremes["max"] = ints["max"]
    numbers = {"decades": decades, "s_lo": c_lo, "s_hi": c_hi, "small_edge": 10.0 * tol_d}
    level = "warn" if kinds else "ok"
    return Finding("cost", level, "+".join(kinds) or "ok", numbers, extremes)


def _assess_matrix(ranges: RangeReport, n_bnd: int, small: float, large: float) -> Finding | None:
    cont = ranges.matrix_continuous
    if cont is None and ranges.integer_matrix is None:
        cont = ranges.matrix if _finite_pos(ranges.matrix[0]) else None
    ends: list[tuple[float, float, str]] = []
    if cont is not None:
        ends.append((cont[0], cont[1], "matrix"))
    if ranges.integer_matrix is not None:
        ends.append(
            (
                math.ldexp(ranges.integer_matrix[0], n_bnd),
                math.ldexp(ranges.integer_matrix[1], n_bnd),
                "integer_matrix",
            )
        )
    if not ends:
        return None
    lo, lo_group = min((e[0], e[2]) for e in ends)
    hi, hi_group = max((e[1], e[2]) for e in ends)
    extremes = {}
    lo_ext = _ext(ranges, lo_group).get("min")
    hi_ext = _ext(ranges, hi_group).get("max")
    if lo_ext is not None:
        extremes["min"] = lo_ext
    if hi_ext is not None:
        extremes["max"] = hi_ext
    numbers = {
        "decades": _log10_ratio(hi, lo),
        "s_lo": lo,
        "s_hi": hi,
        "small_matrix": small,
        "large_matrix": large,
    }
    kinds = []
    level = "ok"
    # Both warn: HiGHS drops the small terms and refuses a model with a large
    # one itself, with its own message; the finding names it ahead of that.
    if hi >= large:
        kinds.append("refused")
        level = "warn"
    if lo <= small:
        kinds.append("dropped")
        level = "warn"
    return Finding("matrix", level, "+".join(kinds) or "ok", numbers, extremes)


def _assess_info(ranges: RangeReport) -> list[Finding]:
    out: list[Finding] = []
    if ranges.bound_rhs_residue is not None and ranges.bound_rhs_residue[0] > 0:
        count, largest = ranges.bound_rhs_residue
        out.append(
            Finding(
                "info",
                "info",
                "residue",
                {
                    "count": float(count),
                    "largest_model": float(largest),
                    "floor": float(ranges.residue_floor or 0.0),
                },
                _ext(ranges, "residue"),
            )
        )
    if ranges.bound_rhs_beyond_infinite:
        out.append(
            Finding(
                "info",
                "info",
                "beyond_infinite",
                {
                    "count": float(ranges.bound_rhs_beyond_infinite),
                    "infinite_bound": float(ranges.infinite_bound or 0.0),
                },
            )
        )
    if ranges.integer_bound is not None:
        out.append(
            Finding(
                "info",
                "info",
                "integer_bounds",
                {"lo": ranges.integer_bound[0], "hi": ranges.integer_bound[1]},
                _ext(ranges, "integer_bound"),
            )
        )
    return out


def assess_scaling(
    ranges: RangeReport,
    plan: Layer3Plan,
    *,
    primal_tol: float,
    dual_tol: float,
    warn_decades: float = 13.0,
    small_matrix: float = HIGHS_SMALL_MATRIX_VALUE,
    large_matrix: float = HIGHS_LARGE_MATRIX_VALUE,
    apply_user_scale: bool = True,
    ranges_pre: RangeReport | None = None,
    window_lo_factor: float = _WINDOW_LO_FACTOR,
    window_hi_factor: float = _WINDOW_HI_FACTOR,
) -> ScalingVerdict:
    """Judge the LP a solver will receive: ``ranges`` (the readout of that
    LP before ``user_*_scale``) under ``plan``'s exponents.

    Pure: no side effects, no solver call.  Rules (``tol`` = ``primal_tol``):

    * **bound / RHS** — the pool ``[b_lo*, b_hi]`` (the meaningful readout,
      else the raw bound ∪ RHS range) scaled by
      ``2**user_bound_scale`` must lie inside ``[window_lo_factor·tol,
      window_hi_factor·tol]`` (error otherwise); a spread of more than
      ``warn_decades`` warns.  ``numbers``: ``decades`` (and
      ``decades_pre``), ``b_lo`` / ``b_hi``, scaled ``s_lo`` / ``s_hi``,
      ``r_lo`` = ``tol / s_lo`` (fraction of the smallest limit the solver may
      violate), ``violation_model`` (that amount in model units),
      ``window_lo`` / ``window_hi``, ``n_min`` / ``n_max`` / ``n_used`` /
      ``fits``, and ``n_fit_large`` / ``fit_large_violation_model`` /
      ``fit_large_violation_ratio`` (what fitting the largest value would
      cost the smallest limit).
    * **cost** — warn only: spread over ``warn_decades``, or the smallest
      scaled cost below ``10·dual_tol``.  Continuous-column costs are scaled
      by ``2**N_obj``, integer-column costs by ``2**(N_obj + N_bnd)``.
    * **matrix** — the continuous-column terms as they are, the
      integer-column terms ``× 2**N_bnd``: a term at or below
      ``small_matrix`` warns (the solver drops it), one at or above
      ``large_matrix`` warns too (kind ``refused``: HiGHS will refuse the
      model and report that itself).  The spread is reported only.
    * **info** — residue left out by the floor, values at or above the
      solver infinity, integer-column bounds.

    ``apply_user_scale=False`` judges the LP at ``N = 0`` for both exponents
    (a solver that does not receive ``user_*_scale``).  ``ranges_pre``, when
    given, sets :attr:`Finding.caused_by_prescale`.  The caller decides what a
    level means (raise, warn, ignore).
    """
    if apply_user_scale:
        n_bnd = int(plan.user_bound_scale)
        n_obj = int(plan.user_objective_scale)
        source = plan.bound_source
    else:
        n_bnd = n_obj = 0
        source = "unscaled"
    findings: list[Finding] = []
    b = _assess_bound_rhs(
        ranges,
        ranges_pre,
        n_bnd,
        source,
        float(primal_tol),
        warn_decades,
        window_lo_factor,
        window_hi_factor,
    )
    if b is not None:
        findings.append(b)
    c = _assess_cost(ranges, n_obj, n_bnd, float(dual_tol), warn_decades)
    if c is not None:
        findings.append(c)
    m = _assess_matrix(ranges, n_bnd, small_matrix, large_matrix)
    if m is not None:
        findings.append(m)
    findings.extend(_assess_info(ranges))
    worst = max((f.level for f in findings), key=lambda lv: _LEVEL_ORDER[lv], default="ok")
    level = "ok" if worst == "info" else worst
    return ScalingVerdict(
        level=level,
        findings=tuple(findings),
        primal_tol=float(primal_tol),
        dual_tol=float(dual_tol),
        window=(window_lo_factor * float(primal_tol), window_hi_factor * float(primal_tol)),
        bound_scale=n_bnd,
        objective_scale=n_obj,
        bound_source=source,
    )


__all__ = [
    "Finding",
    "HIGHS_LARGE_MATRIX_VALUE",
    "HIGHS_SMALL_MATRIX_VALUE",
    "Layer3Plan",
    "ScaleWindow",
    "ScalingVerdict",
    "apply_scaling",
    "assess_scaling",
    "bound_scale_window",
    "recommend_scaling",
]
