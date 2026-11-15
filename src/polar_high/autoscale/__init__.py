"""polar-high autoscale package — Layer 1 detect + Layer 3 recommend.

Public surface:

* :func:`detect_ranges` — compute the four-range (Matrix / Cost / Bound
  / RHS) report from a built :class:`polar_high.Problem` or from a
  :class:`polar_high.Solution` carrying ``streamed_lp_ranges``.
* :func:`recommend_scaling` — derive ``user_objective_scale`` and
  ``user_bound_scale`` exponents from a :class:`RangeReport`, with
  per-axis precedence-respect, manual override, and the tolerance
  window that keeps the smallest meaningful limit enforceable.
* :func:`apply_scaling` — write the :class:`Layer3Plan` back onto a
  ``Problem`` via :meth:`Problem.set_solver_options`.
* :func:`assess_scaling` — judge a report under a plan against the
  solver's tolerance window (a :class:`ScalingVerdict`; the caller decides
  what a warning or an error means); :func:`bound_scale_window` gives the
  window of ``user_bound_scale`` exponents for a bound / RHS pool.
* :class:`ScalingMode` / :class:`ScalingConfig` — library-level policy
  primitives.  Callers decide what each mode does in practice.

Layer 2 (semantic per-quantity scaling) is FlexTool-specific and lives
in the FlexTool repo; polar-high does not know about it.

The legacy ``polar_high.engine._recommend_user_bound_scale`` and the
``auto_user_bound_scale=True`` default path on :class:`Problem` have
been retired in favour of this package.
"""

from ._coef_walk import FlooredMinMax, FlooredMinMaxAbsReducer
from ._config import (
    RESIDUE_FLOOR_FACTOR,
    USER_SCALE_CLAMP_HI,
    USER_SCALE_CLAMP_LO,
    ScalingConfig,
    ScalingMode,
    mode_enables_layer1,
    mode_enables_layer3,
)
from ._layer3 import (
    HIGHS_LARGE_MATRIX_VALUE,
    HIGHS_SMALL_MATRIX_VALUE,
    Finding,
    Layer3Plan,
    ScaleWindow,
    ScalingVerdict,
    apply_scaling,
    assess_scaling,
    bound_scale_window,
    recommend_scaling,
)
from ._precedence import (
    feasibility_tolerances_from_options,
    get_explicit_option,
    has_explicit_option,
    infinite_bound_from_options,
    resolve_feasibility_tolerances,
    resolve_infinite_bound,
)
from ._ranges import (
    Extreme,
    RangeReport,
    detect_ranges,
    ranges_from_arrays,
    ranges_from_streamed,
)

__all__ = [
    "Extreme",
    "Finding",
    "FlooredMinMax",
    "FlooredMinMaxAbsReducer",
    "HIGHS_LARGE_MATRIX_VALUE",
    "HIGHS_SMALL_MATRIX_VALUE",
    "Layer3Plan",
    "RESIDUE_FLOOR_FACTOR",
    "RangeReport",
    "ScaleWindow",
    "ScalingConfig",
    "ScalingMode",
    "ScalingVerdict",
    "USER_SCALE_CLAMP_HI",
    "USER_SCALE_CLAMP_LO",
    "apply_scaling",
    "assess_scaling",
    "bound_scale_window",
    "detect_ranges",
    "feasibility_tolerances_from_options",
    "get_explicit_option",
    "has_explicit_option",
    "infinite_bound_from_options",
    "mode_enables_layer1",
    "mode_enables_layer3",
    "ranges_from_arrays",
    "ranges_from_streamed",
    "recommend_scaling",
    "resolve_feasibility_tolerances",
    "resolve_infinite_bound",
]
