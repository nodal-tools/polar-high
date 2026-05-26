"""polar-high autoscale package — Layer 1 detect + Layer 3 recommend.

Public surface:

* :func:`detect_ranges` — compute the four-range (Matrix / Cost / Bound
  / RHS) report from a built :class:`polar_high.Problem` or from a
  :class:`polar_high.Solution` carrying ``streamed_lp_ranges``.
* :func:`recommend_scaling` — derive ``user_objective_scale`` and
  ``user_bound_scale`` exponents from a :class:`RangeReport`, with
  per-axis precedence-respect, manual override, and D's geometric-
  centering escape branch for severe asymmetric-bound LPs.
* :func:`apply_scaling` — write the :class:`Layer3Plan` back onto a
  ``Problem`` via :meth:`Problem.set_solver_options`.
* :class:`ScalingMode` / :class:`ScalingConfig` — library-level policy
  primitives.  Callers decide what each mode does in practice.

Layer 2 (semantic per-quantity scaling) is FlexTool-specific and lives
in the FlexTool repo; polar-high does not know about it.

The legacy ``polar_high.engine._recommend_user_bound_scale`` and the
``auto_user_bound_scale=True`` default path on :class:`Problem` have
been retired in favour of this package.
"""

from ._config import (
    USER_SCALE_CLAMP_HI,
    USER_SCALE_CLAMP_LO,
    ScalingConfig,
    ScalingMode,
    mode_enables_layer1,
    mode_enables_layer3,
)
from ._layer3 import Layer3Plan, apply_scaling, recommend_scaling
from ._precedence import get_explicit_option, has_explicit_option
from ._ranges import (
    RangeReport,
    detect_ranges,
    ranges_from_arrays,
    ranges_from_streamed,
)

__all__ = [
    "Layer3Plan",
    "RangeReport",
    "ScalingConfig",
    "ScalingMode",
    "USER_SCALE_CLAMP_HI",
    "USER_SCALE_CLAMP_LO",
    "apply_scaling",
    "detect_ranges",
    "get_explicit_option",
    "has_explicit_option",
    "mode_enables_layer1",
    "mode_enables_layer3",
    "ranges_from_arrays",
    "ranges_from_streamed",
    "recommend_scaling",
]
