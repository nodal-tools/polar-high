"""Configuration for the polar-high autoscale package.

Library-level scaling primitives: a :class:`ScalingMode` enum, a
:class:`ScalingConfig` dataclass, and predicate helpers
(:func:`mode_enables_layer1`, :func:`mode_enables_layer3`).  The library
does not commit to a specific policy per mode — callers (e.g. FlexTool)
decide what each mode enables in their orchestration layer.  This
module just declares the enum, the config record, and a conservative
default mapping the helpers expose for callers that want it.

Layer 2 (semantic per-quantity scaling) is FlexTool-specific and lives
in the FlexTool repo; polar-high knows nothing about it.  ``BASIC`` /
``FULL`` are kept in the enum so the library API is symmetric — a
library caller that doesn't model Layer 2 can still use ``BASIC`` to
mean "Layer 1 + Layer 3 only".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# Default trigger threshold: any single-group or cross-group max/min
# ratio greater than ``10 ** _DEFAULT_THRESHOLD_DECADES`` is considered
# worth triggering autoscaling.  Nine decades is the conservative
# operational pain point identified in the H2_trade handoff.
_DEFAULT_THRESHOLD_DECADES = 9.0


# HiGHS' option-range bounds for ``user_bound_scale`` / ``user_objective_scale``.
# Defensive clamp; HiGHS rejects values outside [-30, 30].
USER_SCALE_CLAMP_LO: int = -30
USER_SCALE_CLAMP_HI: int = 30


class ScalingMode(StrEnum):
    """Autoscaler policy modes.

    The library uses these as named buckets; specific callers decide
    what each mode does in practice via the
    :func:`mode_enables_layer1` / :func:`mode_enables_layer3`
    predicates (or by ignoring the enum entirely and reading
    :attr:`ScalingConfig.layer1_enabled` / ``layer3_enabled``
    directly).

    The default mapping the predicate helpers return is:

    * ``OFF`` — autoscaler is fully disabled.  No detection, no
      scaling options applied.
    * ``SOLVER_ONLY`` — leave any solver-native scaling
      (``simplex_scale_strategy``) to the caller, but skip Layer 1
      detection and Layer 3 recommendation.
    * ``BASIC`` — Layer 1 detection + Layer 3 recommendation.  No
      Layer 2 (semantic per-quantity bucketing) — that lives outside
      polar-high.
    * ``FULL`` — same as BASIC for the library; callers that
      implement Layer 2 add it on top.
    """

    OFF = "off"
    SOLVER_ONLY = "solver_only"
    BASIC = "basic"
    FULL = "full"


# Default per-mode enablement.  Callers can override via the
# ``layer1_enabled`` / ``layer3_enabled`` overrides on
# :class:`ScalingConfig`.
_LAYER1_DEFAULT: dict[ScalingMode, bool] = {
    ScalingMode.OFF: False,
    ScalingMode.SOLVER_ONLY: False,
    ScalingMode.BASIC: True,
    ScalingMode.FULL: True,
}
_LAYER3_DEFAULT: dict[ScalingMode, bool] = {
    ScalingMode.OFF: False,
    ScalingMode.SOLVER_ONLY: False,
    ScalingMode.BASIC: True,
    ScalingMode.FULL: True,
}


def mode_enables_layer1(mode: ScalingMode) -> bool:
    """Predicate: does the default policy enable Layer 1 detection for ``mode``?"""
    return _LAYER1_DEFAULT[mode]


def mode_enables_layer3(mode: ScalingMode) -> bool:
    """Predicate: does the default policy enable Layer 3 recommendation for ``mode``?"""
    return _LAYER3_DEFAULT[mode]


@dataclass(frozen=True)
class ScalingConfig:
    """Caller-facing autoscale configuration.

    Parameters
    ----------
    mode:
        Named policy mode (see :class:`ScalingMode`).  The library
        does not consult this directly — callers use
        :func:`mode_enables_layer1` / :func:`mode_enables_layer3`
        when they want the default mapping.
    threshold_decades:
        Layer 1 raises ``trigger=True`` when any single-group max/min
        ratio, or the cross-group max/min ratio, exceeds
        ``10 ** threshold_decades``.
    user_bound_scale:
        Manual override for the Layer 3 ``user_bound_scale`` HiGHS
        option.  When set, :func:`recommend_scaling` uses this integer
        verbatim and skips its own bound-axis recommendation.  Default
        ``None`` (let Layer 3 decide).
    user_objective_scale:
        Manual override for the Layer 3 ``user_objective_scale`` HiGHS
        option.  Same semantics as ``user_bound_scale`` but for the
        objective-axis recommendation.  Default ``None``.
    report_yaml_path:
        Where a caller wishes to write an audit YAML.  Library does not
        read or write this; it's a transport for caller-side reporters.
    """

    mode: ScalingMode = ScalingMode.BASIC
    threshold_decades: float = _DEFAULT_THRESHOLD_DECADES
    user_bound_scale: int | None = None
    user_objective_scale: int | None = None
    report_yaml_path: Path | None = None


__all__ = [
    "ScalingConfig",
    "ScalingMode",
    "USER_SCALE_CLAMP_HI",
    "USER_SCALE_CLAMP_LO",
    "mode_enables_layer1",
    "mode_enables_layer3",
]
