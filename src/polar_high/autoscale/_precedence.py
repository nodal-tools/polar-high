"""Precedence-check helpers for the autoscale package.

Before applying Layer 3's recommendation, callers should check whether
``Problem`` already has ``user_bound_scale`` or ``user_objective_scale``
set explicitly (either by an earlier ``set_solver_options`` call, by
the per-solve ``options=`` kwarg, or by a ``highs.opt`` file the
caller has loaded into ``_solver_options``).  If so, the
recommendation is skipped for that axis only — the caller's explicit
value wins.

The check operates on :meth:`polar_high.Problem.get_solver_option`
(introduced alongside this module), falling back to inspecting
``Problem._solver_options`` directly for callers that wrap a duck-typed
problem.  Either path yields the same answer.
"""
from __future__ import annotations

from typing import Any, Optional


def get_explicit_option(problem: Any, option_name: str) -> Optional[Any]:
    """Return the caller-set value for ``option_name`` on ``problem``.

    Priority:

    1. ``problem.get_solver_option(option_name)`` if the method exists.
    2. ``problem._solver_options[option_name]`` direct read on the
       (private but stable) attribute.
    3. ``None`` if neither path resolves.

    The function is intentionally lenient about the problem's surface
    so it works against test doubles that don't subclass the full
    ``Problem`` class.
    """
    getter = getattr(problem, "get_solver_option", None)
    if callable(getter):
        try:
            return getter(option_name)
        except Exception:
            pass
    opts = getattr(problem, "_solver_options", None)
    if isinstance(opts, dict) and option_name in opts:
        return opts[option_name]
    return None


def has_explicit_option(problem: Any, option_name: str) -> bool:
    """Return ``True`` iff ``option_name`` has a caller-set value.

    Treats ``None`` as "not set" — polar-high stores option values as
    typed scalars (int / float / str / bool), so a literal ``None``
    cannot be a caller-set value.
    """
    return get_explicit_option(problem, option_name) is not None


__all__ = ["get_explicit_option", "has_explicit_option"]
