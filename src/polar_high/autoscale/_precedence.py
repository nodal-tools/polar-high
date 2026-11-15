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

from collections.abc import Mapping
from typing import Any

# HiGHS defaults the scaling checks fall back to when an option is not set.
HIGHS_DEFAULT_FEASIBILITY_TOLERANCE = 1e-7
HIGHS_DEFAULT_KKT_TOLERANCE = 1e-7
HIGHS_DEFAULT_INFINITE_BOUND = 1e20


def get_explicit_option(problem: Any, option_name: str) -> Any | None:
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


def _as_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default  # NaN -> default


def feasibility_tolerances_from_options(
    options: Mapping[str, Any] | None, *, kkt_override: bool = True
) -> tuple[float, float]:
    """``(primal, dual)`` feasibility tolerances for an options mapping: the
    explicit ``primal_feasibility_tolerance`` / ``dual_feasibility_tolerance``
    (else HiGHS' default ``1e-7``).  With ``kkt_override`` (default) both are
    replaced by ``kkt_tolerance`` when that is set to anything other than its
    default — HiGHS' rule for its KKT checks, which the solve diagnostics
    follow.  ``kkt_override=False`` gives the tolerances the simplex enforces
    on the bounds, which the scaling checks (Layer 3 window, residue floor)
    use."""
    opts = options or {}
    p = opts.get("primal_feasibility_tolerance")
    d = opts.get("dual_feasibility_tolerance")
    tol_p = (
        HIGHS_DEFAULT_FEASIBILITY_TOLERANCE
        if p is None
        else _as_float(p, HIGHS_DEFAULT_FEASIBILITY_TOLERANCE)
    )
    tol_d = (
        HIGHS_DEFAULT_FEASIBILITY_TOLERANCE
        if d is None
        else _as_float(d, HIGHS_DEFAULT_FEASIBILITY_TOLERANCE)
    )
    k = opts.get("kkt_tolerance") if kkt_override else None
    if k is not None:
        kkt = _as_float(k, HIGHS_DEFAULT_KKT_TOLERANCE)
        if kkt != HIGHS_DEFAULT_KKT_TOLERANCE:
            tol_p = tol_d = kkt
    return tol_p, tol_d


def infinite_bound_from_options(options: Mapping[str, Any] | None) -> float:
    """HiGHS' ``infinite_bound`` for an options mapping (default ``1e20``):
    a bound at or above it is no limit to the solver."""
    v = (options or {}).get("infinite_bound")
    return HIGHS_DEFAULT_INFINITE_BOUND if v is None else _as_float(v, HIGHS_DEFAULT_INFINITE_BOUND)


def _explicit_options(problem: Any, names: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if problem is None:
        return out
    for name in names:
        v = get_explicit_option(problem, name)
        if v is not None:
            out[name] = v
    return out


def resolve_feasibility_tolerances(
    problem: Any, *, kkt_override: bool = True
) -> tuple[float, float]:
    """``(primal, dual)`` feasibility tolerances set on ``problem`` (see
    :func:`feasibility_tolerances_from_options`, including
    ``kkt_override``); HiGHS' defaults when ``problem`` is ``None`` or sets
    none."""
    return feasibility_tolerances_from_options(
        _explicit_options(
            problem,
            ("primal_feasibility_tolerance", "dual_feasibility_tolerance", "kkt_tolerance"),
        ),
        kkt_override=kkt_override,
    )


def resolve_infinite_bound(problem: Any) -> float:
    """``infinite_bound`` set on ``problem``, else HiGHS' default ``1e20``."""
    return infinite_bound_from_options(_explicit_options(problem, ("infinite_bound",)))


__all__ = [
    "HIGHS_DEFAULT_FEASIBILITY_TOLERANCE",
    "HIGHS_DEFAULT_INFINITE_BOUND",
    "HIGHS_DEFAULT_KKT_TOLERANCE",
    "feasibility_tolerances_from_options",
    "get_explicit_option",
    "has_explicit_option",
    "infinite_bound_from_options",
    "resolve_feasibility_tolerances",
    "resolve_infinite_bound",
]
