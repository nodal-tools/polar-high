"""Common solver-dispatch types and exception hierarchy.

This module is intentionally tiny and dependency-free: it is imported eagerly
by ``polar_high.solvers.__init__`` and re-exported from the top-level
``polar_high`` package, so it must not pull in any solver-specific wrappers.

See ``specs/polar-high-multi-solver-handoff.md`` (Step 1) for the design.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class IOMode(StrEnum):
    """How the model is handed to the underlying solver."""

    DIRECT = "direct"  # in-memory, via solver's Python API
    MPS = "mps"  # write MPS file, hand to solver
    LP = "lp"  # write LP file, hand to solver


class SolverStatus(StrEnum):
    """Normalised solver termination status."""

    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    TIME_LIMIT = "time_limit"
    INTERRUPTED = "interrupted"
    OTHER = "other"


@dataclass
class SolverResult:
    """Solver-agnostic result returned by ``polar_high.solvers.solve``."""

    status: SolverStatus
    objective: float | None
    primal: dict | None  # var_name -> value
    dual: dict | None  # constraint_name -> value (LP only, None for MIP)
    solver_name: str
    raw_status: Any  # solver-native status code, for debugging


# Exception hierarchy — keep these distinct so callers can handle them
# differently.
class SolverNotAvailableError(RuntimeError):
    """The requested solver's Python wrapper is not installed."""


class LicenseError(RuntimeError):
    """Solver is installed but its license check failed."""


class SolverError(RuntimeError):
    """Solver ran but reported an error (not license-related)."""


__all__ = [
    "IOMode",
    "SolverStatus",
    "SolverResult",
    "SolverNotAvailableError",
    "LicenseError",
    "SolverError",
]
