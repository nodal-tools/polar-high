"""polar-high network LP in ``save_memory=True`` mode.

Same build as :mod:`models.polar_net`; only the solve path differs.
See :mod:`models.polar_sm` for the trade-off this exposes.
"""

from __future__ import annotations

from polar_high import Problem

from .polar_net import build  # re-exported for the harness

__all__ = ["build", "solve"]


def solve(model: Problem, time_limit: float | None = None) -> tuple[bool, float]:
    options = {"time_limit": float(time_limit)} if time_limit is not None else None
    sol = model.solve(options=options, save_memory=True)
    return bool(sol.optimal), float(sol.obj)
