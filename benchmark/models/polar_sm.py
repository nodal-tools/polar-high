"""polar-high in ``save_memory=True`` mode.

Same build as :mod:`models.polar` (regular mode); only the solve path
differs. Run alongside ``polar`` to show the warm-restart vs one-shot
trade-off on the same hardware in the same harness.

``save_memory=True``:

* Drops polar's Python LP source (lazy plans, Param frames, caller-side
  bounds/cost arrays, ``col_names``/``row_names`` lists) once HiGHS has
  copied them.
* Disk-roundtrips HiGHS via a temp MPS file to reset HiGHS's
  ``addRows``-induced allocator slack.

Cost: slower wall-clock (no warm restart possible, plus MPS write/read).
Benefit: lower peak RSS in full-HiGHS-solve cells.
"""

from __future__ import annotations

from polar_high import Problem

from .polar import build  # re-exported so the harness imports work uniformly

__all__ = ["build", "solve"]


def solve(model: Problem, time_limit: float | None = None) -> tuple[bool, float]:
    options = {"time_limit": float(time_limit)} if time_limit is not None else None
    sol = model.solve(options=options, save_memory=True)
    return bool(sol.optimal), float(sol.obj)
