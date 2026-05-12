"""Phase 2 tests for the HiGHS adapter behind ``polar_high.solvers``.

These verify that the new dispatch path produces results equivalent to
the legacy ``Problem.solve()`` on the toy LP. The streaming and
non-streaming paths inside ``Problem.solve()`` are already covered by
``test_streaming_parity.py``; here we focus on the new
``solvers.solve(problem, solver_name='highs')`` route.
"""

from __future__ import annotations

from toy_data import make_toy_data
from toy_model import build_dispatch

from polar_high import Problem
from polar_high.solvers import SolverResult, SolverStatus, solve


def test_highs_via_new_dispatch_solves_toy_lp() -> None:
    """``solvers.solve(problem, solver_name='highs')`` returns a populated
    :class:`SolverResult` whose objective matches the hand-verifiable
    toy-LP optimum (6500.0)."""
    pb = Problem()
    build_dispatch(pb, make_toy_data())

    result = solve(pb, solver_name="highs")

    assert isinstance(result, SolverResult)
    assert result.solver_name == "highs"
    assert result.status == SolverStatus.OPTIMAL
    assert result.objective is not None
    assert abs(result.objective - 6500.0) < 1e-6
    # LP solve → both primal and dual dicts populated.
    assert result.primal is not None
    assert result.dual is not None
    assert len(result.primal) > 0
    assert len(result.dual) > 0


def test_highs_via_new_dispatch_matches_problem_solve() -> None:
    """The new dispatch and the legacy ``Problem.solve()`` paths must
    return the same objective on the toy LP, within numerical noise."""
    pb_legacy = Problem()
    build_dispatch(pb_legacy, make_toy_data())
    sol = pb_legacy.solve()
    assert sol.optimal

    pb_new = Problem()
    build_dispatch(pb_new, make_toy_data())
    result = solve(pb_new, solver_name="highs")
    assert result.status == SolverStatus.OPTIMAL

    assert result.objective is not None
    assert abs(result.objective - sol.obj) < 1e-9
