"""Phase 8 tests for the Xpress direct adapter.

The whole module is skipped at import time when ``xpress`` is not
installed — the ``[xpress]`` optional extra is not present in CI.

License-error translation (any Xpress vendor exception whose message
contains a licence keyword -> :class:`LicenseError`) is *manual-only*
per the implementation plan (it requires physically removing or
corrupting a real licence file); the procedure is documented at the
bottom of this file.
"""

from __future__ import annotations

import pytest

# Module-level skip — collected as one "skipped" entry rather than per
# test.  Keep the import inside the skip check so the module body never
# touches xpress when it isn't installed.
xpress = pytest.importorskip("xpress")

import numpy as np  # noqa: E402
from toy_data import make_toy_data  # noqa: E402
from toy_model import build_dispatch  # noqa: E402

from polar_high import Problem  # noqa: E402
from polar_high.solvers import SolverResult, SolverStatus, solve  # noqa: E402
from polar_high.solvers._lp_view import LpView  # noqa: E402
from polar_high.solvers._xpress import run as xpress_run  # noqa: E402


def _toy_lp_problem() -> Problem:
    pb = Problem()
    build_dispatch(pb, make_toy_data())
    return pb


def _toy_mip_view() -> LpView:
    """Hand-built MIP :class:`LpView` (matches the gurobi/copt/cplex toys).

    Maximise ``3x + 2y`` subject to ``x + y <= 4`` and ``x + 3y <= 6``,
    with ``x, y`` non-negative integers in ``[0, 10]``.  Optimum:
    ``x=4, y=0, obj=12``.
    """
    col_obj = np.array([3.0, 2.0])
    col_lb = np.array([0.0, 0.0])
    col_ub = np.array([10.0, 10.0])
    integrality = np.array([1, 1], dtype=np.int8)
    row_lb = np.array([-np.inf, -np.inf])
    row_ub = np.array([4.0, 6.0])
    # CSC: col 0 (x) → rows {0,1} coef {1,1}; col 1 (y) → rows {0,1} coef {1,3}
    a_start = np.array([0, 2, 4], dtype=np.int32)
    a_index = np.array([0, 1, 0, 1], dtype=np.int32)
    a_value = np.array([1.0, 1.0, 1.0, 3.0])
    return LpView(
        n_cols=2,
        n_rows=2,
        col_obj=col_obj,
        col_lb=col_lb,
        col_ub=col_ub,
        integrality=integrality,
        row_lb=row_lb,
        row_ub=row_ub,
        a_start=a_start,
        a_index=a_index,
        a_value=a_value,
        col_names=["x", "y"],
        row_names=["c1", "c2"],
        sense="max",
        obj_offset=0.0,
    )


def _hand_built_ranged_view() -> LpView:
    """Hand-built :class:`LpView` with a single ranged constraint.

    Layout::

        columns: x (lb=0, ub=20),  y (lb=0, ub=20)
        rows:    r0 (R): 5 <= x + y <= 10

    Objective: maximise ``x + y``.  Optimum sits on the upper face
    (``x + y == 10``) — the lower face (5) is non-binding.  This test
    locks the Xpress range convention documented in ``_xpress.py``.
    """
    col_obj = np.array([1.0, 1.0])
    col_lb = np.array([0.0, 0.0])
    col_ub = np.array([20.0, 20.0])
    row_lb = np.array([5.0])
    row_ub = np.array([10.0])
    a_start = np.array([0, 1, 2], dtype=np.int32)
    a_index = np.array([0, 0], dtype=np.int32)
    a_value = np.array([1.0, 1.0])
    return LpView(
        n_cols=2,
        n_rows=1,
        col_obj=col_obj,
        col_lb=col_lb,
        col_ub=col_ub,
        integrality=None,
        row_lb=row_lb,
        row_ub=row_ub,
        a_start=a_start,
        a_index=a_index,
        a_value=a_value,
        col_names=["x", "y"],
        row_names=["ranged"],
        sense="max",
        obj_offset=0.0,
    )


def test_xpress_solves_toy_lp_returns_optimal() -> None:
    """Toy LP solves to optimal and returns a fully-populated result."""
    pb = _toy_lp_problem()
    result = solve(pb, solver_name="xpress")

    assert isinstance(result, SolverResult)
    assert result.solver_name == "xpress"
    assert result.status == SolverStatus.OPTIMAL
    assert result.objective is not None
    assert abs(result.objective - 6500.0) < 1e-6
    assert result.primal is not None and len(result.primal) > 0
    # LP -> dual must be populated.
    assert result.dual is not None and len(result.dual) > 0


def test_xpress_solves_toy_mip_respects_integrality() -> None:
    """MIP variables come back integral and dual is None."""
    view = _toy_mip_view()
    result = xpress_run(view)

    assert result.status == SolverStatus.OPTIMAL
    assert result.objective is not None
    assert abs(result.objective - 12.0) < 1e-6
    assert result.primal is not None
    for nm, val in result.primal.items():
        assert abs(val - round(val)) < 1e-6, f"{nm}={val} not integral"
    # MIP -> dual must be None (matches the other direct adapters).
    assert result.dual is None


def test_xpress_objective_matches_highs() -> None:
    """Same LP solved by both backends agrees within 1e-6."""
    pb_h = _toy_lp_problem()
    pb_x = _toy_lp_problem()

    r_h = solve(pb_h, solver_name="highs")
    r_x = solve(pb_x, solver_name="xpress")

    assert r_h.status == SolverStatus.OPTIMAL
    assert r_x.status == SolverStatus.OPTIMAL
    assert r_h.objective is not None and r_x.objective is not None
    assert abs(r_h.objective - r_x.objective) < 1e-6


def test_xpress_env_passthrough() -> None:
    """``env=`` is reserved for future Xpress runtime contexts.

    The ``xpress.problem()`` constructor does not currently accept an
    ``env`` argument the way ``gurobipy.Model(env=...)`` does — but
    the cross-adapter dispatch contract is "pass ``env`` through
    without inspection".  This test confirms that supplying *any*
    placeholder object as ``env`` does not raise.
    """
    placeholder_env = object()
    pb = _toy_lp_problem()
    result = solve(pb, solver_name="xpress", env=placeholder_env)
    assert result.status == SolverStatus.OPTIMAL
    assert result.objective is not None
    assert abs(result.objective - 6500.0) < 1e-6


def test_xpress_ranged_constraint_native() -> None:
    """A ranged constraint flows through Xpress' native range API.

    This test locks the Xpress range convention documented in
    ``_xpress.py``'s module docstring: ``rhs[i] = row_ub`` and
    ``rng[i] = abs(row_ub - row_lb)`` (Xpress ignores the sign of
    rng).  A ``5 <= x + y <= 10`` constraint with ``max x + y`` must
    hit the upper bound (10.0), and the solver result must carry a
    single constraint row (not two halves like
    ``LpView.split_ranged_rows()`` would produce).
    """
    view = _hand_built_ranged_view()
    result = xpress_run(view)

    assert result.status == SolverStatus.OPTIMAL
    assert result.objective is not None
    # Maximising x + y with 5 <= x + y <= 10 hits the upper face.
    assert abs(result.objective - 10.0) < 1e-6
    assert result.primal is not None
    assert abs(result.primal["x"] + result.primal["y"] - 10.0) < 1e-6
    # Solution lies in the feasible band [5, 10].
    assert 5.0 - 1e-6 <= result.primal["x"] + result.primal["y"] <= 10.0 + 1e-6
    # Native range -> exactly one constraint row, not the lo/hi pair
    # that ``LpView.split_ranged_rows()`` would produce.
    assert result.dual is not None
    assert list(result.dual.keys()) == ["ranged"]


# ---------------------------------------------------------------------------
# Manual licence-error verification (NOT automated)
# ---------------------------------------------------------------------------
# The licence-error path in ``_xpress.run`` translates any caught
# ``xpress.SolverError`` / ``xpress.ModelError`` / ``xpress.InterfaceError``
# whose message contains one of {"license", "licence", "oem", "expired",
# "no token"} into ``polar_high.solvers.LicenseError`` with an
# actionable message.  We deliberately do NOT automate this: triggering
# it requires physically removing or corrupting a real Xpress licence
# file (``xpauth.xpr``), which is destructive to the developer's setup
# and inappropriate for CI.
#
# Manual reproduction procedure (run on a machine with a real Xpress
# install — note that the bundled FICO Community licence will *not*
# trigger a licence error; you need a full / expired licence file):
#
#   1.  Locate the active licence:
#         echo $XPAUTH_PATH       # or check the install dir for
#                                 # xpauth.xpr
#   2.  Rename/move it temporarily:
#         mv <path>/xpauth.xpr <path>/xpauth.xpr.bak
#   3.  In a *fresh* Python session (so xpress.init has not yet been
#       called), run:
#
#         from polar_high import Problem
#         from polar_high.solvers import solve, LicenseError
#         pb = Problem()
#         # ... build a tiny problem ...
#         try:
#             solve(pb, solver_name="xpress")
#         except LicenseError as e:
#             print("Got LicenseError as expected:", e)
#
#   4.  Confirm that the printed message:
#         * is a polar_high LicenseError, not a raw
#           xpress.SolverError / xpress.ModelError
#         * mentions ``xpauth.xpr`` / ``XPRESS_LICENSE`` /
#           ``XPAUTH_PATH`` or the reserved ``env=`` passthrough.
#   5.  Restore the licence:
#         mv <path>/xpauth.xpr.bak <path>/xpauth.xpr
#
# If the test ever exits step 3 with a raw ``xpress.*Error`` reaching
# the caller, that is a Phase 8 regression — the adapter must wrap
# every vendor exception it catches.
