"""Phase 6 tests for the COPT direct adapter.

The whole module is skipped at import time when ``coptpy`` is not
installed — the ``[copt]`` optional extra is not present in CI.

License-error translation (``CoptError`` errno in the provisional
license-code list, or message containing a licence keyword, ->
:class:`LicenseError`) is *manual-only* per the implementation plan
(it requires physically removing or corrupting a real licence file);
the procedure is documented at the bottom of this file.
"""

from __future__ import annotations

import numpy as np
import pytest

# Module-level skip — collected as one "skipped" entry rather than per
# test.  Keep the imports inside the skip check so the module body never
# touches coptpy when it isn't installed.
cp = pytest.importorskip("coptpy")
pytest.importorskip("scipy")

from toy_data import make_toy_data  # noqa: E402
from toy_model import build_dispatch  # noqa: E402

from polar_high import Problem  # noqa: E402
from polar_high.solvers import SolverResult, SolverStatus, solve  # noqa: E402
from polar_high.solvers._copt import run as _copt_run  # noqa: E402
from polar_high.solvers._lp_view import LpView  # noqa: E402


def _toy_lp_problem() -> Problem:
    pb = Problem()
    build_dispatch(pb, make_toy_data())
    return pb


def _toy_mip_view() -> LpView:
    """Hand-built MIP :class:`LpView`.

    ``Problem.add_var`` requires ``dims`` and ``index`` arguments, which
    makes single-scalar-MIP test setup awkward to build via the public
    API; we therefore construct the :class:`LpView` directly.

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


def test_copt_solves_toy_lp_returns_optimal() -> None:
    """Toy LP solves to optimal and returns a fully-populated result."""
    pb = _toy_lp_problem()
    result = solve(pb, solver_name="copt")

    assert isinstance(result, SolverResult)
    assert result.solver_name == "copt"
    assert result.status == SolverStatus.OPTIMAL
    assert result.objective is not None
    assert abs(result.objective - 6500.0) < 1e-6
    assert result.primal is not None and len(result.primal) > 0
    # LP -> dual must be populated.
    assert result.dual is not None and len(result.dual) > 0


def test_copt_solves_toy_mip_respects_integrality() -> None:
    """MIP variables come back integral and dual is None."""
    view = _toy_mip_view()
    result = _copt_run(view)

    assert result.status == SolverStatus.OPTIMAL
    assert result.objective is not None
    assert abs(result.objective - 12.0) < 1e-6
    assert result.primal is not None
    for nm, val in result.primal.items():
        assert abs(val - round(val)) < 1e-6, f"{nm}={val} not integral"
    # MIP -> dual must be None (HiGHS / Gurobi adapters do the same).
    assert result.dual is None


def test_copt_objective_matches_highs() -> None:
    """Same LP solved by both backends agrees within 1e-6."""
    pb_h = _toy_lp_problem()
    pb_c = _toy_lp_problem()

    r_h = solve(pb_h, solver_name="highs")
    r_c = solve(pb_c, solver_name="copt")

    assert r_h.status == SolverStatus.OPTIMAL
    assert r_c.status == SolverStatus.OPTIMAL
    assert r_h.objective is not None and r_c.objective is not None
    assert abs(r_h.objective - r_c.objective) < 1e-6


def test_copt_env_passthrough() -> None:
    """A pre-built ``coptpy.Envr`` is honoured by the adapter.

    We construct an Envr, pass it through ``solve(..., env=env)``, and
    assert the solve succeeds.  COPT's Envr does not expose a generic
    ``setParam`` for env-level options the way Gurobi does — most
    parameters live on the Model — so the strongest contract we can
    assert here is "the adapter uses the env we gave it without
    crashing, and produces the expected objective."
    """
    env = cp.Envr()
    try:
        pb = _toy_lp_problem()
        result = solve(pb, solver_name="copt", env=env)
        assert result.status == SolverStatus.OPTIMAL
        assert result.objective is not None
        assert abs(result.objective - 6500.0) < 1e-6
    finally:
        # COPT envs don't strictly need explicit disposal, but mirror
        # the Gurobi test's try/finally hygiene.
        if hasattr(env, "close"):
            env.close()


# ---------------------------------------------------------------------------
# Manual licence-error verification (NOT automated)
# ---------------------------------------------------------------------------
# The licence-error path in ``_copt.run`` translates ``CoptError`` whose
# errno is in ``_LICENSE_ERROR_CODES`` (provisional list documented
# inline) OR whose message contains one of {"license", "licence",
# "no token", "expired", "wls"} into
# ``polar_high.solvers.LicenseError`` with an actionable message.  We
# deliberately do NOT automate this: triggering it requires physically
# removing or corrupting a real COPT licence file, which is destructive
# to the developer's setup and inappropriate for CI.
#
# Manual reproduction procedure (run on a machine with a working COPT
# install):
#
#   1.  Locate the active licence:
#         echo $COPT_LICENSE_DIR     # or check the COPT install dir
#   2.  Rename/move it temporarily:
#         mv $COPT_LICENSE_DIR/copt.lic $COPT_LICENSE_DIR/copt.lic.bak
#   3.  In a Python session, run:
#
#         from polar_high import Problem
#         from polar_high.solvers import solve, LicenseError
#         pb = Problem()
#         pb.add_var("x", lower=0.0)
#         pb.set_objective(pb._vars["x"], sense="min")
#         try:
#             solve(pb, solver_name="copt")
#         except LicenseError as e:
#             print("Got LicenseError as expected:", e)
#
#   4.  Confirm that the printed message:
#         * is a polar_high LicenseError, not a raw CoptError
#         * includes the errno
#         * mentions ``COPT_LICENSE_DIR`` / ``copt.lic`` / the ``env=``
#           passthrough.
#   5.  Restore the licence:
#         mv $COPT_LICENSE_DIR/copt.lic.bak $COPT_LICENSE_DIR/copt.lic
#
# If the test ever exits step 3 with a raw ``CoptError`` reaching the
# caller, that is a Phase 6 regression — the adapter must wrap every
# ``CoptError`` it catches.
#
# While doing the manual verification, also confirm the provisional
# errno list in ``_copt._LICENSE_ERROR_CODES`` against the live
# ``coptpy`` retcode table — if the codes you observe differ, please
# tighten the list.
